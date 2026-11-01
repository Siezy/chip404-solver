
// chip404_t4.cu - Keccak-256 nonce search for Chip404, optimized for Tesla T4 (sm_75)
// Build: nvcc -O3 -arch=sm_75 -o chip404_t4 chip404_t4.cu
// Usage: ./chip404_t4 <start_nonce> [blocks] [threads] [seconds_limit]
// Pass: keccak256(salt32|addr20|24zero|nonce8BE) < 2^213  (weight 43, top 43 bits zero)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>

__constant__ uint64_t d_RC[24];
__constant__ uint64_t d_SALTADDR[7]; // salt(32)+addr(20) = 52 bytes = 6.5 words; use 7 words (words 0..6), word6 partial
// Layout: message 84B -> 10.5 lanes; we build 17 lanes of input block.
// words: w0..w3 = salt, w4 = salt[28..31]|addr[0..3], w5 = addr[4..11], w6 = addr[12..19]|zero, w7,w8 = 0,
// w9 = nonce bytes 0..7 (BE) at bytes 72..79 -> but bytes 72..75 = 0 (part of 24 zeros) and 76..79 = nonce[0..3]
// w10 = nonce[4..7] at bytes 80..83 + 0x01 at byte 84
// w11..w15 = 0, w16 = 0x80 at byte 135
// All as LITTLE-ENDIAN uint64 lanes.

static const uint64_t h_RC[24] = {
  0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
  0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
  0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
  0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
  0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
  0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL};

__device__ __forceinline__ uint64_t rol64(uint64_t x, int n){ return (x<<n)|(x>>(64-n)); }


__device__ __forceinline__ void keccakf1600(uint64_t *s){
  #pragma unroll
  for(int r=0;r<24;r++){
    uint64_t c0=s[0]^s[5]^s[10]^s[15]^s[20];
    uint64_t c1=s[1]^s[6]^s[11]^s[16]^s[21];
    uint64_t c2=s[2]^s[7]^s[12]^s[17]^s[22];
    uint64_t c3=s[3]^s[8]^s[13]^s[18]^s[23];
    uint64_t c4=s[4]^s[9]^s[14]^s[19]^s[24];
    uint64_t d0=rol64(c1,1)^c4;
    uint64_t d1=rol64(c2,1)^c0;
    uint64_t d2=rol64(c3,1)^c1;
    uint64_t d3=rol64(c4,1)^c2;
    uint64_t d4=rol64(c0,1)^c3;
    s[0]^=d0; s[5]^=d0; s[10]^=d0; s[15]^=d0; s[20]^=d0;
    s[1]^=d1; s[6]^=d1; s[11]^=d1; s[16]^=d1; s[21]^=d1;
    s[2]^=d2; s[7]^=d2; s[12]^=d2; s[17]^=d2; s[22]^=d2;
    s[3]^=d3; s[8]^=d3; s[13]^=d3; s[18]^=d3; s[23]^=d3;
    s[4]^=d4; s[9]^=d4; s[14]^=d4; s[19]^=d4; s[24]^=d4;
    uint64_t t;
    t = s[1];
    s[1]  = rol64(s[6], 44); s[6]  = rol64(s[9], 20); s[9]  = rol64(s[22],61);
    s[22] = rol64(s[14],39); s[14] = rol64(s[20],18); s[20] = rol64(s[2], 62);
    s[2]  = rol64(s[12],43); s[12] = rol64(s[13],25); s[13] = rol64(s[19], 8);
    s[19] = rol64(s[23],56); s[23] = rol64(s[15],41); s[15] = rol64(s[4], 27);
    s[4]  = rol64(s[24],14); s[24] = rol64(s[21], 2); s[21] = rol64(s[8], 55);
    s[8]  = rol64(s[16],45); s[16] = rol64(s[5], 36); s[5]  = rol64(s[3], 28);
    s[3]  = rol64(s[18],21); s[18] = rol64(s[17],10); s[17] = rol64(s[11], 6);
    s[11] = rol64(s[7],  3); s[7]  = rol64(t,     1);
    #pragma unroll
    for(int j=0;j<25;j+=5){
      uint64_t a0=s[j],a1=s[j+1],a2=s[j+2],a3=s[j+3],a4=s[j+4];
      s[j]   ^= (~a1)&a2;
      s[j+1] ^= (~a2)&a3;
      s[j+2] ^= (~a3)&a4;
      s[j+3] ^= (~a4)&a0;
      s[j+4] ^= (~a0)&a1;
    }
    s[0]^=d_RC[r];
  }
}

// each thread mines NONCES_PER_THREAD sequential nonces

__global__ void selftest_kernel(uint64_t nonce, uint64_t *out_hash){
  uint64_t s[25];
  uint8_t n0=(uint8_t)(nonce>>56),n1=(uint8_t)(nonce>>48),n2=(uint8_t)(nonce>>40),n3=(uint8_t)(nonce>>32);
  uint8_t n4=(uint8_t)(nonce>>24),n5=(uint8_t)(nonce>>16),n6=(uint8_t)(nonce>>8),n7=(uint8_t)(nonce);
  for(int w=0;w<7;w++) s[w]=d_SALTADDR[w];
  s[7]=0; s[8]=0;
  s[9]=((uint64_t)n0<<32)|((uint64_t)n1<<40)|((uint64_t)n2<<48)|((uint64_t)n3<<56);
  s[10]=(uint64_t)n4|((uint64_t)n5<<8)|((uint64_t)n6<<16)|((uint64_t)n7<<24)|(0x01ULL<<32);
  s[11]=0;s[12]=0;s[13]=0;s[14]=0;s[15]=0;s[16]=0x8000000000000000ULL;
  s[17]=0;s[18]=0;s[19]=0;s[20]=0;s[21]=0;s[22]=0;s[23]=0;s[24]=0;
  keccakf1600(s);
  for(int i=0;i<4;i++) out_hash[i]=s[i];
}
__global__ void mine_kernel(uint64_t start_nonce, unsigned long long *result, uint64_t *result_nonce,
                            unsigned long long *hashes_done, int nonces_per_thread){
  uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t nonce = start_nonce + idx * (uint64_t)nonces_per_thread;
  uint64_t found = 0; uint64_t found_nonce = 0;
  uint64_t s[25];
  #pragma unroll 1
  for(int k=0;k<nonces_per_thread;k++){
    // build block lanes (little-endian uint64 of message bytes)
    // w0..w3 salt; w4 = salt last 4 bytes + addr first 4; w5 = addr 4..11; w6 = addr last 8;
    // careful: constant table approach - load from constant memory
    uint64_t base0 = d_SALTADDR[0], base1 = d_SALTADDR[1], base2 = d_SALTADDR[2], base3 = d_SALTADDR[3];
    uint64_t base4 = d_SALTADDR[4], base5 = d_SALTADDR[5], base6 = d_SALTADDR[6];
    s[0]=base0; s[1]=base1; s[2]=base2; s[3]=base3; s[4]=base4; s[5]=base5; s[6]=base6;
    s[7]=0; s[8]=0;
    // bytes 72..79 = w9 = BE bytes of (nonce>>24)? Let's do byte-accurate:
    // message bytes 72..75 are zero (24 zeros span 52..75), bytes 76..83 = BE nonce (8 bytes)
    // w9 = LE u64 of bytes 72..79 = bytes[72..75]=0 | bytes[76..79]=nonce[0..3] (BE) -> as LE uint64:
    s[9] = (uint64_t)((uint32_t)(nonce >> 24)); // bytes 76..79 = (n>>24) BE? need byte layout check below
    // Simpler & correct: compute via byte assembly for word9/word10:
    uint8_t n0 = (uint8_t)(nonce>>56), n1=(uint8_t)(nonce>>48), n2=(uint8_t)(nonce>>40), n3=(uint8_t)(nonce>>32);
    uint8_t n4 = (uint8_t)(nonce>>24), n5=(uint8_t)(nonce>>16), n6=(uint8_t)(nonce>>8),  n7=(uint8_t)(nonce);
    // bytes 72..75 = 0 (zero region), bytes 76..79 = n0..n3 (BE) => LE lane:
    s[9] = ((uint64_t)n0<<32) | ((uint64_t)n1<<40) | ((uint64_t)n2<<48) | ((uint64_t)n3<<56);
    s[10] = (uint64_t)n4 | ((uint64_t)n5<<8) | ((uint64_t)n6<<16) | ((uint64_t)n7<<24) | (0x01ULL<<32);
    s[11]=0; s[12]=0; s[13]=0; s[14]=0; s[15]=0; s[16]=0x8000000000000000ULL;
    s[17]=0; s[18]=0; s[19]=0; s[20]=0; s[21]=0; s[22]=0; s[23]=0; s[24]=0;
    keccakf1600(s);
    uint64_t l0 = s[0];
    if((l0 & 0xFFFFFFFFFFULL)==0 && (l0 & 0xE00000000000ULL)==0){
      found = 1; found_nonce = nonce;
      break;
    }
    nonce++;
  }
  if(found){ *result = 1; *result_nonce = found_nonce; }
  atomicAdd(hashes_done, (unsigned long long)nonces_per_thread);
}

#define SALT_HEX "445aa0838438a376563c290a2c15cf7ddea0883854555b68b0e01efefafff39a"
#define ADDR_HEX "BC4Dd3032b7Cfa8aF05c157CaFd5dde8f12b4d06"

static int hexval(char c){ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return -1; }

int main(int argc, char** argv){
  uint64_t start = (argc>1)? strtoull(argv[1],0,10) : 0ULL;
  int blocks = (argc>2)? atoi(argv[2]) : 320;
  int threads = (argc>3)? atoi(argv[3]) : 256;
  int secs = (argc>4)? atoi(argv[4]) : 3000;
  int npt = (argc>5)? atoi(argv[5]) : 1000; // nonces per thread per launch

  cudaMemcpyToSymbol(d_RC, h_RC, sizeof(h_RC));
  // build SALTADDR words: message bytes 0..55 (7 words, word6 = bytes 48..55 -> addr bytes 16..19 + 4 zero bytes)
  uint8_t msg[56]; memset(msg,0,56);
  for(int i=0;i<32;i++) msg[i] = (uint8_t)(hexval(SALT_HEX[2*i])*16 + hexval(SALT_HEX[2*i+1]));
  for(int i=0;i<20;i++) msg[32+i] = (uint8_t)(hexval(ADDR_HEX[2*i])*16 + hexval(ADDR_HEX[2*i+1]));
  uint64_t h_sa[7];
  for(int w=0;w<7;w++){
    uint64_t v=0;
    for(int j=0;j<8;j++) v |= ((uint64_t)msg[w*8+j]) << (8*j);
    h_sa[w]=v;
  }
  cudaMemcpyToSymbol(d_SALTADDR, h_sa, sizeof(h_sa));

  // Instead: verify via known vectors from GPU itself. Expected hash for nonce0 known:
  // 4697b4d517d780dbcc24dc62dd33aff29586543f77bea295ba753a27dd62b805 -> lane0 LE = 0x05b862dd33af9746? compute below with python-generated constant
  // lane0 bytes 0..7 of that hash: 46 97 b4 d5 17 d7 80 db -> LE u64 = 0xdb80d717d5b49746
  const uint64_t EXPECT_LANE0_NONCE0 = 0xdb80d717d5b49746ULL;

  unsigned long long *d_result; uint64_t *d_rn; unsigned long long *d_done;
  cudaMalloc(&d_result,8); cudaMalloc(&d_rn,8); cudaMalloc(&d_done,8);

  // SELFTEST: mine starting at 0 with npt=1: should find nonce 0? No - nonce0 hash fails weight43.
  // Instead run kernel with npt=1, blocks=1, threads=1 and capture hash by temporarily... simplest: add debug kernel? 
  // Pragmatic: run mine for real; the correctness guarantee = the lane assembly was byte-checked + keccak identical to reference C below.
  // Do a host-side reference check of word layout:
  // message for nonce=0: 84 bytes, hash known. We'll trust byte assembly verified by harness on CPU earlier.
  {
    uint64_t *d_h; cudaMalloc(&d_h,32); 
    uint64_t h_h[4];
    selftest_kernel<<<1,1>>>(0ULL, d_h);
    cudaMemcpy(h_h,d_h,32,cudaMemcpyDeviceToHost);
    cudaFree(d_h);
    char hexbuf[65];
    for(int i=0;i<4;i++) sprintf(hexbuf+16*i, "%016llx", (unsigned long long)h_h[i]);
    printf("device hash nonce0 = %s\n", hexbuf);
    if(strcmp(hexbuf, "4697b4d517d780dbcc24dc62dd33aff29586543f77bea295ba753a27dd62b805")==0)
      printf("SELFTEST PASS - safe to mine\n");
    else { printf("SELFTEST FAIL - DO NOT MINE\n"); return 3; }
  }

  unsigned long long zero=0;
  uint64_t total=0;
  time_t t0=time(0);
  uint64_t current = start;
  while(difftime(time(0),t0) < secs){
    cudaMemcpy(d_result,&zero,8,cudaMemcpyHostToDevice);
    cudaMemcpy(d_done,&zero,8,cudaMemcpyHostToDevice);
    mine_kernel<<<blocks,threads>>>(current, d_result, d_rn, d_done, npt);
    cudaDeviceSynchronize();
    unsigned long long done; uint64_t res;
    cudaMemcpy(&done,d_done,8,cudaMemcpyDeviceToHost);
    cudaMemcpy(&res,d_rn,8,cudaMemcpyDeviceToHost);
    unsigned long long found=0;
    cudaMemcpy(&found,d_result,8,cudaMemcpyDeviceToHost);
    total += done;
    double el = difftime(time(0),t0);
    printf("\r%.2f GH/s  total=%.2fG  cur=%llu   ", done/el/1e9, total/1e9, (unsigned long long)current);
    fflush(stdout);
    if(found){ printf("\nFOUND nonce=%llu\n",(unsigned long long)res); return 0; }
    current += (uint64_t)blocks*threads*npt;
  }
  printf("\ntime limit reached\n");
  return 1;
}
