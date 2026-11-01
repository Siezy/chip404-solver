// chip404_avx512: AVX-512 batch Keccak-256 nonce search for Chip404.
// Adapted from tailfast3.c (Boltbook Pod AVX-512 batch Keccak, 8 hashes per
// permutation). Chip404 input is 84 bytes: salt(32) + address(20) + nonce
// (32-byte big-endian uint256). Pass when hash <= targetNow (weight 34:
// hash < 2^222, i.e. lane0 low 32 bits == 0 AND bits 38-39 == 0).
// Usage: chip404_avx512 [start_nonce] [nthreads]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>

static const uint64_t RC[24] = {
  0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
  0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
  0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
  0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
  0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
  0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL};
static const int ROTC[24] = {1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
static const int PILN[24] = {10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
static inline uint64_t rol64(uint64_t x, int n){ return (x<<n)|(x>>(64-n)); }

/* ---- scalar Keccak-f1600 + Keccak-256 (reference / verification) ---- */
static void keccakf(uint64_t a[25]){
  for(int r=0;r<24;r++){
    uint64_t c[5], t;
    for(int i=0;i<5;i++) c[i]=a[i]^a[i+5]^a[i+10]^a[i+15]^a[i+20];
    for(int i=0;i<5;i++){ t=c[(i+4)%5]^rol64(c[(i+1)%5],1); for(int j=0;j<25;j+=5) a[j+i]^=t; }
    t=a[1];
    for(int i=0;i<24;i++){ int j=PILN[i]; uint64_t tmp=a[j]; a[j]=rol64(t,ROTC[i]); t=tmp; }
    for(int j=0;j<25;j+=5){ uint64_t bc[5]; for(int i=0;i<5;i++) bc[i]=a[j+i];
      for(int i=0;i<5;i++) a[j+i]^=(~bc[(i+1)%5])&bc[(i+2)%5]; }
    a[0]^=RC[r];
  }
}
static void keccak256(const uint8_t* msg, size_t len, uint8_t out[32]){
  uint64_t a[25]={0};
  uint8_t block[136]={0};
  size_t off=0;
  while(len-off>=136){ for(int i=0;i<17;i++){ uint64_t w; memcpy(&w,msg+off+8*i,8); a[i]^=w; } keccakf(a); off+=136; }
  size_t rem=len-off;
  memcpy(block,msg+off,rem); block[rem]=0x01; block[135]=0x80;
  for(int i=0;i<17;i++){ uint64_t w; memcpy(&w,block+8*i,8); a[i]^=w; }
  keccakf(a);
  for(int i=0;i<4;i++){ uint64_t w=a[i]; memcpy(out+8*i,&w,8); }
}

/* ---- AVX-512 batch Keccak-f1600 (from tailfast3.c) ---- */
static void keccakf512(__m512i a[25]){
  for(int r=0;r<24;r++){
    __m512i c[5], t;
    for(int i=0;i<5;i++) c[i]=_mm512_xor_epi64(_mm512_xor_epi64(_mm512_xor_epi64(a[i],a[i+5]),_mm512_xor_epi64(a[i+10],a[i+15])),a[i+20]);
    for(int i=0;i<5;i++){ t=_mm512_xor_epi64(c[(i+4)%5],_mm512_rol_epi64(c[(i+1)%5],1)); for(int j=0;j<25;j+=5) a[j+i]=_mm512_xor_epi64(a[j+i],t); }
    t=a[1];
    { __m512i tmp;
      tmp=a[10]; a[10]=_mm512_rol_epi64(t,1); t=tmp;
      tmp=a[7]; a[7]=_mm512_rol_epi64(t,3); t=tmp;
      tmp=a[11]; a[11]=_mm512_rol_epi64(t,6); t=tmp;
      tmp=a[17]; a[17]=_mm512_rol_epi64(t,10); t=tmp;
      tmp=a[18]; a[18]=_mm512_rol_epi64(t,15); t=tmp;
      tmp=a[3]; a[3]=_mm512_rol_epi64(t,21); t=tmp;
      tmp=a[5]; a[5]=_mm512_rol_epi64(t,28); t=tmp;
      tmp=a[16]; a[16]=_mm512_rol_epi64(t,36); t=tmp;
      tmp=a[8]; a[8]=_mm512_rol_epi64(t,45); t=tmp;
      tmp=a[21]; a[21]=_mm512_rol_epi64(t,55); t=tmp;
      tmp=a[24]; a[24]=_mm512_rol_epi64(t,2); t=tmp;
      tmp=a[4]; a[4]=_mm512_rol_epi64(t,14); t=tmp;
      tmp=a[15]; a[15]=_mm512_rol_epi64(t,27); t=tmp;
      tmp=a[23]; a[23]=_mm512_rol_epi64(t,41); t=tmp;
      tmp=a[19]; a[19]=_mm512_rol_epi64(t,56); t=tmp;
      tmp=a[13]; a[13]=_mm512_rol_epi64(t,8); t=tmp;
      tmp=a[12]; a[12]=_mm512_rol_epi64(t,25); t=tmp;
      tmp=a[2]; a[2]=_mm512_rol_epi64(t,43); t=tmp;
      tmp=a[20]; a[20]=_mm512_rol_epi64(t,62); t=tmp;
      tmp=a[14]; a[14]=_mm512_rol_epi64(t,18); t=tmp;
      tmp=a[22]; a[22]=_mm512_rol_epi64(t,39); t=tmp;
      tmp=a[9]; a[9]=_mm512_rol_epi64(t,61); t=tmp;
      tmp=a[6]; a[6]=_mm512_rol_epi64(t,20); t=tmp;
      tmp=a[1]; a[1]=_mm512_rol_epi64(t,44); t=tmp;
    }
    for(int j=0;j<25;j+=5){ __m512i bc[5]; for(int i=0;i<5;i++) bc[i]=a[j+i];
      for(int i=0;i<5;i++) a[j+i]=_mm512_xor_epi64(a[j+i],_mm512_andnot_epi64(bc[(i+1)%5],bc[(i+2)%5])); }
    a[0]=_mm512_xor_epi64(a[0],_mm512_set1_epi64((long long)RC[r]));
  }
}

/* ---- Chip404 ---- */
static const uint8_t SALT[32] = {
    0x44,0x5a,0xa0,0x83,0x84,0x38,0xa3,0x76,0x56,0x3c,0x29,0x0a,0x2c,0x15,0xcf,0x7d,
    0xde,0xa0,0x88,0x38,0x54,0x55,0x5b,0x68,0xb0,0xe0,0x1e,0xfe,0xfa,0xff,0xf3,0x9a
};
static const uint8_t ADDR[20] = {
    0xBC,0x4D,0xd3,0x03,0x2b,0x7C,0xfa,0x8a,0xF0,0x5c,0x15,0x7C,0xaF,0xd5,0xdd,0xe8,
    0xf1,0x2b,0x4d,0x06
};
// Broadcast (lane-invariant) block words for the 84-byte input.
// Word w = little-endian 64-bit of input bytes [8w, 8w+8).
// Words 0-6: salt+addr+nonce-high-zeros. Words 7-8: zeros. Word 10: low nonce
// bytes + 0x01 padding (handled per-lane). Words 11-15: zeros.
// Word 16: 0x80 at byte 135.
static uint64_t BW[17];
static void init_broadcast_words(void){
    uint8_t in[136];
    memset(in, 0, sizeof(in));
    memcpy(in, SALT, 32);
    memcpy(in+32, ADDR, 20);
    // nonce bytes are zero in the template; words 9,10 are per-lane.
    in[84] = 0x01;
    in[135] = 0x80;
    for(int i=0;i<17;i++){
        uint64_t w=0;
        for(int j=0;j<8;j++) w |= ((uint64_t)in[i*8+j]) << (8*j);
        BW[i]=w;
    }
}
// Build the 84-byte input for a given nonce (uint64 -> 32-byte BE at bytes 52-83).
static void build_input(uint8_t in[84], uint64_t nonce){
    memcpy(in, SALT, 32);
    memcpy(in+32, ADDR, 20);
    for(int i=0;i<24;i++) in[52+i]=0;
    for(int i=0;i<8;i++) in[52+24+i]=(uint8_t)(nonce >> (56-8*i));
}
// Pass check: hash < 2^222  <=>  lane0 low 32 bits == 0 AND bits 38-39 == 0.
static inline int passes(uint64_t lane0){
    return (lane0 & 0xFFFFFFFFFFULL)==0; /* target 2^216: top 40 bits of hash zero */
}

static volatile int FOUND=0;
static volatile uint64_t FOUND_NONCE=0;
static volatile unsigned long long TRIED=0;
static int NTHREADS=4;
static uint64_t START=0;

// One batch of 8 nonces. Returns 1 if a verified hit.
// Words 9 and 10 are the only nonce-dependent block words; computed directly
// from the nonce (no 84-byte input buffer) to avoid memory traffic:
//   word 9  = (nonce >> 32) << 32   (nonce BE bytes 0-3 in bits 32-63)
//   word 10 = bswap32(nonce low 32) | (0x01 << 32)  (BE bytes 4-7 + padding)
static int batch8(uint64_t base_nonce){
    __m512i a[25];
    uint64_t w9[8], w10[8];
    for(int j=0;j<8;j++){
        uint64_t n = base_nonce + j;
        w9[j]  = (n >> 32) << 32;
        w10[j] = (uint64_t)__builtin_bswap32((uint32_t)n) | (0x01ULL << 32);
    }
    for(int w=0;w<17;w++){
        if(w==9) a[w]=_mm512_loadu_si512(w9);
        else if(w==10) a[w]=_mm512_loadu_si512(w10);
        else a[w]=_mm512_set1_epi64((long long)BW[w]);
    }
    for(int w=17;w<25;w++) a[w]=_mm512_setzero_si512();
    keccakf512(a);
    uint64_t lane0[8];
    _mm512_storeu_si512(lane0, a[0]);
    for(int j=0;j<8;j++){
        if(passes(lane0[j])){
            // scalar re-verify
            uint8_t in[84];
            build_input(in, base_nonce + j);
            uint8_t h[32];
            keccak256(in, 84, h);
            uint64_t l0=0;
            memcpy(&l0, h, 8);
            if(passes(l0)){
                FOUND_NONCE = base_nonce + j;
                return 1;
            }
        }
    }
    return 0;
}

static void* search_thread(void* arg){
    long tid=(long)arg;
    uint64_t nonce = START + (uint64_t)tid*8;
    uint64_t stride = (uint64_t)8 * NTHREADS;
    while(!FOUND){
        if(batch8(nonce)){ FOUND=1; break; }
        nonce += stride;
        __sync_fetch_and_add(&TRIED, 8);
    }
    return NULL;
}

int main(int argc, char** argv){
    if(argc>2 && strcmp(argv[1],"check")==0){
        init_broadcast_words();
        uint64_t n=strtoull(argv[2],0,10);
        uint8_t in[84], h[32]; char hex[65];
        build_input(in,n); keccak256(in,84,h);
        for(int i=0;i<32;i++) sprintf(hex+2*i,"%02x",h[i]);
        printf("%s\n",hex); return 0;
    }
    if(argc>1) START=strtoull(argv[1], NULL, 10);
    if(argc>2) NTHREADS=atoi(argv[2]);
    init_broadcast_words();
    // scalar self-test vs known vectors
    uint8_t h[32]; char hex[65];
    keccak256((uint8_t*)"",0,h); for(int i=0;i<32;i++) sprintf(hex+2*i,"%02x",h[i]);
    if(strcmp(hex,"c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")){ fprintf(stderr,"SELFTEST FAIL (empty)\n"); return 3; }
    keccak256((uint8_t*)"abc",3,h); for(int i=0;i<32;i++) sprintf(hex+2*i,"%02x",h[i]);
    if(strcmp(hex,"4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45")){ fprintf(stderr,"SELFTEST FAIL (abc)\n"); return 3; }
    // batch self-test: AVX-512 vs scalar for nonces 0..7 (direct word computation)
    {
        __m512i a[25];
        uint64_t w9[8], w10[8];
        for(int j=0;j<8;j++){
            uint64_t n = (uint64_t)j;
            w9[j]  = (n >> 32) << 32;
            w10[j] = (uint64_t)__builtin_bswap32((uint32_t)n) | (0x01ULL<<32);
        }
        for(int w=0;w<17;w++){
            if(w==9) a[w]=_mm512_loadu_si512(w9);
            else if(w==10) a[w]=_mm512_loadu_si512(w10);
            else a[w]=_mm512_set1_epi64((long long)BW[w]);
        }
        for(int w=17;w<25;w++) a[w]=_mm512_setzero_si512();
        keccakf512(a);
        uint64_t lane0[8]; _mm512_storeu_si512(lane0, a[0]);
        for(int j=0;j<8;j++){
            uint8_t in[84]; build_input(in, j);
            uint8_t hv[32]; keccak256(in, 84, hv);
            uint64_t l0=0; memcpy(&l0, hv, 8);
            if(l0 != lane0[j]){ fprintf(stderr,"SELFTEST FAIL (batch lane %d): scalar %016llx vs avx512 %016llx\n", j, (unsigned long long)l0, (unsigned long long)lane0[j]); return 3; }
        }
    }
    printf("selftest ok (scalar + AVX-512 batch)\n"); fflush(stdout);
    pthread_t* th=malloc(sizeof(pthread_t)*NTHREADS);
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
    for(long i=0;i<NTHREADS;i++) pthread_create(&th[i],NULL,search_thread,(void*)i);
    while(!FOUND){
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
        double el=(ts.tv_sec-t0.tv_sec)+(ts.tv_nsec-t0.tv_nsec)/1e9;
        if(el>=5.0){
            unsigned long long tr=TRIED;
            printf("\r%.1fM tried, %.0fs, %.2f M/s", tr/1e6, el, tr/el/1e6);
            fflush(stdout);
            for(int k=0;k<50 && !FOUND;k++){ struct timespec s={0,100000000}; nanosleep(&s,NULL); }
            if(el>7200){ printf("\nTimeout 2h.\n"); break; }
        } else { struct timespec s={0,100000000}; nanosleep(&s,NULL); }
    }
    for(int i=0;i<NTHREADS;i++) pthread_join(th[i],NULL);
    if(FOUND) printf("\nFOUND nonce=%llu\n", (unsigned long long)FOUND_NONCE);
    else printf("\nNOT FOUND\n");
    return FOUND?0:1;
}
