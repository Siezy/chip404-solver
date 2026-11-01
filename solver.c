
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#define ROL(x,n) (((x)<<(n))|((x)>>(64-(n))))
static const uint64_t RC[24]={
0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL};
static const int rot[25]={0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14};
static void keccakf(uint64_t st[25]){
    for(int round=0;round<24;round++){
        uint64_t bc[5], t;
        for(int i=0;i<5;i++) bc[i]=st[i]^st[i+5]^st[i+10]^st[i+15]^st[i+20];
        for(int i=0;i<5;i++){
            t=bc[(i+4)%5]^ROL(bc[(i+1)%5],1);
            for(int j=0;j<25;j+=5) st[j+i]^=t;
        }
        uint64_t b[25];
        for(int y=0;y<5;y++)for(int x=0;x<5;x++){
            int nx=y, ny=(2*x+3*y)%5;
            b[nx+5*ny]=ROL(st[x+5*y],rot[x+5*y]);
        }
        for(int y=0;y<5;y++)for(int x=0;x<5;x++)
            st[x+5*y]=b[x+5*y]^((~b[(x+1)%5+5*y])&b[(x+2)%5+5*y]);
        st[0]^=RC[round];
    }
}
// keccak256 legacy (0x01 padding) for len<=135, out 32 bytes
static void k256(const uint8_t*msg,int len,uint8_t out[32]){
    uint8_t blk[136]; memset(blk,0,136);
    memcpy(blk,msg,len); blk[len]=0x01; blk[135]=0x80;
    uint64_t st[25]; memset(st,0,200);
    for(int i=0;i<17;i++){uint64_t v;memcpy(&v,blk+i*8,8);st[i]=v;}
    keccakf(st);
    for(int i=0;i<4;i++) memcpy(out+i*8,&st[i],8);
}
int main(int argc,char**argv){
    // mode test: argv[1]=="test" -> hash argv[2] (hex msg), print digest; else mine
    if(argc>=3 && strcmp(argv[1],"test")==0){
        int L=strlen(argv[2])/2;
        uint8_t msg[136], out[32];
        for(int i=0;i<L;i++){unsigned v;sscanf(argv[2]+2*i,"%2x",&v);msg[i]=v;}
        k256(msg,L,out);
        for(int i=0;i<32;i++) printf("%02x",out[i]); printf("\n");
        return 0;
    }
    if(argc<5){ printf("usage: %s mine <salthex> <addrhex> <start>\n",argv[0]); return 1; }
    const char* salthex=argv[2]; const char* addrhex=argv[3];
    uint64_t start=strtoull(argv[4],0,10);
    uint8_t blk[136]; memset(blk,0,136);
    for(int i=0;i<32;i++){unsigned v;sscanf(salthex+2*i,"%2x",&v);blk[i]=v;}
    for(int i=0;i<20;i++){unsigned v;sscanf(addrhex+2*i,"%2x",&v);blk[32+i]=v;}
    blk[84]=0x01; blk[135]=0x80;
    uint64_t W[17]; for(int i=0;i<17;i++){uint64_t v;memcpy(&v,blk+i*8,8);W[i]=v;}
    uint64_t target_hi=0x0000000007ffffffffULL;
    uint64_t target_lo=0xffffffffffffffffULL;
    uint64_t be = __builtin_bswap64(start); // bytes: BE of start (only low 8 bytes vary; bytes 52..75 stay zero)
    uint64_t n=start, cnt=0;
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    while(1){
        uint64_t st[25];
        for(int i=0;i<17;i++) st[i]=W[i];
        // set nonce last-8-bytes at bytes 76..83: word9 high 4 bytes stay from W[9] (bytes76..79 = bytes 4..7 of word9) 
        // Simpler: rebuild blk each iter is slow; do targeted: bytes76..83 -> st[9] bytes4..7 (LE lane: bytes76..79 are HIGH half of st[9]) and st[10] bytes0..3 (LOW half of st[10])
        uint64_t lo32 = be & 0xFFFFFFFFULL;        // bytes 79..76 in BE order? Careful:
        // memory bytes76..83 = [b76,b77,...,b83]; be stored LE gives b76 = lowest byte of be.
        // We want b76..b83 = big-endian bytes of start => b76 = MSB of start.
        // be = bswap64(start) so its LE memory layout = big-endian bytes of start. memcpy(&buf,&be,8) gives bytes BE(start). 
        // word9 = bytes 72..79 => bytes76..79 = top 4 bytes of be's memory = 4 MSB bytes of start.
        // st[9] = W[9] with top 4 bytes replaced: (W[9] & 0x00000000FFFFFFFF) | (top4 << 32)
        uint64_t mem_be[2]; mem_be[0]=be; // just use array indexing
        uint8_t*beb=(uint8_t*)mem_be;
        uint64_t top4; memcpy(&top4, beb, 4);        // LE load of first 4 mem bytes = b76..b79
        uint64_t bot4; memcpy(&bot4, beb+4, 4);      // b80..b83
        st[9] = (W[9] & 0x00000000FFFFFFFFULL) | (top4 << 32);
        st[10] = (W[10] & 0xFFFFFFFF00000000ULL) | bot4;
        keccakf(st);
        uint64_t hi; memcpy(&hi,&st[0],8); hi=__builtin_bswap64(hi);
        if(hi<target_hi){
            uint64_t outn[2]; outn[0]=n; outn[1]=0;
            fwrite(outn,8,1,stdout); fflush(stdout);
            return 0;
        }
        n++; cnt++;
        if((cnt & 0xFFFFF)==0){
            clock_gettime(CLOCK_MONOTONIC,&t1);
            double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
            if(el>3600*10) break; // 10h cap
        }
    }
    return 2;
}
