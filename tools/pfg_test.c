/* Standalone A/B harness: coli_cuda_prefill_attn_gemm vs the reference
 * coli_cuda_attention_project_batch_dev_out on random data (no model needed).
 * Build (next to a CUDA=1 build of the engine, for backend_cuda.o):
 *   gcc -O2 -I../c -o pfg_test pfg_test.c ../c/backend_cuda.o \
 *       -lm -L/usr/local/cuda/lib64 -lcudart -lstdc++
 * Usage: ./pfg_test S T [zero_rope] [zero_nope] [magnitude] [sel_topk]
 * Prints per-row max rel error vs the fp32 absorb kernel; expect ~1e-3
 * (fp16 input rounding), flags rows >5e-2.
 * With sel_topk>0: DSA phase-B A/B instead — rows [S/2,S) get a random
 * selection list and the scalar sel-absorb (COLI_DSA_TCGATHER=0) is the
 * reference against the TC gather path (=1), same inputs, same lists. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "backend_cuda.h"

static unsigned rngs=12345;
static float frand(void){ rngs=rngs*1664525u+1013904223u; return ((rngs>>8)&0xFFFF)/65536.0f-0.5f; }

int main(int argc,char**argv){
    int S=argc>1?atoi(argv[1]):341, T=argc>2?atoi(argv[2]):341;
    int zero_rope=argc>3?atoi(argv[3]):0, zero_nope=argc>4?atoi(argv[4]):0;
    int H=64,Q=192,R=64,V=256,K=512,O=6144;
    int dev0[1]={0};
    if(!coli_cuda_init(dev0,1)){ fprintf(stderr,"cuda init failed\n"); return 1; }
    int dev=coli_cuda_device_at(0);
    int kvO=H*(Q+V), oI=H*V;
    size_t kvb_bytes=(size_t)kvO*((K+1)/2), ob=(size_t)O*((oI+1)/2);
    unsigned char *kvw=malloc(kvb_bytes), *ow=malloc(ob);
    float *kvs=malloc(kvO*4), *os=malloc(O*4);
    for(size_t i=0;i<kvb_bytes;i++) kvw[i]=(unsigned char)(rand()&0xFF);
    for(size_t i=0;i<ob;i++) ow[i]=(unsigned char)(rand()&0xFF);
    for(int i=0;i<kvO;i++) kvs[i]=0.02f+0.01f*fabsf(frand());
    for(int i=0;i<O;i++) os[i]=0.02f+0.01f*fabsf(frand());
    ColiCudaTensor *kvt=NULL,*ot=NULL;
    if(!coli_cuda_tensor_upload(&kvt,kvw,kvs,2,K,kvO,dev)||
       !coli_cuda_tensor_upload(&ot,ow,os,2,oI,O,dev)){ fprintf(stderr,"upload failed\n"); return 1; }
    size_t qn=(size_t)S*H*(Q+R), ln=(size_t)T*K, rn=(size_t)T*R;
    float *qh=malloc(qn*4), *lh=malloc(ln*4), *rh=malloc(rn*4);
    { float mag=argc>5?atof(argv[5]):1.0f;
    for(size_t i=0;i<qn;i++) qh[i]=frand()*mag;
    for(size_t i=0;i<ln;i++) lh[i]=frand()*mag;
    for(size_t i=0;i<rn;i++) rh[i]=frand()*mag; }
    if(zero_rope) memset(rh,0,rn*4);
    if(zero_nope){ /* zero q_nope so only the rope term drives scores */
        for(int s=0;s<S;s++)for(int h=0;h<H;h++)
            memset(qh+((size_t)s*H+h)*(Q+R),0,(size_t)Q*4);
    }
    float *qd=coli_cuda_pipe_alloc(dev,qn*4), *ld=coli_cuda_pipe_alloc(dev,ln*4);
    float *rd=coli_cuda_pipe_alloc(dev,rn*4);
    float *o1=coli_cuda_pipe_alloc(dev,(size_t)S*O*4), *o2=coli_cuda_pipe_alloc(dev,(size_t)S*O*4);
    coli_cuda_pipe_upload(dev,qd,qh,qn*4);
    coli_cuda_pipe_upload(dev,ld,lh,ln*4);
    coli_cuda_pipe_upload(dev,rd,rh,rn*4);
    float scale=1.0f/sqrtf((float)(Q+R));
    int sel_topk=argc>6?atoi(argv[6]):0, sB0=S/2;
    int *sel=NULL;
    if(sel_topk>0){
        if(sel_topk>T){ fprintf(stderr,"sel_topk>T\n"); return 1; }
        /* la funzione legge sel_host+sB0*topk: il buffer copre TUTTE le S
         * righe (come m->dsa_sel nel modello), non solo quelle di fase B */
        sel=malloc((size_t)S*sel_topk*sizeof(int));
        for(int s=sB0;s<S;s++)for(int j=0;j<sel_topk;j++){
            /* posizioni ~uniche in [0,T): passo fisso + jitter deterministico */
            long p=((long)j*T)/sel_topk+(int)(frand()*((float)T/sel_topk));
            if(p<0)p=0; if(p>=T)p=T-1; sel[(size_t)s*sel_topk+j]=(int)p;
        }
        fprintf(stderr,"sel A/B: scalar...\n");
        setenv("COLI_DSA_TCGATHER","0",1);
        if(!coli_cuda_prefill_attn_gemm(kvt,ot,o1,qd,ld,rd,S,H,Q,R,V,K,T,scale,sel,sB0,sel_topk)){
            fprintf(stderr,"scalar sel path failed\n"); return 1; }
        fprintf(stderr,"sel A/B: tc gather...\n");
        setenv("COLI_DSA_TCGATHER","1",1);
        if(!coli_cuda_prefill_attn_gemm(kvt,ot,o2,qd,ld,rd,S,H,Q,R,V,K,T,scale,sel,sB0,sel_topk)){
            fprintf(stderr,"tc gather sel path failed\n"); return 1; }
        fprintf(stderr,"sel A/B: done\n");
    } else {
    if(!coli_cuda_attention_project_batch_dev_out(kvt,ot,o1,qd,ld,rd,S,H,Q,R,V,K,T,scale)){
        fprintf(stderr,"reference path failed\n"); return 1; }
    if(!coli_cuda_prefill_attn_gemm(kvt,ot,o2,qd,ld,rd,S,H,Q,R,V,K,T,scale,NULL,0,0)){
        fprintf(stderr,"gemm path failed\n"); return 1; }
    }
    float *h1=malloc((size_t)S*O*4), *h2=malloc((size_t)S*O*4);
    coli_cuda_pipe_download(dev,o1,h1,(size_t)S*O*4);
    coli_cuda_pipe_download(dev,o2,h2,(size_t)S*O*4);
    int first_bad=-1; float worst=0;
    for(int s=0;s<S;s++){
        double rms=0; float mx=0;
        for(int j=0;j<O;j++){ float a=h1[(size_t)s*O+j]; rms+=(double)a*a;
            float d=fabsf(h2[(size_t)s*O+j]-a); if(d>mx)mx=d; }
        rms=sqrt(rms/O);
        float rel=(float)(mx/(rms>1e-20?rms:1e-20));
        if(rel>worst)worst=rel;
        if(rel>0.05f&&first_bad<0)first_bad=s;
        if(s%16==0||s==S-1) printf("row %3d: rel %.3e\n",s,rel);
    }
    printf("S=%d T=%d zero_rope=%d zero_nope=%d -> worst rel %.3e, first_bad_row %d\n",
           S,T,zero_rope,zero_nope,worst,first_bad);
    return 0;
}
