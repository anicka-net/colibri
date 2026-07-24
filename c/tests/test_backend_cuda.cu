#include "../backend_cuda.h"
#include "../kv_dtype.h"
#include "../nvfp4.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
/* MSVC has no POSIX setenv/unsetenv */
static int setenv(const char *name, const char *value, int overwrite) {
    (void)overwrite; return _putenv_s(name, value);
}
static int unsetenv(const char *name) { return _putenv_s(name, ""); }
#endif

static int close_enough(const float *got, const float *want, int n) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(got[i] - want[i]) > 1e-4f) {
            std::fprintf(stderr, "mismatch %d: got %.6f want %.6f\n", i, got[i], want[i]);
            return 0;
        }
    }
    return 1;
}

static int relative_rms(const float *got,const float *want,int n,float limit){
    double err=0,ref=0; for(int i=0;i<n;i++){double d=got[i]-want[i];err+=d*d;ref+=(double)want[i]*want[i];}
    float r=(float)std::sqrt(err/(ref+1e-20));
    if(r>limit){std::fprintf(stderr,"relative RMS %.5f exceeds %.5f\n",r,limit);return 0;} return 1;
}

extern "C" int coli_cuda_test_gemv_q4_cached(int device,const float *x,
        const uint8_t *w,const float *scales,int I,int O,float *shared_out,
        float *cached_out);

int main(int argc, char **argv) {
    int devices[COLI_CUDA_MAX_DEVICES], ndev = argc > 1 ? argc - 1 : 1;
    if (ndev > COLI_CUDA_MAX_DEVICES) return 2;
    for (int i = 0; i < ndev; i++) devices[i] = argc > 1 ? std::atoi(argv[i + 1]) : 0;
    if (!coli_cuda_init(devices, ndev)) return 77;
    if (coli_cuda_device_count() != ndev) return 1;
    int d0 = devices[0], d1 = devices[ndev > 1 ? 1 : 0];
    int integrated = coli_cuda_device_is_integrated(d0);
    {
        enum { I=32,O=17 };float tx[I],ts[O],a[O],b[O];uint8_t tw[O*I/2];
        for(int i=0;i<I;i++)tx[i]=std::sin((float)(i+1)*.31f)*3.f;
        for(int o=0;o<O;o++)ts[o]=.01f+.003f*(o%7);
        for(int i=0;i<(int)sizeof(tw);i++){
            int lo=((i*5+3)%16),hi=((i*7+1)%16);tw[i]=(uint8_t)(lo|(hi<<4));
        }
        if(!coli_cuda_test_gemv_q4_cached(d0,tx,tw,ts,I,O,a,b)||
           !close_enough(a,b,O))return 1;
        std::fprintf(stderr,"cached q4 GEMV parity: ok\n");
    }
    size_t count = 99, bytes = 99;
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;
    const float x[8] = {1, -2, 3, -4, 2, 1, -1, 0.5f};
    float got[4];

    const int8_t q8[8] = {1, 2, 3, 4, -1, 2, -3, 4};
    const float s8[2] = {0.5f, 2.0f};
    const float want8[4] = {-5.0f, -60.0f, 1.5f, 10.0f};
    ColiCudaTensor *t8 = nullptr;
    if (!coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d0)) return 1;
    if (coli_cuda_tensor_upload(&t8, q8, s8, 1, 5, 2, d0)) return 1;
    if (ndev > 1 && coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d1)) return 1;
    if (!coli_cuda_matmul(&t8, got, x, q8, s8, 1, 2, 4, 2, d0, 0) || !close_enough(got, want8, 4)) return 1;
    /* Cached tensor must stay callable without live host pointers
     * (CUDA_RELEASE_HOST slots null theirs after upload) — including
     * SUSTAINED reuse, not just the first call. */
    for (int rep = 0; rep < 64; rep++)
        if (!coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0) ||
            !close_enough(got, want8, 4)) return 1;
    /* A tensor uploaded from a TEMPORARY host buffer must survive the buffer
     * being scribbled and freed (the release-host lifecycle). */
    {
        int8_t *tmpw = static_cast<int8_t *>(std::malloc(8));
        float  *tmps = static_cast<float *>(std::malloc(2 * sizeof(float)));
        if (!tmpw || !tmps) return 2;
        for (int i = 0; i < 8; i++) tmpw[i] = q8[i];
        tmps[0] = s8[0]; tmps[1] = s8[1];
        ColiCudaTensor *tt = nullptr;
        if (!coli_cuda_tensor_upload(&tt, tmpw, tmps, 1, 4, 2, d0)) return 1;
        for (int i = 0; i < 8; i++) tmpw[i] = 99;
        std::free(tmpw); std::free(tmps);
        if (!coli_cuda_matmul(&tt, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0) ||
            !close_enough(got, want8, 4)) return 1;
        coli_cuda_tensor_free(tt);
    }
    /* Upload failures must be graceful and must not corrupt accounting —
     * and must not poison LATER healthy launches (sticky-error regression). */
    {
        size_t c0 = 0, b0 = 0, c1 = 0, b1 = 0;
        coli_cuda_stats(-1, &c0, &b0);
        ColiCudaTensor *bad = nullptr;
        if (coli_cuda_tensor_upload(&bad, q8, s8, 1, 4, 2, 9999)) return 1;
        if (coli_cuda_tensor_upload(&bad, q8, s8, 7, 4, 2, d0)) return 1;
        if (coli_cuda_tensor_upload(&bad, q8, nullptr, 1, 4, 2, d0)) return 1;
        if (coli_cuda_tensor_upload(&bad, nullptr, s8, 1, 4, 2, d0)) return 1;
        if (coli_cuda_tensor_upload(&bad, q8, s8, 1, 1 << 20, 1 << 24, d0)) return 1; /* ~16 TB */
        if (bad) return 1;
        coli_cuda_stats(-1, &c1, &b1);
        if (c0 != c1 || b0 != b1) return 1;
        /* healthy launch immediately after the failed allocation */
        if (!coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0) ||
            !close_enough(got, want8, 4)) return 1;
    }
    /* Fault injection hook: on/off, restores cleanly. */
    if (setenv("COLI_GPU_FAIL_AFTER", "0", 1)) return 2;
    if (coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0)) return 1;
    if (unsetenv("COLI_GPU_FAIL_AFTER")) return 2;
    if (!coli_cuda_matmul(&t8, got, x, nullptr, nullptr, 1, 2, 4, 2, d0, 0) ||
        !close_enough(got, want8, 4)) return 1;
    const int8_t q8b[8]={-1,-2,-3,-4, 1,-2,3,-4};
    const float s8b[2]={1.f,.5f},want8b[4]={10.f,15.f,-3.f,-2.5f};
    if(!coli_cuda_tensor_update(t8,q8b,s8b)||
       !coli_cuda_matmul(&t8,got,x,q8b,s8b,1,2,4,2,d0,0)||
       !close_enough(got,want8b,4))return 1;

    /* Rows [-8,-1,0,7] and [1,2,3,4], packed low nibble first. */
    const uint8_t q4[4] = {0x70, 0xf8, 0xa9, 0xcb};
    const float s4[2] = {1.0f, 0.25f};
    const float want4[2] = {-34.0f, -2.5f};
    ColiCudaTensor *t4 = nullptr;
    if (!coli_cuda_matmul(&t4, got, x, q4, s4, 2, 1, 4, 2, d1, 0) || !close_enough(got, want4, 2)) return 1;

    /* Grouped INT4 keeps offset-binary nibbles and has O*ceil(I/gs) scales.
     * Guard the generic upload/matmul/accounting path independently of the
     * routed-expert grouped kernels. */
    const uint8_t q4g[4] = {0x80,0x7f,0xa9,0xcb};
    const float s4g[4] = {1.0f,0.5f,0.25f,2.0f};
    const float want4g[2] = {4.5f,-14.75f};
    ColiCudaTensor *t4g = nullptr;
    if (!coli_cuda_matmul(&t4g,got,x,q4g,s4g,COLI_TENSOR_INT4_GROUP,
                          1,4,2,d1,2) ||
        !close_enough(got,want4g,2) ||
        coli_cuda_tensor_bytes(t4g)!=sizeof(q4g)+sizeof(s4g)) return 1;

    const uint8_t q2[2] = {0xe4, 0x1b};
    const float s2[2] = {0.5f, 2.0f};
    const float want2[2] = {-2.0f, 12.0f};
    ColiCudaTensor *t2 = nullptr;
    if (!coli_cuda_matmul(&t2, got, x, q2, s2, 3, 1, 4, 2, d1, 0) || !close_enough(got, want2, 2)) return 1;

    const float wf[8] = {1, 0, -1, 2, 0.5f, 0.5f, 0.5f, 0.5f};
    const float wantf[2] = {-10.0f, -1.0f};
    ColiCudaTensor *tf = nullptr;
    if (!coli_cuda_matmul(&tf, got, x, wf, nullptr, 0, 1, 4, 2, d0, 0) || !close_enough(got, wantf, 2)) return 1;

    const float eg[8] = {1,0,0,0, 0,1,0,0};
    const float eu[8] = {1,0,0,0, 0,1,0,0};
    const float ed[8] = {1,0, 0,1, 1,1, 1,-1};
    ColiCudaTensor *tg=nullptr,*tu=nullptr,*td=nullptr;
    if (!coli_cuda_tensor_upload_g(&tg,eg,nullptr,0,4,2,d0,0) ||
        !coli_cuda_tensor_upload_g(&tu,eu,nullptr,0,4,2,d0,0) ||
        !coli_cuda_tensor_upload_g(&td,ed,nullptr,0,2,4,d0,0)) return 1;
    float expert[8], want_expert[8];
    for(int s=0;s<2;s++){
        float a=x[s*4], b=x[s*4+1];
        a=(a/(1.0f+std::exp(-a)))*a; b=(b/(1.0f+std::exp(-b)))*b;
        want_expert[s*4]=a; want_expert[s*4+1]=b;
        want_expert[s*4+2]=a+b; want_expert[s*4+3]=a-b;
    }
    if (!coli_cuda_expert_mlp(tg,tu,td,expert,x,2) ||
        !close_enough(expert,want_expert,8)) return 1;
    ColiCudaTensor *gates[2]={tg,tg},*ups[2]={tu,tu},*downs[2]={td,td};
    int group_rows[2]={1,1}; float grouped[8];
    if (!coli_cuda_expert_group(gates,ups,downs,group_rows,2,grouped,x,nullptr,nullptr,0,0,0,nullptr) ||
        !close_enough(grouped,want_expert,8)) return 1;
    float *dx=(float*)coli_cuda_pipe_alloc(d0,sizeof(x));
    float *dout=(float*)coli_cuda_pipe_alloc(d0,sizeof(x));
    float zero[8]={0},device_accum[8],want_accum[8];int tok2[2]={0,1};float wt2[2]={.5f,.25f};
    if(!dx||!dout){std::fprintf(stderr,"device path allocation failed\n");return 1;}
    if(!coli_cuda_pipe_upload(d0,dx,x,sizeof(x))||
       !coli_cuda_pipe_upload(d0,dout,zero,sizeof(zero))){
        std::fprintf(stderr,"device path upload failed\n");return 1;
    }
    int device_group_rc=coli_cuda_expert_group(gates,ups,downs,group_rows,2,grouped,
                                                dx,wt2,tok2,2,0,1,dout);
    if(device_group_rc!=2){
        std::fprintf(stderr,"device path group returned %d\n",device_group_rc);return 1;
    }
    if(!coli_cuda_pipe_download(d0,dout,device_accum,sizeof(device_accum))){
        std::fprintf(stderr,"device path download failed\n");return 1;
    }
    for(int i=0;i<4;i++){want_accum[i]=want_expert[i]*.5f;want_accum[4+i]=want_expert[4+i]*.25f;}
    if(!close_enough(device_accum,want_accum,8))return 1;
    if(!coli_cuda_pipe_upload(d0,dout,zero,sizeof(zero))){
        std::fprintf(stderr,"device ordering setup upload failed\n");return 1;
    }
    for(int i=0;i<64;i++)
        if(coli_cuda_expert_group(gates,ups,downs,group_rows,2,grouped,dx,
                                  wt2,tok2,2,0,1,dout)!=2){
            std::fprintf(stderr,"device ordering group %d failed\n",i);return 1;
        }
    cudaError_t order_err=cudaMemsetAsync(dout,0,sizeof(zero),0);
    if(order_err==cudaSuccess)order_err=cudaDeviceSynchronize();
    if(order_err!=cudaSuccess){
        std::fprintf(stderr,"device ordering sync failed: %s\n",cudaGetErrorString(order_err));return 1;
    }
    if(!coli_cuda_pipe_download(d0,dout,device_accum,sizeof(device_accum))){
        std::fprintf(stderr,"device ordering download failed\n");return 1;
    }
    if(!close_enough(device_accum,zero,8))return 1;
    std::fprintf(stderr,"device group ordering: ok\n");

    /* Resident dense-layer MLP composition: gate/up -> SwiGLU -> down ->
       residual add, using the same pipe primitives as pipe_layer_cuda(). */
    float *dgate=(float*)coli_cuda_pipe_alloc(d0,4*sizeof(float));
    float *dup=(float*)coli_cuda_pipe_alloc(d0,4*sizeof(float));
    float dense_got[8],dense_want[8];
    if(!dgate||!dup){std::fprintf(stderr,"dense pipe allocation failed\n");return 1;}
    if(!coli_cuda_pipe_upload(d0,dout,zero,sizeof(zero))||
       !coli_cuda_pipe_gemm(tg,dgate,dx,2)||
       !coli_cuda_pipe_gemm(tu,dup,dx,2)||
       !coli_cuda_pipe_silu_mul(d0,dgate,dup,4)||
       !coli_cuda_pipe_gemm(td,dout,dgate,2)||
       !coli_cuda_pipe_add(d0,dout,dx,8)||
       !coli_cuda_pipe_download(d0,dout,dense_got,sizeof(dense_got))){
        std::fprintf(stderr,"dense pipe composition failed\n");return 1;
    }
    for(int i=0;i<8;i++)dense_want[i]=want_expert[i]+x[i];
    if(!close_enough(dense_got,dense_want,8))return 1;
    std::fprintf(stderr,"resident dense MLP composition: ok\n");
    coli_cuda_pipe_free(d0,dgate);coli_cuda_pipe_free(d0,dup);
    coli_cuda_pipe_free(d0,dx);coli_cuda_pipe_free(d0,dout);

    const int8_t q8d[8]={1,0, 0,1, 1,1, 1,-1};const float s8d[4]={1,1,1,1};
    ColiCudaTensor *d8=nullptr;float device8[8],host8[8];
    if(!coli_cuda_tensor_upload(&d8,q8d,s8d,1,2,4,d0))return 1;
    ColiCudaTensor *gg8[2]={t8,t8},*ug8[2]={t8,t8},*dg8[2]={d8,d8};
    if(!coli_cuda_expert_group(gg8,ug8,dg8,group_rows,2,device8,x,nullptr,nullptr,0,0,0,nullptr))return 1;
    ColiCudaTensor *hg8=nullptr,*hu8=nullptr,*hd8=nullptr;
    int host8_ok=coli_cuda_tensor_wrap_host(&hg8,q8b,s8b,1,4,2,d0)&&
                 coli_cuda_tensor_wrap_host(&hu8,q8b,s8b,1,4,2,d0)&&
                 coli_cuda_tensor_wrap_host(&hd8,q8d,s8d,1,2,4,d0);
    if(host8_ok){
        ColiCudaTensor *hgg8[2]={hg8,hg8},*hug8[2]={hu8,hu8},*hdg8[2]={hd8,hd8};
        if(!coli_cuda_expert_group(hgg8,hug8,hdg8,group_rows,2,host8,x,nullptr,nullptr,0,0,0,nullptr)||
           !close_enough(host8,device8,8))return 1;
    }
    coli_cuda_tensor_free(hg8);coli_cuda_tensor_free(hu8);coli_cuda_tensor_free(hd8);
    coli_cuda_tensor_free(d8);

    const float aw[16]={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    const float aq[4]={1,2,.5f,-.5f},al[12]={1,0,0,0, 0,1,0,0, 0,0,1,0};
    const float ar[6]={1,0, 0,1, 1,1};float actx[2],aref[2];
    ColiCudaTensor *at=nullptr;if(!coli_cuda_tensor_upload_g(&at,aw,nullptr,0,4,4,d0,0))return 1;
    float score[3];for(int t=0;t<3;t++)score[t]=aq[0]*al[t*4]+aq[1]*al[t*4+1]+aq[2]*ar[t*2]+aq[3]*ar[t*2+1];
    float mx=score[0],z=0;for(int t=1;t<3;t++)mx=score[t]>mx?score[t]:mx;
    for(int t=0;t<3;t++){score[t]=std::exp(score[t]-mx);z+=score[t];}for(int t=0;t<3;t++)score[t]/=z;
    for(int v=0;v<2;v++){aref[v]=0;for(int t=0;t<3;t++)aref[v]+=score[t]*al[t*4+2+v];}
    if(!coli_cuda_attention_absorb(at,actx,aq,al,ar,1,2,2,2,4,3,1.f)||
       !close_enough(actx,aref,2))return 1;
    coli_cuda_tensor_free(at);
    uint16_t abf16[16];
    for(int i=0;i<16;i++){ uint32_t bits;std::memcpy(&bits,&aw[i],sizeof(bits));abf16[i]=(uint16_t)(bits>>16); }
    ColiCudaTensor *atbf=nullptr;
    if(!coli_cuda_tensor_upload(&atbf,abf16,nullptr,COLI_TENSOR_BF16,4,4,d0)||
       !coli_cuda_attention_absorb(atbf,actx,aq,al,ar,1,2,2,2,4,3,1.f)||
       !close_enough(actx,aref,2))return 1;
    if(coli_cuda_kv_dtype()==COLI_KV_FP8_E4M3){
        void *dl=coli_cuda_pipe_alloc(d0,sizeof(al)),*dr=coli_cuda_pipe_alloc(d0,sizeof(ar));
        float *dls=(float*)coli_cuda_pipe_alloc(d0,3*sizeof(float));
        float *drs=(float*)coli_cuda_pipe_alloc(d0,3*sizeof(float));
        if(!dl||!dr||!dls||!drs||
           !coli_cuda_pipe_upload_kv_rows(d0,dl,dls,al,3,4,0)||
           !coli_cuda_pipe_upload_kv_rows(d0,dr,drs,ar,3,2,0)||
           !coli_cuda_attention_absorb_kvdev(atbf,actx,aq,(const float*)dl,(const float*)dr,
                dls,drs,1,2,2,2,4,3,1.f)||
           !relative_rms(actx,aref,2,.03f))return 1;
        uint64_t qrows=0,rrows=0,fb=0;coli_cuda_kv_fp8_stats(&qrows,&rrows,&fb);
        if(qrows!=6||rrows!=3||fb)return 1;
        coli_cuda_pipe_free(d0,dl);coli_cuda_pipe_free(d0,dr);
        coli_cuda_pipe_free(d0,dls);coli_cuda_pipe_free(d0,drs);
        std::fprintf(stderr,"FP8 KV BF16 attention reader: ok\n");

        /* Long-context shadow gate: exercise the real 32k row quantizers and
           dequantizing attention reader, including their scale arrays.  Zero
           KV has an exact finite zero result independent of softmax rounding. */
        constexpr int LT=32768;
        float *ll=(float*)std::calloc((size_t)LT*4,sizeof(float));
        float *lr=(float*)std::calloc((size_t)LT*2,sizeof(float));
        void *lld=coli_cuda_pipe_alloc(d0,(size_t)LT*4);
        void *lrd=coli_cuda_pipe_alloc(d0,(size_t)LT*2);
        float *lls=(float*)coli_cuda_pipe_alloc(d0,(size_t)LT*sizeof(float));
        float *lrs=(float*)coli_cuda_pipe_alloc(d0,(size_t)LT*sizeof(float));
        if(!ll||!lr||!lld||!lrd||!lls||!lrs){
            std::fprintf(stderr,"FP8 KV 32k allocation failed\n");return 1;
        }
        if(!coli_cuda_pipe_upload_kv_rows(d0,lld,lls,ll,LT,4,0)){
            std::fprintf(stderr,"FP8 KV 32k latent quantization failed\n");return 1;
        }
        if(!coli_cuda_pipe_upload_kv_rows(d0,lrd,lrs,lr,LT,2,0)){
            std::fprintf(stderr,"FP8 KV 32k rope quantization failed\n");return 1;
        }
        if(!coli_cuda_attention_absorb_kvdev(atbf,actx,aq,(const float*)lld,
                (const float*)lrd,lls,lrs,1,2,2,2,4,LT,1.f)){
            std::fprintf(stderr,"FP8 KV 32k attention reader failed\n");return 1;
        }
        if(!std::isfinite(actx[0])||!std::isfinite(actx[1])||
           std::fabs(actx[0])>1e-6f||std::fabs(actx[1])>1e-6f){
            std::fprintf(stderr,"FP8 KV 32k result invalid: %g %g\n",actx[0],actx[1]);return 1;
        }
        coli_cuda_kv_fp8_stats(&qrows,&rrows,&fb);
        if(qrows!=6+2u*LT||rrows!=3u+LT||fb){
            std::fprintf(stderr,"FP8 KV 32k counters invalid: q=%llu r=%llu fallback=%llu\n",
                (unsigned long long)qrows,(unsigned long long)rrows,(unsigned long long)fb);
            return 1;
        }
        std::free(ll);std::free(lr);
        coli_cuda_pipe_free(d0,lld);coli_cuda_pipe_free(d0,lrd);
        coli_cuda_pipe_free(d0,lls);coli_cuda_pipe_free(d0,lrs);
        std::fprintf(stderr,"FP8 KV 32k shadow/reader: ok\n");
    }
    coli_cuda_tensor_free(atbf);

    /* Faithful snapshots use BF16 resident attention tensors.  Exercise the
       generic DSA prefill split (two causal rows + one selected row) against a
       zero oracle in every configured KV-shadow dtype. */
    {
        constexpr int S=3,H=1,Q=16,R=16,V=16,K=16,T=3,TOPK=2;
        uint16_t wk[H*(Q+V)*K]={},wo[V*H*V]={};
        ColiCudaTensor *kw=nullptr,*ow=nullptr;
        float q[S*H*(Q+R)]={},lh[T*K]={},rh[T*R]={},got[S*V]={};
        int sel[S*TOPK]={0,0,0,0,0,1};
        int kd=coli_cuda_kv_dtype(),esz=kd==COLI_KV_FP8_E4M3?1:kd==COLI_KV_FP16?2:4;
        void *qd=coli_cuda_pipe_alloc(d0,sizeof(q));
        void *ld=coli_cuda_pipe_alloc(d0,(size_t)T*K*esz);
        void *rd=coli_cuda_pipe_alloc(d0,(size_t)T*R*esz);
        float *ls=kd==COLI_KV_FP8_E4M3?(float*)coli_cuda_pipe_alloc(d0,T*sizeof(float)):nullptr;
        float *rs=kd==COLI_KV_FP8_E4M3?(float*)coli_cuda_pipe_alloc(d0,T*sizeof(float)):nullptr;
        void *od=coli_cuda_pipe_alloc(d0,sizeof(got));
        if(!coli_cuda_tensor_upload(&kw,wk,nullptr,COLI_TENSOR_BF16,K,H*(Q+V),d0)||
           !coli_cuda_tensor_upload(&ow,wo,nullptr,COLI_TENSOR_BF16,H*V,V,d0)||
           !qd||!ld||!rd||!od||(kd==COLI_KV_FP8_E4M3&&(!ls||!rs))||
           !coli_cuda_pipe_upload(d0,(float*)qd,q,sizeof(q)/sizeof(float))||
           !coli_cuda_pipe_upload_kv_rows(d0,ld,ls,lh,T,K,0)||
           !coli_cuda_pipe_upload_kv_rows(d0,rd,rs,rh,T,R,0)||
           !coli_cuda_prefill_attn_gemm(kw,ow,(float*)od,(const float*)qd,
                (const float*)ld,(const float*)rd,ls,rs,S,H,Q,R,V,K,T,1.f,sel,2,TOPK)||
           !coli_cuda_pipe_download(d0,(const float*)od,got,sizeof(got)))return 1;
        for(float value:got)if(value!=0.f||!std::isfinite(value))return 1;
        coli_cuda_tensor_free(kw);coli_cuda_tensor_free(ow);
        coli_cuda_pipe_free(d0,qd);coli_cuda_pipe_free(d0,ld);coli_cuda_pipe_free(d0,rd);
        coli_cuda_pipe_free(d0,od);if(ls)coli_cuda_pipe_free(d0,ls);if(rs)coli_cuda_pipe_free(d0,rs);
        std::fprintf(stderr,"BF16 generic DSA prefill split: ok\n");
    }

    /* Native s4 WMMA path: compare the quantized-activation result against the
       existing FP32-activation/s4-weight grouped implementation. */
    uint8_t w4[32*32/2]; float ws4[32], gx4[64], scalar4[64], tensor4[64];
    for(int i=0;i<(int)sizeof(w4);i++){
        int lo=((i%15)-7)&15,hi=(((i*3)%15)-7)&15;
        w4[i]=(uint8_t)(lo|(hi<<4));
    }
    for(int i=0;i<32;i++)ws4[i]=0.01f+(i%5)*0.002f;
    for(int i=0;i<64;i++)gx4[i]=std::sin((float)(i+1)*0.17f)*2.f;
    ColiCudaTensor *g4=nullptr,*u4=nullptr,*d4=nullptr;
    if(!coli_cuda_tensor_upload_g(&g4,w4,ws4,2,32,32,d0,0)||
       !coli_cuda_tensor_upload_g(&u4,w4,ws4,2,32,32,d0,0)||
       !coli_cuda_tensor_upload_g(&d4,w4,ws4,2,32,32,d0,0))return 1;
    ColiCudaTensor *gg4[2]={g4,g4},*ug4[2]={u4,u4},*dg4[2]={d4,d4};
    if(!coli_cuda_expert_group(gg4,ug4,dg4,group_rows,2,scalar4,gx4,nullptr,nullptr,0,0,0,nullptr))return 1;
    ColiCudaTensor *hg4=nullptr,*hu4=nullptr,*hd4=nullptr;float host4[64];
    int host_ok=coli_cuda_tensor_wrap_host(&hg4,w4,ws4,2,32,32,d0)&&
                coli_cuda_tensor_wrap_host(&hu4,w4,ws4,2,32,32,d0)&&
                coli_cuda_tensor_wrap_host(&hd4,w4,ws4,2,32,32,d0);
    if(host_ok){
        ColiCudaTensor *hgg4[2]={hg4,hg4},*hug4[2]={hu4,hu4},*hdg4[2]={hd4,hd4};
        if(!coli_cuda_expert_group(hgg4,hug4,hdg4,group_rows,2,host4,gx4,nullptr,nullptr,0,0,0,nullptr)||
           !close_enough(host4,scalar4,64))return 1;
    }
    coli_cuda_tensor_free(hg4);coli_cuda_tensor_free(hu4);coli_cuda_tensor_free(hd4);
    setenv("COLI_CUDA_TC_INT4","1",1);
    setenv("COLI_CUDA_TC_MIN_ROWS","1",1);
    if(!coli_cuda_expert_group(gg4,ug4,dg4,group_rows,2,tensor4,gx4,nullptr,nullptr,0,0,0,nullptr)||
       !relative_rms(tensor4,scalar4,64,0.30f))return 1;
    unsetenv("COLI_CUDA_TC_INT4");
    unsetenv("COLI_CUDA_TC_MIN_ROWS");
    coli_cuda_tensor_free(g4);coli_cuda_tensor_free(u4);coli_cuda_tensor_free(d4);

    /* Native ModelOpt NVFP4: the same fixture exercises a single GEMM, fused
       expert MLP, and routed grouping. On SM120/121 compare native W4A4 with
       the software oracle; elsewhere require exact generic W4A32 behavior. */
    enum { NI=32,NO=32,NS=2 };
    uint8_t nw[NO*NI/2],nscale[512];float nx[NS*NI],ngot[NS*NO],nref[NS*NO];
    std::memset(nscale,0,sizeof(nscale));
    for(int i=0;i<(int)sizeof(nw);i++)nw[i]=(uint8_t)(((i+1)&15)|(((i*3+5)&15)<<4));
    for(int i=0;i<NS*NI;i++)nx[i]=std::sin((float)(i+1)*.11f)*1.5f;
    for(int o=0;o<NO;o++)for(int g=0;g<NI/16;g++)
        nscale[coli_nvfp4_cutlass_scale_offset(o,g,NI)]=0x38;
    ColiCudaTensor *ng=nullptr,*nu=nullptr,*nd=nullptr;uint64_t native0=0,native1=0;
    coli_cuda_nvfp4_stats(&native0,nullptr,nullptr,nullptr);
    if(!coli_cuda_matmul_nvfp4(&ng,ngot,nx,nw,nscale,.25f,.5f,
          COLI_SCALE_CUTLASS_SM1XX_128X4,NS,NI,NO,d0))return 1;
    coli_cuda_nvfp4_stats(&native1,nullptr,nullptr,nullptr);
    if(native1>native0){
        if(!coli_matmul_nvfp4_w4a4_ref(nref,nx,nw,nscale,.25f,.5f,
              COLI_SCALE_CUTLASS_SM1XX_128X4,NS,NI,NO)||
           !relative_rms(ngot,nref,NS*NO,.12f))return 1;
        std::fprintf(stderr,"native NVFP4 GEMM oracle: ok\n");
    }else{
        if(!coli_matmul_nvfp4_w4a32_ref(nref,nx,nw,nscale,.25f,
              COLI_SCALE_CUTLASS_SM1XX_128X4,NS,NI,NO)||
           !close_enough(ngot,nref,NS*NO))return 1;
        std::fprintf(stderr,"generic NVFP4 GEMM oracle: ok\n");
    }
    if(!coli_cuda_tensor_upload_nvfp4(&nu,nw,nscale,.25f,.5f,
          COLI_SCALE_CUTLASS_SM1XX_128X4,NI,NO,d0)||
       !coli_cuda_tensor_upload_nvfp4(&nd,nw,nscale,.25f,.5f,
          COLI_SCALE_CUTLASS_SM1XX_128X4,NI,NO,d0))return 1;
    float nmlp[NS*NI],ngroup[NS*NI];
    if(!coli_cuda_expert_mlp(ng,nu,nd,nmlp,nx,NS))return 1;
    ColiCudaTensor *ngs[2]={ng,ng},*nus[2]={nu,nu},*nds[2]={nd,nd};
    if(!coli_cuda_expert_group(ngs,nus,nds,group_rows,2,ngroup,nx,
                               nullptr,nullptr,0,0,0,nullptr)||
       !close_enough(ngroup,nmlp,NS*NI))return 1;
    uint64_t ngcalls=0,ngproblems=0,ngfallbacks=0;
    coli_cuda_nvfp4_grouped_stats(&ngcalls,&ngproblems,&ngfallbacks);
    if(native1>native0&&(ngcalls!=3||ngproblems!=6||ngfallbacks))return 1;
    std::fprintf(stderr,"grouped NVFP4 expert parity: ok\n");
    coli_cuda_tensor_free(ng);coli_cuda_tensor_free(nu);coli_cuda_tensor_free(nd);

    uint64_t group_calls=0,group_experts=0,group_total_rows=0;
    coli_cuda_group_stats(&group_calls,&group_experts,&group_total_rows,nullptr,nullptr,nullptr);
    uint64_t want_groups=70+(host8_ok?1:0)+(host_ok?1:0);
    if(group_calls!=want_groups||group_experts!=want_groups*2||group_total_rows!=want_groups*2){
        std::fprintf(stderr,"group stats: got %llu/%llu/%llu, want %llu/%llu/%llu\n",
            (unsigned long long)group_calls,(unsigned long long)group_experts,
            (unsigned long long)group_total_rows,(unsigned long long)want_groups,
            (unsigned long long)(want_groups*2),(unsigned long long)(want_groups*2));
        return 1;
    }

    coli_cuda_stats(-1, &count, &bytes);
    if (count != 8 || bytes != 186) {
        std::fprintf(stderr, "unexpected CUDA stats: %zu tensors, %zu bytes\n", count, bytes);
        return 1;
    }
    if (coli_cuda_tensor_device(t8) != d0 || coli_cuda_tensor_device(tf) != d0 ||
        coli_cuda_tensor_device(t4) != d1 || coli_cuda_tensor_device(t4g) != d1 ||
        coli_cuda_tensor_device(t2) != d1) return 1;
    coli_cuda_stats(d0, &count, &bytes);
    if (ndev > 1) {
        if (count != 5 || bytes != 144) return 1;
        coli_cuda_stats(d1, &count, &bytes);
        if (count != 3 || bytes != 42) return 1;
    } else if (count != 8 || bytes != 186) return 1;

    coli_cuda_tensor_free(t8);
    coli_cuda_tensor_free(t4);
    coli_cuda_tensor_free(t4g);
    coli_cuda_tensor_free(t2);
    coli_cuda_tensor_free(tf);
    coli_cuda_tensor_free(tg);
    coli_cuda_tensor_free(tu);
    coli_cuda_tensor_free(td);
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;
    coli_cuda_shutdown();
    std::printf("cuda backend: q8/q4/q2/f32 correctness ok on %d device(s), integrated=%d\n",
                ndev,integrated);
    return 0;
}
