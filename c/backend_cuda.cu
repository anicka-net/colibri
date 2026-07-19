#include "backend_cuda.h"

#include <cuda_runtime.h>
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#include <ctype.h>
#endif
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <mutex>

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int tracked;
    int host_backed;
};

typedef struct {
    int device;
    int compute_major,compute_minor,integrated;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    uint8_t *qx; float *qscale;
    size_t qx_cap, qscale_cap;
    float *host_x,*host_y; size_t host_x_cap,host_y_cap;
    float *aq,*al,*ar,*ac; size_t aq_cap,al_cap,ar_cap,ac_cap;
    float *cvt; size_t cvt_cap;                 /* staging fp32 per upload_kv f16 */
    float *pf_q,*pf_c,*pf_s; size_t pf_q_cap,pf_c_cap,pf_s_cap; /* prefill GEMM attention */
    int smem_optin;                             /* smem opt-in per l'absorb a T lunghi */
    int absorb_attr_set, ragged_attr_set;       /* attributo gia' alzato per kernel */
    int absorb_attr_set_h;                      /* idem, istanza __half (COLI_KV_F16) */
    float *pipe_buf[44]; size_t pipe_cap[44];   /* scratch persistenti del resident pipeline */
    float *accum; size_t accum_cap;             /* device-side routed-expert accumulate */
    float *wrow_d; size_t wrow_cap;
    cudaEvent_t group_ev; int group_ev_init, accum_pending;
    cudaStream_t stream;
    void *group_desc; size_t group_desc_cap;
    size_t tensor_count, tensor_bytes;
} DeviceContext;

typedef struct {
    const void *g,*u,*d; const float *gs,*us,*ds;
    int gf,uf,df,rows,offset;
    int go,uo,dof;
} GroupDesc;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;
static uint64_t g_group_calls,g_group_experts,g_group_rows;
static double g_group_h2d_ms,g_group_kernel_ms,g_group_d2h_ms;
static std::mutex g_group_stats_mu;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

/* cudaSetDevice on every call doubles expert-matmul time on 2 GPUs when the
 * serial expert loop alternates devices (measured on RTX 5090 + 4090: 14.3s
 * -> 25.4s per 32 tokens). The current device is per-thread in the CUDA
 * runtime, so a thread-local cache skips the redundant switches. */
static thread_local int g_current_device = -1;

static int select_ctx(DeviceContext *ctx) {
    if (!ctx) return 0;
    if (g_current_device == ctx->device) return 1;
    if (!cuda_ok(cudaSetDevice(ctx->device), "select device")) return 0;
    g_current_device = ctx->device;
    return 1;
}

__host__ __device__ static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    return 0;
}

__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2) {
        uint8_t v = q[i >> 1];
        int n=(i&1)?(v>>4):(v&15); return static_cast<float>(n&8?n-16:n);
    }
    uint8_t v = q[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
}

__device__ static float group_weight_at(const void *weights,int fmt,size_t row,int i,int offset_s4){
    if(fmt!=2||!offset_s4)return weight_at(weights,fmt,row,i);
    const uint8_t v=static_cast<const uint8_t*>(weights)[row+(i>>1)];
    const int n=(i&1)?v>>4:v&15;
    return static_cast<float>(n-8);
}

__device__ static float decode_group_s4(int n,int offset_s4){
    return static_cast<float>(offset_s4?n-8:(n^8)-8);
}

__global__ static void offset_to_signed_s4(uint8_t *q,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)q[i]^=0x88;
}

/* ---- COLI_KV_F16 (default 1): ombre device fp16 -----------------------------
 * Le ombre device della cache KV (Lc/Rc) e della cache indexer (Ic) vivono in
 * __half: metà VRAM (~24 vs ~48 GB a 256k) e metà banda nei kernel che le
 * leggono; nei percorsi tensor-core lo staging fp32→fp16 sparisce del tutto.
 * L'HOST resta fp32 canonico ed ESATTO: i writer calcolano in fp32 (staging),
 * scaricano l'fp32 all'host e convertono solo la copia device.  =0 ripristina
 * le ombre fp32 (A/B, pfg_test).  I kernel lettori sono template sul tipo di
 * storage; shf/shh sono le load convertite. */
static int kv_f16_mode(void){
    static int m=-1;
    if(m<0){ const char *e=getenv("COLI_KV_F16"); m=e?(atoi(e)!=0):1; }
    return m;
}
extern "C" int coli_cuda_kv_f16(void){ return kv_f16_mode(); }
__device__ static inline float  shf(const float *p){ return *p; }
__device__ static inline float  shf(const __half *p){ return __half2float(*p); }
__device__ static inline __half shh(const float *p){ return __float2half(*p); }
__device__ static inline __half shh(const __half *p){ return *p; }
__global__ static void cvt_f32_f16(__half *__restrict__ dst,
                                   const float *__restrict__ src, size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) dst[i]=__float2half(src[i]);
}
/* righe strided: dst riga y (pitch dpitch, half) <- src riga y (pitch spitch, fp32) */
__global__ static void cvt2d_f32_f16(__half *__restrict__ dst,int dpitch,
                                     const float *__restrict__ src,int spitch,int w){
    int y=blockIdx.y, x=blockIdx.x*blockDim.x+threadIdx.x;
    if(x<w) dst[(size_t)y*dpitch+x]=__float2half(src[(size_t)y*spitch+x]);
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    for (int i = threadIdx.x; i < I; i += blockDim.x)
        sum += xs[i] * weight_at(weights, fmt, row, i);

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        y[(size_t)s * O + o] = partial[0] * (fmt ? scales[o] : 1.0f);
}

__global__ static void silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

/* Four warps share one A tile and compute 16x64 outputs.  This matters for
 * prefill: the first prototype reloaded/converter A once per 16 output cols. */
__global__ static void w4a16_matmul(float *y,const float *x,const uint8_t *w,
                                    const float *scale,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    size_t rb=(size_t)(K+1)/2;
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){
            int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);
        }
        for(int z=lane;z<256;z+=32){
            int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}
            bh[warp][z]=__float2half(v);           /* [Ntile,Ktile] == B col-major */
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

/* Gate and up use the same input.  Eight warps compute both 16x64 projections
 * while sharing the FP32->FP16 conversion of A. */
__global__ static void w4a16_gate_up(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const float *gs,const float *us,
        int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31,which=warp&1,tile=warp>>1;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+tile*16;const uint8_t *w=which?uw:gw;
    const float *scale=which?us:gs;float *y=which?up:gate;size_t rb=(size_t)(K+1)/2;
    __shared__ __half ah[256],bh[8][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);}
        for(int z=lane;z<256;z+=32){int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}bh[warp][z]=__float2half(v);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[8][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

__global__ static void quantize_s4_rows(uint8_t *q,float *scale,const float *x,int S,int K){
    int s=blockIdx.x; if(s>=S)return; const float *xs=x+(size_t)s*K;
    float v=0; for(int i=threadIdx.x;i<K;i+=blockDim.x)v=fmaxf(v,fabsf(xs[i]));
    __shared__ float m[256]; m[threadIdx.x]=v; __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)m[threadIdx.x]=fmaxf(m[threadIdx.x],m[threadIdx.x+n]);__syncthreads();}
    float sc=m[0]>0?m[0]/7.f:1.f; if(!threadIdx.x)scale[s]=sc;
    uint8_t *dst=q+(size_t)s*((K+1)/2);
    for(int b=threadIdx.x;b<(K+1)/2;b+=blockDim.x){
        int i=b*2,a=__float2int_rn(xs[i]/sc),c=i+1<K?__float2int_rn(xs[i+1]/sc):0;
        a=max(-8,min(7,a)); c=max(-8,min(7,c)); dst[b]=(uint8_t)((a&15)|((c&15)<<4));
    }
}

__global__ static void grouped_s4_wmma(float *y,const uint8_t *x,const float *xscale,
                                        const GroupDesc *desc,int K,int O,int which){
#if __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    int warp=threadIdx.x/32,lane=threadIdx.x%32,tile=blockIdx.x*8+warp,c=blockIdx.y;
    if(tile*8>=O)return; GroupDesc d=desc[c];
    const void *w=which==0?d.g:(which==1?d.u:d.d);
    const float *ws=which==0?d.gs:(which==1?d.us:d.ds);
    int fmt=which==0?d.gf:(which==1?d.uf:d.df);
    if(fmt!=2)return;
    wmma::fragment<wmma::accumulator,8,8,32,int> acc; wmma::fill_fragment(acc,0);
    const uint8_t *a=x+(size_t)d.offset*((K+1)/2);
    const uint8_t *b=(const uint8_t*)w+(size_t)(tile*8)*((K+1)/2);
    for(int k=0;k<K;k+=32){
        wmma::fragment<wmma::matrix_a,8,8,32,wmma::experimental::precision::s4,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,8,8,32,wmma::experimental::precision::s4,wmma::col_major> bf;
        wmma::load_matrix_sync(af,a+k/2,K);
        wmma::load_matrix_sync(bf,b+k/2,K);
        wmma::mma_sync(acc,af,bf,acc);
    }
    __shared__ int out[8][64]; wmma::store_matrix_sync(out[warp],acc,8,wmma::mem_row_major);
    for(int i=lane;i<64;i+=32){int s=i/8,o=tile*8+i%8;
        if(s<d.rows&&o<O)y[(size_t)(d.offset+s)*O+o]=(float)out[warp][i]*xscale[d.offset+s]*ws[o];}
#endif
}

__global__ static void grouped_hidden(float *y,const float *x,const GroupDesc *desc,
                                      int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    const void *w=which?d.u:d.g; const float *sc=which?d.us:d.gs; int fmt=which?d.uf:d.gf;
    int off=which?d.uo:d.go;
    size_t rb=row_bytes(fmt,D),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0; for(int i=threadIdx.x;i<D;i+=blockDim.x) sum+=xs[i]*group_weight_at(w,fmt,row,i,off);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*I+o]=p[0]*(fmt?sc[o]:1.f);
}

__global__ static void grouped_down(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    size_t rb=row_bytes(d.df,I),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*I;
    float sum=0; for(int i=threadIdx.x;i<I;i+=blockDim.x) sum+=xs[i]*group_weight_at(d.d,d.df,row,i,d.dof);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*D+o]=p[0]*(d.df?d.ds[o]:1.f);
}

__device__ static void unpack_s4(uint8_t v,int offset,float *lo,float *hi){
    int a=v&15,b=v>>4;
    *lo=(float)(offset?(a-8):(a&8?a-16:a));
    *hi=(float)(offset?(b-8):(b&8?b-16:b));
}

/* Exact low-row W4A32 path. It consumes each packed weight byte once instead
 * of routing both nibbles through weight_at(), preserving FP32 activations. */
__global__ static void grouped_hidden_w4(float *y,const float *x,const GroupDesc *desc,
                                         int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *w=(const uint8_t*)(which?d.u:d.g);const float *sc=which?d.us:d.gs;
    const uint8_t *row=w+(size_t)o*((D+1)/2);const float *xs=x+(size_t)(d.offset+s)*D;
    int off=which?d.uo:d.go;
    float sum=0;for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],off,&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<D)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*I+o]=p[0]*sc[o];
}

__global__ static void grouped_hidden_w4_dual(float *gate,float *up,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],d.go,&g0,&g1);unpack_s4(ur[b],d.uo,&u0,&u1);
        int i=b*2;ga+=xs[i]*g0;ua+=xs[i]*u0;if(i+1<D){ga+=xs[i+1]*g1;ua+=xs[i+1]*u1;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;gate[z]=gp[0]*d.gs[o];up[z]=upv[0]*d.us[o];}
}

__global__ static void grouped_down_w4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],d.dof,&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<I)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0]*d.ds[o];
}

__global__ static void attention_absorb_kernel(float *ctx,const float *q,const float *latent,
                                                const float *rope,const void *weights,const float *wscale,
                                                int fmt,int H,int Q,int R,int V,int K,int T,float scale){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*scores=cl+K;
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*(fmt?wscale[rbase+d]:1.f);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<T;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        for(int k=0;k<K;k++)a+=qa[k]*lt[k];for(int d=0;d<R;d++)a+=q[(size_t)h*(Q+R)+Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    if(!tid){float mx=scores[0];for(int t=1;t<T;t++)mx=fmaxf(mx,scores[t]);float z=0;
        for(int t=0;t<T;t++){scores[t]=expf(scores[t]-mx);z+=scores[t];}for(int t=0;t<T;t++)scores[t]/=z;}
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<T;t++)a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k);ctx[(size_t)h*V+v]=a*(fmt?wscale[row]:1.f);}
}

template<typename KT>
__global__ static void attention_absorb_batch_kernel(float *ctx,const float *q,
        const KT *latent,const KT *rope,const void *weights,const float *wscale,
        int fmt,int S,int H,int Q,int R,int V,int K,int T,float scale){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=T-S+s+1,rbase=h*(Q+V);
    if(s>=S||nt<1)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          (fmt?wscale[rbase+d]:1.f);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;const KT *lt=latent+(size_t)t*K;
        const KT *rt=rope+(size_t)t*R;for(int k=0;k<K;k++)a+=qa[k]*shf(lt+k);
        for(int d=0;d<R;d++)a+=qs[Q+d]*shf(rt+d);scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    /* BARRIERA tra lettura di red[0] (mx) e la sua riscrittura sotto: senza,
     * un warp veloce scrive red[tid]=somma prima che uno lento legga il MAX
     * -> softmax incoerente nel blocco. Seme del nondeterminismo run-to-run
     * (±0.5 sui logit a 2.7k ctx) inseguito in PERF-QUEUE. */
    float mx=red[0];__syncthreads();
    local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++)
        a+=scores[t]*shf(latent+(size_t)t*K+k);cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k);
        ctx[((size_t)s*H+h)*V+v]=a*(fmt?wscale[row]:1.f);}
}

/* Independent KV sequence per row. latent/rope are packed as [S,T,*], while
 * lengths selects the valid prefix for each row. */
__global__ static void attention_absorb_ragged_kernel(float *ctx,const float *q,
        const float *latent,const float *rope,const int *lengths,
        const void *weights,const float *wscale,int fmt,int S,int H,int Q,int R,
        int V,int K,int T,float scale){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=lengths[s],rbase=h*(Q+V);
    if(s>=S||nt<1||nt>T)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    const float *ls=latent+(size_t)s*T*K,*rs=rope+(size_t)s*T*R;
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          (fmt?wscale[rbase+d]:1.f);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;const float *lt=ls+(size_t)t*K;
        const float *rt=rs+(size_t)t*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];__syncthreads();    /* stessa barriera anti-race del kernel batch */
    local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++)a+=scores[t]*ls[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k);
        ctx[((size_t)s*H+h)*V+v]=a*(fmt?wscale[row]:1.f);}
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(cudaMalloc(ptr, bytes), "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}

static int reserve_bytes(void **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes) return 1; if(*ptr) cudaFree(*ptr); *ptr=nullptr; *cap=0;
    if(!cuda_ok(cudaMalloc(ptr,bytes),"descriptor allocation")) return 0; *cap=bytes; return 1;
}

/* NUMA node of a GPU's PCIe root (sysfs), cached per CUDA ordinal; -1 = unknown. */
static int dev_numa_node(int device){
#ifdef __linux__
    static int cache[16]; static bool have[16];
    if(device<0||device>=16) return -1;
    if(have[device]) return cache[device];
    have[device]=true; cache[device]=-1;
    char bus[32];
    if(cudaDeviceGetPCIBusId(bus,sizeof(bus),device)==cudaSuccess){
        for(char *p=bus;*p;p++) *p=tolower(*p);
        char path[96]; snprintf(path,sizeof(path),"/sys/bus/pci/devices/%s/numa_node",bus);
        FILE *f=fopen(path,"r");
        if(f){ int n=-1; if(fscanf(f,"%d",&n)==1) cache[device]=n; fclose(f); }
    }
    return cache[device];
#else
    (void)device; return -1;
#endif
}

/* Pinned staging feeds one specific GPU: bind it to that GPU's local node so the
 * DMA and the CPU memcpy into it stay off the socket interconnect.  Raw mbind as
 * in glm.c's COLI_NUMA (no libnuma); COLI_NUMA_STAGING=0 disables. */
static void bind_local(void *p,size_t bytes,int device){
#ifdef __linux__
    static int on=-1;
    if(on<0){ const char *e=getenv("COLI_NUMA_STAGING"); on=!(e&&!atoi(e)); }
    int node=dev_numa_node(device);
    if(!on||node<0||node>63||!p) return;
    unsigned long mask=1UL<<node;
    long pg=sysconf(_SC_PAGESIZE);
    uintptr_t a=(uintptr_t)p & ~(uintptr_t)(pg-1);
    size_t len=((uintptr_t)p+bytes+pg-1-a) & ~(uintptr_t)(pg-1);
    syscall(SYS_mbind,a,len,2/*MPOL_BIND*/,&mask,
            (unsigned long)65,(unsigned)2/*MPOL_MF_MOVE*/);
#else
    (void)p;(void)bytes;(void)device;
#endif
}

static int reserve_pinned(float **ptr,size_t *cap,size_t bytes,int device){
    if(*cap>=bytes)return 1;if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned staging allocation"))return 0;
    bind_local(*ptr,bytes,device);
    *cap=bytes;return 1;
}

extern "C" int coli_cuda_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        cudaDeviceProp prop{};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        ctx->compute_major=prop.major;ctx->compute_minor=prop.minor;ctx->integrated=prop.integrated;
        if(cudaDeviceGetAttribute(&ctx->smem_optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                  device)!=cudaSuccess) ctx->smem_optin=49152;
        if(!cuda_ok(cudaStreamCreateWithFlags(&ctx->stream,cudaStreamNonBlocking),"stream creation")){
            g_nctx=0;return 0;
        }
        g_nctx++;
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" void coli_cuda_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        if (ctx->x) cudaFree(ctx->x);
        if (ctx->y) cudaFree(ctx->y);
        if (ctx->gate) cudaFree(ctx->gate);
        if (ctx->up) cudaFree(ctx->up);
        if (ctx->qx) cudaFree(ctx->qx);
        if (ctx->qscale) cudaFree(ctx->qscale);
        if(ctx->aq)cudaFree(ctx->aq);if(ctx->al)cudaFree(ctx->al);if(ctx->ar)cudaFree(ctx->ar);if(ctx->ac)cudaFree(ctx->ac);
        if(ctx->pf_q)cudaFree(ctx->pf_q);if(ctx->pf_c)cudaFree(ctx->pf_c);if(ctx->pf_s)cudaFree(ctx->pf_s);
        if(ctx->accum)cudaFree(ctx->accum);if(ctx->wrow_d)cudaFree(ctx->wrow_d);
        for(int b=0;b<44;b++) if(ctx->pipe_buf[b]) cudaFree(ctx->pipe_buf[b]);
        if (ctx->host_x) cudaFreeHost(ctx->host_x);
        if (ctx->host_y) cudaFreeHost(ctx->host_y);
        if (ctx->group_ev_init) cudaEventDestroy(ctx->group_ev);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        if (ctx->group_desc) cudaFree(ctx->group_desc);
        ctx->x = ctx->y = ctx->gate = ctx->up = nullptr;
        ctx->qx=nullptr; ctx->qscale=nullptr;
        ctx->aq=ctx->al=ctx->ar=ctx->ac=nullptr;
        ctx->cvt=nullptr; ctx->cvt_cap=0;
        ctx->pf_q=ctx->pf_c=ctx->pf_s=nullptr;
        ctx->accum=ctx->wrow_d=nullptr;
        ctx->host_x=ctx->host_y=nullptr;ctx->stream=nullptr;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->qx_cap=ctx->qscale_cap=0;
        ctx->aq_cap=ctx->al_cap=ctx->ar_cap=ctx->ac_cap=0;
        ctx->pf_q_cap=ctx->pf_c_cap=ctx->pf_s_cap=0;
        ctx->accum_cap=ctx->wrow_cap=0;ctx->group_ev=nullptr;ctx->group_ev_init=ctx->accum_pending=0;
        ctx->host_x_cap=ctx->host_y_cap=0;
        ctx->group_desc=nullptr; ctx->group_desc_cap=0;
    }
    g_nctx = 0;
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

extern "C" int coli_cuda_device_is_integrated(int device) {
    DeviceContext *ctx = find_ctx(device);
    return ctx ? ctx->integrated : 0;
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                        double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    if(calls) *calls=g_group_calls; if(experts) *experts=g_group_experts; if(rows) *rows=g_group_rows;
    if(h2d_ms) *h2d_ms=g_group_h2d_ms; if(kernel_ms) *kernel_ms=g_group_kernel_ms;
    if(d2h_ms) *d2h_ms=g_group_d2h_ms;
}

extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!tensor || !weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;
    if (*tensor) {
        ColiCudaTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device;
    }
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation") ||
        !cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    if(fmt==2){offset_to_signed_s4<<<(unsigned)((t->weight_bytes+255)/256),256>>>((uint8_t*)t->weights,t->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight conversion")){coli_cuda_tensor_free(t);return 0;}}
    if (fmt) {
        if (!cuda_ok(cudaMalloc(&t->scales, (size_t)O * sizeof(float)), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, (size_t)O * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);
    *tensor = t;
    return 1;
}

extern "C" int coli_cuda_tensor_wrap_host(ColiCudaTensor **tensor,
                                           const void *weights,const float *scales,
                                           int fmt,int I,int O,int device){
    DeviceContext *ctx=find_ctx(device);
    if(!tensor||*tensor||!weights||I<1||O<1||!select_ctx(ctx))return 0;
    size_t rb=row_bytes(fmt,I);
    if(!rb||(fmt&&!scales))return 0;
    int pageable=0,host_pt=0;
    if(cudaDeviceGetAttribute(&pageable,cudaDevAttrPageableMemoryAccess,device)!=cudaSuccess||
       cudaDeviceGetAttribute(&host_pt,cudaDevAttrPageableMemoryAccessUsesHostPageTables,device)!=cudaSuccess||
       !pageable||!host_pt)return 0;
    ColiCudaTensor *t=static_cast<ColiCudaTensor*>(std::calloc(1,sizeof(*t)));
    if(!t)return 0;
    t->weights=const_cast<void*>(weights);t->scales=const_cast<float*>(scales);
    t->weight_bytes=rb*(size_t)O;t->fmt=fmt;t->I=I;t->O=O;t->device=device;t->host_backed=1;
    *tensor=t;
    return 1;
}

extern "C" int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                                          const void *weights,
                                          const float *scales) {
    if (!tensor || tensor->host_backed || !weights || (tensor->fmt && !scales)) return 0;
    DeviceContext *ctx=find_ctx(tensor->device);
    if (!select_ctx(ctx)) return 0;
    if (!cuda_ok(cudaMemcpy(tensor->weights,weights,tensor->weight_bytes,
                            cudaMemcpyHostToDevice),"tensor refresh")) return 0;
    if(tensor->fmt==2){
        offset_to_signed_s4<<<(unsigned)((tensor->weight_bytes+255)/256),256>>>(
            (uint8_t*)tensor->weights,tensor->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight refresh")) return 0;
    }
    return !tensor->fmt || cuda_ok(cudaMemcpy(tensor->scales,scales,
        (size_t)tensor->O*sizeof(float),cudaMemcpyHostToDevice),"scale refresh");
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device) {
    if (S < 1 || !coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0;
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "input upload")) return 0;
    dim3 grid((unsigned)O, (unsigned)S);
    quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights, t->scales, fmt, S, I, O, rb);
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                      ColiCudaTensor *down, float *y,
                                      const float *x, int S) {
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;
    int D = gate->I, I = gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float);
    size_t yb=(size_t)S*D*sizeof(float);
    if (!reserve(&ctx->x,&ctx->x_cap,xb) || !reserve(&ctx->y,&ctx->y_cap,yb) ||
        !reserve(&ctx->gate,&ctx->gate_cap,ib) || !reserve(&ctx->up,&ctx->up_cap,ib)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice),"expert input upload")) return 0;
    dim3 hidden_grid((unsigned)I,(unsigned)S), output_grid((unsigned)D,(unsigned)S);
    quant_matmul<<<hidden_grid,256>>>(ctx->gate,ctx->x,gate->weights,gate->scales,
        gate->fmt,S,D,I,row_bytes(gate->fmt,D));
    quant_matmul<<<hidden_grid,256>>>(ctx->up,ctx->x,up->weights,up->scales,
        up->fmt,S,D,I,row_bytes(up->fmt,D));
    size_t n=(size_t)S*I;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(ctx->gate,ctx->up,n);
    quant_matmul<<<output_grid,256>>>(ctx->y,ctx->gate,down->weights,down->scales,
        down->fmt,S,I,D,row_bytes(down->fmt,I));
    if (!cuda_ok(cudaGetLastError(),"expert MLP launch") ||
        !cuda_ok(cudaMemcpy(y,ctx->y,yb,cudaMemcpyDeviceToHost),"expert output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate,ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S){
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=2||up->fmt!=2||down->fmt!=2||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx)||ctx->compute_major<7)return 0;
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb,ctx->device)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb,ctx->device))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "shared w4a16 input upload"))return 0;
    dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
    dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
    w4a16_gate_up<<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,
        (const uint8_t*)gate->weights,(const uint8_t*)up->weights,gate->scales,up->scales,S,D,I);
    silu_mul<<<(unsigned)(((size_t)S*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)S*I);
    w4a16_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,(const uint8_t*)down->weights,down->scales,S,I,D);
    if(!cuda_ok(cudaGetLastError(),"shared w4a16 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "shared w4a16 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"shared w4a16 synchronize"))return 0;
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}

__global__ static void gemv_q4(float *__restrict__ y, const float *__restrict__ x,
                               const uint8_t *__restrict__ w, const float *__restrict__ scales,
                               int I, int O);

/* Grouped decode gemv (rows==1 per expert): block = (rows8, expert).  The
 * activation row is staged to smem ONCE per block and feeds BOTH the gate and
 * up projections (dual), so weight bandwidth dominates.  Same signed-nibble
 * decode as gemv_q4. */
__global__ static void group_accum_kernel(float *__restrict__ acc,
        const float *__restrict__ y, const float *__restrict__ w,
        const int *__restrict__ tok, int nrows, int S, int D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= D) return;
    float a[4] = {0,0,0,0};
    for (int rr = 0; rr < nrows; rr++) a[tok[rr]] += w[rr] * y[(size_t)rr * D + t];
    for (int s = 0; s < S; s++) acc[(size_t)s * D + t] = a[s];
}

template<int RMAX> __global__ static void grouped_gemv_q4_dual(float *__restrict__ gate_out,
        float *__restrict__ up_out, const float *__restrict__ x_all,
        const GroupDesc *__restrict__ g, int I_in, int O) {
    const GroupDesc d = g[blockIdx.y];
    const int r = d.rows;                      /* <=4, contiguous from d.offset */
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row  = blockIdx.x * 8 + warp;
    extern __shared__ char shmem[];
    __half *sx = reinterpret_cast<__half *>(shmem);
    const float *x = x_all + (size_t)d.offset * I_in;
    for (int i = threadIdx.x; i < r * I_in; i += 256) sx[i] = __float2half(x[i]);
    __syncthreads();
    if (row >= O) return;
    const int rb = I_in >> 1;
    const uint8_t *gp = (const uint8_t *)d.g + (size_t)row * rb;
    const uint8_t *up = (const uint8_t *)d.u + (size_t)row * rb;
    float sg[RMAX], su[RMAX];
    #pragma unroll
    for (int j = 0; j < RMAX; j++) { sg[j] = 0; su[j] = 0; }
    for (int b = lane * 4; b < rb; b += 128) {
        uint32_t pg = *reinterpret_cast<const uint32_t *>(gp + b);
        uint32_t pu = *reinterpret_cast<const uint32_t *>(up + b);
        const int e = b * 2;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float wg=decode_group_s4((int)((pg>>(4*k))&0xF),d.go);
            float wu=decode_group_s4((int)((pu>>(4*k))&0xF),d.uo);
            #pragma unroll
            for (int j = 0; j < RMAX; j++) if (RMAX == 1 || j < r) {
                float xv = __half2float(sx[j * I_in + e + k]);
                sg[j] += xv * wg; su[j] += xv * wu;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < RMAX; j++)
        for (int o = 16; o; o >>= 1) {
            sg[j] += __shfl_down_sync(0xFFFFFFFF, sg[j], o);
            su[j] += __shfl_down_sync(0xFFFFFFFF, su[j], o);
        }
    if (lane == 0) for (int j = 0; j < r; j++) {
        gate_out[(size_t)(d.offset + j) * O + row] = sg[j] * d.gs[row];
        up_out  [(size_t)(d.offset + j) * O + row] = su[j] * d.us[row];
    }
}

template<int RMAX> __global__ static void grouped_gemv_q4_down(float *__restrict__ y,
        const float *__restrict__ in_all, const GroupDesc *__restrict__ g,
        int I_in, int O) {
    const GroupDesc d = g[blockIdx.y];
    const int r = d.rows;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row  = blockIdx.x * 8 + warp;
    extern __shared__ char shmem[];
    __half *sx = reinterpret_cast<__half *>(shmem);
    const float *x = in_all + (size_t)d.offset * I_in;
    for (int i = threadIdx.x; i < r * I_in; i += 256) sx[i] = __float2half(x[i]);
    __syncthreads();
    if (row >= O) return;
    const int rb = I_in >> 1;
    const uint8_t *rp = (const uint8_t *)d.d + (size_t)row * rb;
    float sum[RMAX];
    #pragma unroll
    for (int j = 0; j < RMAX; j++) sum[j] = 0;
    for (int b = lane * 4; b < rb; b += 128) {
        uint32_t p = *reinterpret_cast<const uint32_t *>(rp + b);
        const int e = b * 2;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float w=decode_group_s4((int)((p>>(4*k))&0xF),d.dof);
            #pragma unroll
            for (int j = 0; j < RMAX; j++) if (RMAX == 1 || j < r)
                sum[j] += __half2float(sx[j * I_in + e + k]) * w;
        }
    }
    #pragma unroll
    for (int j = 0; j < RMAX; j++)
        for (int o = 16; o; o >>= 1) sum[j] += __shfl_down_sync(0xFFFFFFFF, sum[j], o);
    if (lane == 0) for (int j = 0; j < r; j++)
        y[(size_t)(d.offset + j) * O + row] = sum[j] * d.ds[row];
}

extern "C" int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                                        ColiCudaTensor *const *ups,
                                        ColiCudaTensor *const *downs,
                                        const int *rows, int count,
                                        float *y, const float *x,
                                        const float *wrow, const int *tokrow,
                                        int S_tok, int accum_ok) {
    if (!gates || !ups || !downs || !rows || !x || !y || count < 1) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0;
    GroupDesc host[64]; if(count>64) return 0;
    int all_s4=1,all_device=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,
                 g->host_backed,u->host_backed,d->host_backed};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        all_device&=!g->host_backed&&!u->host_backed&&!d->host_backed;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    int async=!getenv("COLI_CUDA_ASYNC")||atoi(getenv("COLI_CUDA_ASYNC"));
    if(async&&(!reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb,ctx->device)||
               !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb,ctx->device)))return 0;
    cudaError_t copy_desc=async?cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                                cudaMemcpyHostToDevice,ctx->stream)
                               :cudaMemcpy(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_desc,"expert group descriptors"))return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    cudaEvent_t ev[4]={};
    if(profile) for(int i=0;i<4;i++) if(!cuda_ok(cudaEventCreate(&ev[i]),"profile event")) profile=0;
    if(profile) cudaEventRecord(ev[0],ctx->stream);
    if(async)std::memcpy(ctx->host_x,x,xb);
    cudaError_t copy_x=async?cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream)
                            :cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_x,"expert group input upload")) return 0;
    if(profile) cudaEventRecord(ev[1],ctx->stream);
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    int tc=getenv("COLI_CUDA_TC_INT4")&&atoi(getenv("COLI_CUDA_TC_INT4"));
    tc=tc&&all_s4&&all_device&&D%32==0&&I%32==0&&D%8==0&&I%8==0;
    int tc_min=getenv("COLI_CUDA_TC_MIN_ROWS")?atoi(getenv("COLI_CUDA_TC_MIN_ROWS")):8;
    for(int c=0;c<count&&tc;c++)tc=rows[c]>=tc_min;
    if(tc){
        size_t qb=(size_t)(total+7)*(size_t)(D>I?D:I)/2;
        if(!reserve_bytes((void**)&ctx->qx,&ctx->qx_cap,qb)||
           !reserve(&ctx->qscale,&ctx->qscale_cap,(size_t)(total+7)*sizeof(float)))return 0;
        cudaMemsetAsync(ctx->qx,0,qb,ctx->stream);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->x,total,D);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->gate,ctx->qx,ctx->qscale,dev,D,I,0);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->up,ctx->qx,ctx->qscale,dev,D,I,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->gate,total,I);
        grouped_s4_wmma<<<dim3((unsigned)((D+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->y,ctx->qx,ctx->qscale,dev,I,D,2);
    }else if(all_s4&&all_device&&ctx->compute_major>=7&&getenv("COLI_CUDA_TC_W4A16")&&
             atoi(getenv("COLI_CUDA_TC_W4A16"))){
        /* W4A16 Tensor Core per gruppo: attivazioni fp16 per tile (lossless al
         * contrario del path W4A4), un lancio per expert dentro lo stream —
         * l'overhead di lancio e' trascurabile rispetto ai GEMM. */
        int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
        int off16=0;
        for(int c=0;c<count;c++){
            int r=rows[c];
            float *g16=ctx->gate+(size_t)off16*I,*u16=ctx->up+(size_t)off16*I;
            float *x16=ctx->x+(size_t)off16*D,*y16=ctx->y+(size_t)off16*D;
            if(r>=tc16_min){
                dim3 hg16((unsigned)((I+63)/64),(unsigned)((r+15)/16));
                dim3 og16((unsigned)((D+63)/64),(unsigned)((r+15)/16));
                w4a16_gate_up<<<hg16,256,0,ctx->stream>>>(g16,u16,x16,
                    (const uint8_t*)host[c].g,(const uint8_t*)host[c].u,host[c].gs,host[c].us,r,D,I);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                w4a16_matmul<<<og16,128,0,ctx->stream>>>(y16,g16,
                    (const uint8_t*)host[c].d,host[c].ds,r,I,D);
            }else{
                /* piccoli batch: tile TC quasi vuoti + overhead di lancio — il
                 * kernel naive per-elemento resta piu' veloce (misurato in decode) */
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(g16,x16,
                    host[c].g,host[c].gs,host[c].gf,r,D,I,row_bytes(host[c].gf,D));
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(u16,x16,
                    host[c].u,host[c].us,host[c].uf,r,D,I,row_bytes(host[c].uf,D));
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                quant_matmul<<<dim3((unsigned)D,(unsigned)r),256,0,ctx->stream>>>(y16,g16,
                    host[c].d,host[c].ds,host[c].df,r,I,D,row_bytes(host[c].df,I));
            }
            off16+=r;
        }
    }else if(all_s4&&D%8==0&&I%8==0&&max_rows<=4&&(size_t)max_rows*D*sizeof(__half)<=49152&&
             (!getenv("COLI_CUDA_DECODE_GEMV")||atoi(getenv("COLI_CUDA_DECODE_GEMV")))){
        /* Decode shape (one row per expert): the grouped block-per-output kernels
         * reach ~14% of HBM peak here; the warp-per-8-rows gemv_q4 (already
         * validated on these same uploaded tensors by the fused chain) is far
         * denser.  A few extra launches per call, all stream-ordered. */
        size_t smem_h=(size_t)max_rows*D*sizeof(__half), smem_d=(size_t)max_rows*I*sizeof(__half);
        if(max_rows==1)
            grouped_gemv_q4_dual<1><<<dim3(((unsigned)I+7)/8,(unsigned)count),256,smem_h,ctx->stream>>>(
                ctx->gate,ctx->up,ctx->x,dev,D,I);
        else
            grouped_gemv_q4_dual<4><<<dim3(((unsigned)I+7)/8,(unsigned)count),256,smem_h,ctx->stream>>>(
                ctx->gate,ctx->up,ctx->x,dev,D,I);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        if(max_rows==1)
            grouped_gemv_q4_down<1><<<dim3(((unsigned)D+7)/8,(unsigned)count),256,smem_d,ctx->stream>>>(
                ctx->y,ctx->gate,dev,I,D);
        else
            grouped_gemv_q4_down<4><<<dim3(((unsigned)D+7)/8,(unsigned)count),256,smem_d,ctx->stream>>>(
                ctx->y,ctx->gate,dev,I,D);
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual)grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else{
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        }
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else{
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }
    if(wrow&&tokrow&&accum_ok&&all_s4&&max_rows<=4&&S_tok>=1&&S_tok<=4){
        /* Device-side weighted accumulate: no y download, no sync.  The caller
         * orders the NULL stream behind group_ev via _collect before any other
         * write touches the residual. */
        /* wrow_d packs the fp32 weights then the int32 token rows */
        if(reserve(&ctx->accum,&ctx->accum_cap,(size_t)S_tok*D*sizeof(float))&&
           reserve(&ctx->wrow_d,&ctx->wrow_cap,(size_t)total*(sizeof(float)+sizeof(int)))&&
           cuda_ok(cudaMemcpyAsync(ctx->wrow_d,wrow,(size_t)total*sizeof(float),
                                   cudaMemcpyHostToDevice,ctx->stream),"group weights upload")&&
           cuda_ok(cudaMemcpyAsync((char*)ctx->wrow_d+(size_t)total*sizeof(float),tokrow,
                                   (size_t)total*sizeof(int),
                                   cudaMemcpyHostToDevice,ctx->stream),"group tokens upload")){
            group_accum_kernel<<<((unsigned)D+255)/256,256,0,ctx->stream>>>(
                ctx->accum,ctx->y,ctx->wrow_d,
                (const int*)((const char*)ctx->wrow_d+(size_t)total*sizeof(float)),total,S_tok,D);
            if(!ctx->group_ev_init){
                if(cuda_ok(cudaEventCreateWithFlags(&ctx->group_ev,cudaEventDisableTiming),
                           "group event")) ctx->group_ev_init=1;
            }
            if(ctx->group_ev_init&&
               cuda_ok(cudaEventRecord(ctx->group_ev,ctx->stream),"group event record")&&
               cuda_ok(cudaGetLastError(),"group accum launch")){
                ctx->accum_pending=1;
                if(profile) for(int i=0;i<4;i++) cudaEventDestroy(ev[i]);
                { std::lock_guard<std::mutex> lock(g_group_stats_mu);
                  g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
                return 2;
            }
        }
        /* accumulate setup failed: fall through to the host path */
    }
    if(profile) cudaEventRecord(ev[2],ctx->stream);
    if(!async&&!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
    cudaError_t copy_y=async?cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream)
                            :cudaMemcpy(y,ctx->y,xb,cudaMemcpyDeviceToHost);
    if(!cuda_ok(cudaGetLastError(),"expert group launch")||!cuda_ok(copy_y,"expert group output download"))return 0;
    if(async){if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
        std::memcpy(y,ctx->host_y,xb);}
    if(profile){
        cudaEventRecord(ev[3],ctx->stream); cudaEventSynchronize(ev[3]); float a=0,b=0,c=0;
        cudaEventElapsedTime(&a,ev[0],ev[1]); cudaEventElapsedTime(&b,ev[1],ev[2]);
        cudaEventElapsedTime(&c,ev[2],ev[3]);
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_h2d_ms+=a; g_group_kernel_ms+=b; g_group_d2h_ms+=c; }
        for(int i=0;i<4;i++) cudaEventDestroy(ev[i]);
    }
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
    return 1;
}


extern "C" int coli_cuda_attention_absorb(ColiCudaTensor *w,float *ctx,const float *q,
                                            const float *latent,const float *rope,int H,int Q,
                                            int R,int V,int K,int T,float scale){
    if(!w||!ctx||!q||!latent||!rope||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>4096||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention rope upload"))return 0;
    size_t shared=(size_t)(2*K+T)*sizeof(float);
    attention_absorb_kernel<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,dc->ar,w->weights,w->scales,
        w->fmt,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"attention absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"attention context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention synchronize"))return 0;
    return 1;
}

/* Il tetto T dell'absorb-batch e' la shared memory ((2K+T+256) float), non un
 * limite algoritmico: sopra i 48KB di default si chiede l'opt-in del device
 * (fino a 227KB su Hopper -> T fino a ~56k).  Ritorna 0 se nemmeno l'opt-in
 * basta — il chiamante ripiega sul percorso CPU come prima. */
static int absorb_smem_ok(DeviceContext *dc,int K,int T,size_t *shared,
                          const void *kernel,int *attr_set){
    size_t need=(size_t)(2*K+T+256)*sizeof(float);
    *shared=need;
    if(need<=49152) return 1;
    if(need>(size_t)dc->smem_optin) return 0;
    if(!*attr_set){
        if(cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,dc->smem_optin)!=cudaSuccess)
            return 0;
        *attr_set=1;
    }
    return 1;
}
static int attention_absorb_batch_run(ColiCudaTensor *w,ColiCudaTensor *proj,float *out,
        const float *q,const float *latent,const float *rope,int S,int H,int Q,int R,int V,
        int K,int T,float scale){
    if(!w||!out||!q||!latent||!rope||S<1||H<1||Q<1||R<1||V<1||K<1||K>512||
       T<S||w->I!=K||w->O!=H*(Q+V))return 0;
    if(proj&&(proj->device!=w->device||proj->I!=H*V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention batch q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention batch latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention batch rope upload"))return 0;
    size_t shared;
    if(!absorb_smem_ok(dc,K,T,&shared,(const void*)attention_absorb_batch_kernel<float>,
                       &dc->absorb_attr_set))return 0;
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,
        dc->ar,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"attention batch launch"))return 0;
    const float *src=dc->ac;size_t ob=cb;
    if(proj){
        ob=(size_t)S*proj->O*sizeof(float);if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
        quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
            proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I));
        if(!cuda_ok(cudaGetLastError(),"attention o_proj launch"))return 0;src=dc->y;
    }
    if(!cuda_ok(cudaMemcpyAsync(out,src,ob,cudaMemcpyDeviceToHost,dc->stream),
                               proj?"attention projected output download":"attention batch context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention batch synchronize"))return 0;
    return 1;
}

extern "C" int coli_cuda_attention_absorb_batch(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent,const float *rope,int S,int H,int Q,int R,int V,int K,int T,
        float scale){
    return attention_absorb_batch_run(w,nullptr,ctx,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_batch(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const float *latent,const float *rope,int S,int H,int Q,
        int R,int V,int K,int T,float scale){
    return attention_absorb_batch_run(w,proj,out,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_ragged(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!proj||!out||!q||!latent||!rope||!lengths||S<1||S>512||T<1||T>512||
       H<1||Q<1||R<1||V<1||K<1||K>512||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    size_t ln=(size_t)S*T*K,rn=(size_t)S*T*R;
    float *lh=(float*)std::calloc(ln,sizeof(float)),*rh=(float*)std::calloc(rn,sizeof(float));
    if(!lh||!rh){std::free(lh);std::free(rh);return 0;}
    for(int s=0;s<S;s++){
        if(lengths[s]<1||lengths[s]>T){std::free(lh);std::free(rh);return 0;}
        std::memcpy(lh+(size_t)s*T*K,latent[s],(size_t)lengths[s]*K*sizeof(float));
        std::memcpy(rh+(size_t)s*T*R,rope[s],(size_t)lengths[s]*R*sizeof(float));
    }
    DeviceContext *dc=find_ctx(w->device);
    if(!select_ctx(dc)){std::free(lh);std::free(rh);return 0;}
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float),lb=ln*sizeof(float),rb=rn*sizeof(float);
    size_t cb=(size_t)S*H*V*sizeof(float),ob=(size_t)S*proj->O*sizeof(float);
    int ok=reserve(&dc->aq,&dc->aq_cap,qb)&&reserve(&dc->al,&dc->al_cap,lb)&&
           reserve(&dc->ar,&dc->ar_cap,rb)&&reserve(&dc->ac,&dc->ac_cap,cb)&&
           reserve(&dc->y,&dc->y_cap,ob)&&
           reserve_bytes(&dc->group_desc,&dc->group_desc_cap,(size_t)S*sizeof(int));
    if(ok)ok=cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"ragged q upload")&&
             cuda_ok(cudaMemcpyAsync(dc->al,lh,lb,cudaMemcpyHostToDevice,dc->stream),"ragged latent upload")&&
             cuda_ok(cudaMemcpyAsync(dc->ar,rh,rb,cudaMemcpyHostToDevice,dc->stream),"ragged rope upload")&&
             cuda_ok(cudaMemcpyAsync(dc->group_desc,lengths,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged lengths upload");
    std::free(lh);std::free(rh);if(!ok)return 0;
    size_t shared;
    if(!absorb_smem_ok(dc,K,T,&shared,(const void*)attention_absorb_ragged_kernel,
                       &dc->ragged_attr_set))return 0;
    attention_absorb_ragged_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,dc->ar,
        (const int*)dc->group_desc,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale);
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I));
    return cuda_ok(cudaGetLastError(),"ragged attention launch")&&
           cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"ragged output download")&&
           cuda_ok(cudaStreamSynchronize(dc->stream),"ragged attention synchronize");
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        size_t bytes = tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (!tensor->host_backed && tensor->weights) cudaFree(tensor->weights);
    if (!tensor->host_backed && tensor->scales) cudaFree(tensor->scales);
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    return tensor ? tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0) : 0;
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}

/* ==== resident-pipeline primitives (Inc.0, 2026-07-13) ====
 * Device-side building blocks so the residual stream can stay on the layer's
 * home device across a whole layer. Control flow stays on CPU; only the data
 * plane lives here. All entry points take DEVICE pointers (no transfers) —
 * the caller owns staging via the pipe buffer API below. */

__global__ static void pipe_rmsnorm_rows(float *y,const float *x,const float *w,
                                         int D,float eps,int xstride,int ystride){
    const float *xr=x+(size_t)blockIdx.x*xstride; float *yr=y+(size_t)blockIdx.x*ystride;
    __shared__ double sh[256];
    double a=0; for(int i=threadIdx.x;i<D;i+=blockDim.x){ double v=xr[i]; a+=v*v; }
    sh[threadIdx.x]=a; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
    float r=rsqrtf((float)(sh[0]/D)+eps);
    for(int i=threadIdx.x;i<D;i+=blockDim.x) yr[i]=xr[i]*r*w[i];
}

/* RoPE interleaved, identical math to glm.c rope_interleave. One block per row;
 * row layout: v + row*stride + offset holds R floats. pos index = row/heads
 * (heads=1 for k_rot rows, heads=H for [S,H,qh] query rows). */
__global__ static void pipe_rope_rows(float *v,const int *pos,int pos_base,int stride,
                                      int offset,int R,int heads,float theta){
    float *p=v+(size_t)blockIdx.x*stride+offset;
    int half=R/2, ps=pos?pos[blockIdx.x/heads]:pos_base+(int)(blockIdx.x/heads);
    __shared__ float in[256];
    for(int j=threadIdx.x;j<R;j+=blockDim.x) in[j]=p[j];
    __syncthreads();
    for(int j=threadIdx.x;j<half;j+=blockDim.x){
        float inv=__powf(theta,-2.0f*j/R);
        float ang=ps*inv, cs=__cosf(ang), sn=__sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        p[j]=a*cs-b*sn; p[half+j]=b*cs+a*sn;
    }
}

__global__ static void pipe_add_n(float *x,const float *t,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]+=t[i];
}

/* Fixed-order partial merge: block b adds partial row b into x row rows[b].
 * Target rows are unique by construction (CPU pre-sums per token), so no
 * atomics — the 9.20.7 lesson. */
__global__ static void pipe_rows_add(float *x,const float *partial,const int *rows,
                                     int D){
    float *xr=x+(size_t)rows[blockIdx.x]*D;
    const float *pr=partial+(size_t)blockIdx.x*D;
    for(int i=threadIdx.x;i<D;i+=blockDim.x) xr[i]+=pr[i];
}

/* scratch persistente per (device,slot): cresce e resta — niente cudaMalloc/Free
 * per layer (78 x ~10 alloc/richiesta erano puro churn). */
extern "C" float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    DeviceContext *ctx=find_ctx(device);
    if(slot<0||slot>=44||!select_ctx(ctx)) return NULL;
    if(!reserve(&ctx->pipe_buf[slot],&ctx->pipe_cap[slot],bytes)) return NULL;
    return ctx->pipe_buf[slot];
}
__global__ static void gemv_q4(float *__restrict__ y,
                               const float *__restrict__ x,
                               const uint8_t *__restrict__ w,
                               const float *__restrict__ scales,
                               int I, int O) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * 8 + warp;

    extern __shared__ char shmem[];
    __half *sx = reinterpret_cast<__half *>(shmem);
    for (int i = threadIdx.x; i < I; i += 256)
        sx[i] = __float2half(x[i]);
    __syncthreads();

    if (row >= O) return;

    const int rb = I >> 1;
    const uint8_t *rp = w + (size_t)row * rb;
    float sum = 0.0f;

    for (int b = lane * 4; b < rb; b += 128) {
        uint32_t p = *reinterpret_cast<const uint32_t *>(rp + b);
        const int e = b * 2;
        sum += __half2float(sx[e  ]) * (float)(((int)( p        & 0xF) ^ 8) - 8);
        sum += __half2float(sx[e+1]) * (float)(((int)((p >>  4) & 0xF) ^ 8) - 8);
        sum += __half2float(sx[e+2]) * (float)(((int)((p >>  8) & 0xF) ^ 8) - 8);
        sum += __half2float(sx[e+3]) * (float)(((int)((p >> 12) & 0xF) ^ 8) - 8);
        sum += __half2float(sx[e+4]) * (float)(((int)((p >> 16) & 0xF) ^ 8) - 8);
        sum += __half2float(sx[e+5]) * (float)(((int)((p >> 20) & 0xF) ^ 8) - 8);
        sum += __half2float(sx[e+6]) * (float)(((int)((p >> 24) & 0xF) ^ 8) - 8);
        sum += __half2float(sx[e+7]) * (float)(((int)((p >> 28)      ) ^ 8) - 8);
    }

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    if (lane == 0) y[row] = sum * scales[row];
}
__global__ static void rmsnorm_kernel(float *out,
                                       const float *x,
                                       const float *__restrict__ weight,
                                       int size, float eps) {
    int tid = threadIdx.x;
    extern __shared__ float smem_rms[];

    float local_ss = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float v = x[i];
        local_ss += v * v;
    }

    smem_rms[tid] = local_ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s; s >>= 1) {
        if (tid < s) smem_rms[tid] += smem_rms[tid + s];
        __syncthreads();
    }

    float rms = rsqrtf(smem_rms[0] / (float)size + eps);
    for (int i = tid; i < size; i += blockDim.x)
        out[i] = weight[i] * (x[i] * rms);
}
__global__ static void memcpy_f32_kernel(float *__restrict__ dst,
                                          const float *__restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
template<typename KT>
__device__ static void absorption_body(
    float *__restrict__ ctx_out,
    const float *__restrict__ Q,
    const uint8_t *__restrict__ kv_b_w,
    const float *__restrict__ kv_b_s,
    int kv_b_I,
    const KT *__restrict__ Lc,
    const KT *__restrict__ Rc,
    int H, int qk_nope, int qk_rope, int vh,
    int kv_start, int pos, float attn_scale)
{
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= H) return;

    int kvl = kv_b_I;
    int qh = qk_nope + qk_rope;
    int rbase = h * (qk_nope + vh);
    const float *qp = Q + (size_t)h * qh;
    const float *qr = qp + qk_nope;
    int nt = pos + 1 - kv_start;
    int rb = kvl >> 1;

    extern __shared__ float smem[];
    float *s_qabs = smem;
    float *s_buf = smem + kvl;

    /* Phase 1: absorbed query — qabs[i] = Σ_d qp[d] * dequant(kv_b[rbase+d, i]) * scale[rbase+d] */
    for (int i = tid; i < kvl; i += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < qk_nope; d++) {
            const uint8_t *rp = kv_b_w + (size_t)(rbase + d) * rb;
            uint8_t v = rp[i >> 1];
            int n = (i & 1) ? (v >> 4) : (v & 15);
            float w = (float)((n ^ 8) - 8);
            sum += qp[d] * w * kv_b_s[rbase + d];
        }
        s_qabs[i] = sum;
    }
    __syncthreads();

    /* Phase 2+3: scores + softmax — process tokens in shared-memory-friendly chunks */
    /* We compute scores into s_buf, do softmax, then use scores for context accumulation. */
    /* For small nt (decode), this fits easily. For large nt, we use online softmax. */

    /* Online softmax + context accumulation in one pass over tokens.
     * Avoids storing all nt scores, handles arbitrarily long contexts. */
    float max_s = -1e30f;

    /* First pass: compute scores and find max (for softmax stability) */
    /* We store scores temporarily in s_buf. For nt up to ~2000, this fits in smem. */
    for (int j = tid; j < nt; j += blockDim.x) {
        int t = kv_start + j;
        const KT *Lt = Lc + (size_t)t * kvl;
        const KT *kr = Rc + (size_t)t * qk_rope;
        float a = 0.0f;
        for (int i = 0; i < kvl; i++) a += s_qabs[i] * shf(Lt + i);
        for (int d = 0; d < qk_rope; d++) a += qr[d] * shf(kr + d);
        s_buf[j] = a * attn_scale;
    }
    __syncthreads();

    /* Find max across all scores (parallel reduction) */
    float local_max = -1e30f;
    for (int j = tid; j < nt; j += blockDim.x)
        if (s_buf[j] > local_max) local_max = s_buf[j];

    /* Warp-level max reduction, then block-level */
    for (int off = 16; off >= 1; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, off));

    /* Store warp max at lane 0 into shared memory for block reduction */
    int warp_id = tid >> 5;
    int lane = tid & 31;
    __shared__ float warp_vals[8];
    if (lane == 0) warp_vals[warp_id] = local_max;
    __syncthreads();

    if (tid == 0) {
        float m = warp_vals[0];
        for (int w = 1; w < 8; w++) m = fmaxf(m, warp_vals[w]);
        warp_vals[0] = m;
    }
    __syncthreads();
    max_s = warp_vals[0];
    /* Barrier between reading the max and the sum-store below reusing
     * warp_vals[0]: a fast warp 0 could overwrite it before a slow warp
     * reads max_s (the run-to-run nondeterminism seed, see PERF-QUEUE). */
    __syncthreads();

    /* Compute exp(score - max) in place and find sum */
    float local_sum = 0.0f;
    for (int j = tid; j < nt; j += blockDim.x) {
        float e = expf(s_buf[j] - max_s);
        s_buf[j] = e;
        local_sum += e;
    }

    for (int off = 16; off >= 1; off >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, off);
    if (lane == 0) warp_vals[warp_id] = local_sum;
    __syncthreads();

    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < 8; w++) s += warp_vals[w];
        warp_vals[0] = s;
    }
    __syncthreads();
    float inv_sum = 1.0f / warp_vals[0];

    /* Normalize scores */
    for (int j = tid; j < nt; j += blockDim.x)
        s_buf[j] *= inv_sum;
    __syncthreads();

    /* Phase 4: context accumulation — clat[i] = Σ_t score[t] * Lc[t][i] */
    /* Reuse s_qabs for clat (we're done with qabs) */
    float *s_clat = s_qabs;
    for (int i = tid; i < kvl; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < nt; j++) {
            int t = kv_start + j;
            sum += s_buf[j] * shf(Lc + (size_t)t * kvl + i);
        }
        s_clat[i] = sum;
    }
    __syncthreads();

    /* Phase 5: value decompression — ctx[row] = Σ_i clat[i] * dequant(kv_b[rbase+r0v+row, i]) * scale */
    int r0v = qk_nope;
    for (int row = tid; row < vh; row += blockDim.x) {
        int krow = rbase + r0v + row;
        const uint8_t *rp = kv_b_w + (size_t)krow * rb;
        float sum = 0.0f;
        for (int i = 0; i < kvl; i++) {
            uint8_t v = rp[i >> 1];
            int n = (i & 1) ? (v >> 4) : (v & 15);
            float w = (float)((n ^ 8) - 8);
            sum += s_clat[i] * w;
        }
        ctx_out[(size_t)h * vh + row] = sum * kv_b_s[krow];
    }
}
template<typename KT>
__global__ static void absorption_kernel(
    float *__restrict__ ctx_out, const float *__restrict__ Q,
    const uint8_t *__restrict__ kv_b_w, const float *__restrict__ kv_b_s,
    int kv_b_I, const KT *__restrict__ Lc, const KT *__restrict__ Rc,
    int H, int qk_nope, int qk_rope, int vh,
    int kv_start, int pos, float attn_scale)
{
    absorption_body(ctx_out, Q, kv_b_w, kv_b_s, kv_b_I, Lc, Rc,
                    H, qk_nope, qk_rope, vh, kv_start, pos, attn_scale);
}
__global__ static void add_vec_kernel(float *dst, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i];
}
/* LayerNorm (weight+bias) in place — the DSA indexer k_norm.  One block per
 * row (grid 1 = single row, the decode chain; grid S = prefill batch). */
__global__ static void layernorm_kernel(float *xrows, const float *__restrict__ w,
                                        const float *__restrict__ b, int n, float eps) {
    float *x = xrows + (size_t)blockIdx.x * n;
    __shared__ float red[256];
    int tid = threadIdx.x;
    float s = 0;
    for (int i = tid; i < n; i += blockDim.x) s += x[i];
    red[tid] = s; __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float mean = red[0] / n;
    __syncthreads();                       /* red[] riusato sotto: barriera anti-race */
    s = 0;
    for (int i = tid; i < n; i += blockDim.x) { float d = x[i] - mean; s += d * d; }
    red[tid] = s; __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / n + eps);
    for (int i = tid; i < n; i += blockDim.x)
        x[i] = (x[i] - mean) * inv * w[i] + b[i];
}
/* DSA lightning-indexer selection scores over the device Ic shadow:
 *   isc[y][t] = wsc * Σ_h w32[y][h] * ReLU(rs * qi[y][h]·k_idx[t])
 * One query row per blockIdx.y (row y has nk0+y causal keys), qi [nh,hd]
 * (roped) and w32 [nh] of that row in smem, one thread per token.  The
 * decode chain uses grid.y==1 with nk0 = the single row's key count. */
/* Top-k ESATTO per riga sui punteggi DSA, stessa semantica del top-k host
 * (partial_select threshold; poi scan in ordine di POSIZIONE: > prima, == dopo
 * fino a keep).  Radix select a 4 passate sui byte della chiave fp32 monotona,
 * poi compattazione stabile.  Un blocco per riga (blockIdx.x), 256 thread;
 * la riga y ha nk0+y chiavi causali.  sel righe a stride topk.
 * Nota: ±0.0 hanno chiavi diverse ma confronto float uguale — con punteggi
 * ReLU-sommati un -0.0 non si presenta; nel caso, cambierebbe solo l'ordine
 * interno di un pareggio a punteggio nullo. */
__device__ static inline unsigned dsa_key(float x){
    unsigned u=__float_as_uint(x);
    return (u&0x80000000u)?~u:(u|0x80000000u);   /* mappa monotona crescente */
}
__global__ static void dsa_topk_rows(int *__restrict__ sel,
        const float *__restrict__ scores,int Tstride,int nk0,int topk){
    int y=blockIdx.x, tid=threadIdx.x, nk=nk0+y;
    int keep=nk<topk?nk:topk;
    const float *row=scores+(size_t)y*Tstride;
    int *out=sel+(size_t)y*topk;
    __shared__ int hist[256], scan[256];
    __shared__ unsigned s_prefix; __shared__ int s_rem,s_g;
    if(!tid){ s_prefix=0; s_rem=keep; s_g=0; }
    __syncthreads();
    for(int p=3;p>=0;p--){
        int shift=8*p;
        unsigned prefix=s_prefix;
        for(int i=tid;i<256;i+=blockDim.x) hist[i]=0;
        __syncthreads();
        for(int t=tid;t<nk;t+=blockDim.x){
            unsigned k=dsa_key(row[t]);
            if(p==3||(k>>(shift+8))==prefix)
                atomicAdd(&hist[(k>>shift)&255],1);
        }
        __syncthreads();
        if(!tid){
            int cum=0,rem=s_rem;
            for(int b=255;b>=0;b--){
                if(cum+hist[b]>=rem){ s_prefix=(prefix<<8)|(unsigned)b;
                    s_g+=cum; s_rem=rem-cum; break; }
                cum+=hist[b];
            }
        }
        __syncthreads();
    }
    unsigned thr=s_prefix;
    int g=s_g;                 /* count(key>thr); i pareggi da prendere: s_rem */
    int e=s_rem;
    __shared__ int s_goff,s_eoff;
    if(!tid){ s_goff=0; s_eoff=0; }
    __syncthreads();
    for(int c0=0;c0<nk;c0+=blockDim.x){
        int t=c0+tid, valid=t<nk;
        unsigned k=valid?dsa_key(row[t]):0u;
        int fg=valid&&k>thr, fe=valid&&k==thr;
        /* scan esclusivo di blocco (256): fg in scan[], fe in hist[] */
        scan[tid]=fg; hist[tid]=fe; __syncthreads();
        for(int off=1;off<256;off<<=1){
            int a=tid>=off?scan[tid-off]:0, b=tid>=off?hist[tid-off]:0;
            __syncthreads();
            scan[tid]+=a; hist[tid]+=b;
            __syncthreads();
        }
        int idxg=scan[tid]-fg, idxe=hist[tid]-fe;   /* inclusivo -> esclusivo */
        if(fg) out[s_goff+idxg]=t;
        if(fe&&s_eoff+idxe<e) out[g+s_eoff+idxe]=t;
        __syncthreads();
        if(!tid){ s_goff+=scan[255]; s_eoff+=hist[255]; }
        __syncthreads();
    }
}
template<typename KT>
__global__ static void dsa_score_kernel(float *__restrict__ isc,
        const float *__restrict__ qi, const float *__restrict__ w32,
        const KT *__restrict__ Ic, int nk0, int nh, int hd,
        float wsc, float rs, int Tstride) {
    extern __shared__ float sm[];
    int y = blockIdx.y, nk = nk0 + y;
    float *sq = sm, *sw = sm + nh * hd;
    const float *qy = qi + (size_t)y * nh * hd, *wy = w32 + (size_t)y * nh;
    for (int i = threadIdx.x; i < nh * hd; i += blockDim.x) sq[i] = qy[i];
    for (int i = threadIdx.x; i < nh; i += blockDim.x) sw[i] = wy[i];
    __syncthreads();
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nk) return;
    const KT *kt = Ic + (size_t)t * hd;
    float a = 0;
    for (int h = 0; h < nh; h++) {
        const float *qh = sq + (size_t)h * hd;
        float d = 0;
        for (int i = 0; i < hd; i++) d += qh[i] * shf(kt + i);
        d *= rs;
        if (d > 0) a += sw[h] * d;
    }
    isc[(size_t)y * Tstride + t] = a * wsc;
}

/* ---- Fused small-S attention chain (decode fast path) ----
 * The whole per-layer attention — in_ln, q/kv projections, q_a norm, latent+rope
 * KV write (device-resident), RoPE, absorbed attention, o_proj, residual add,
 * post_ln — as one uninterrupted launch chain per position, with a single sync
 * at the end (the canonical-host downloads).  Replaces ~14 pipe_* round trips
 * per layer; measured 0.32 ms/layer vs 0.75 ms for the op-by-op chain on H100.
 * int4 (fmt=2) weights only; S<=4 (decode + MTP verify), falls back otherwise. */
/* fp32 GEMV for the MoE router: one block per output row. */
__global__ static void gemv_f32_kernel(float *__restrict__ y, const float *__restrict__ x,
                                       const float *__restrict__ w, int I, int O) {
    __shared__ float red[256];
    int o = blockIdx.x;
    if (o >= O) return;
    const float *wr = w + (size_t)o * I;
    float s = 0;
    for (int i = threadIdx.x; i < I; i += blockDim.x) s += wr[i] * x[i];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[o] = red[0];
}

template<typename KT>
__global__ static void attention_absorb_sel_kernel(float *ctx,const float *q,
        const KT *latent,const KT *rope,const int *sel,int ns,
        const void *weights,const float *wscale,int fmt,
        int H,int Q,int R,int V,int K,float scale);
/* COLI_DBG_DSACHAIN=1: tempi cumulativi del punto di sync a metà catena
 * (drenaggio pipeline + download punteggi) e del top-k host.  Esposti con una
 * funzione, non con dati extern: il loader Windows risolve solo simboli-funzione. */
static double coli_dsac_t_sync=0.0, coli_dsac_t_topk=0.0;
extern "C" void coli_cuda_dsac_times(double *sync_s, double *topk_s){
    if(sync_s) *sync_s=coli_dsac_t_sync;
    if(topk_s) *topk_s=coli_dsac_t_topk;
}
static double dsac_now(void){
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

extern "C" int coli_cuda_pipe_attn_chain_v2(int device,
        float *x_dev, float *nrm_dev, float *nrm_host,
        float *kv_host_L, float *kv_host_R,
        const ColiCudaTensor *qa, const ColiCudaTensor *qb,
        const ColiCudaTensor *kva, const ColiCudaTensor *kvb,
        const ColiCudaTensor *o_proj,
        const float *w_in, const float *w_qa, const float *w_kva, const float *w_post,
        float *d_Lc, float *d_Rc,
        int D, int H, int q_lora, int kv_lora,
        int qk_nope, int qk_rope, int vh,
        int S, int pos_base, int kv_start,
        float eps, float theta, float attn_scale,
        const float *d_router, int E, float *scores_host,
        const ColiCudaTensor *shg, const ColiCudaTensor *shu,
        const ColiCudaTensor *shd, int sI, float *xn_host,
        ColiCudaDsaChain *dsa) {
    DeviceContext *ctx=find_ctx(device);
    if(!ctx||!select_ctx(ctx)) return 0;
    if(qa->fmt!=2||qb->fmt!=2||kva->fmt!=2||kvb->fmt!=2||o_proj->fmt!=2) return 0;
    int qh=qk_nope+qk_rope, kva_O=kv_lora+qk_rope;
    int T=pos_base+S-kv_start;
    /* DSA selection mode (phase 2 inc.2): sel entries are absolute positions,
     * so a shifted window is out of contract.  FULL layers additionally score
     * the shadow and refresh sel/nsel through the host top-k callback. */
    int selm = dsa && dsa->sel && dsa->nsel && dsa->topk>0;
    int dsa_kidx = selm && dsa->ix_wk && dsa->knw_dev && dsa->knb_dev &&
                   dsa->d_Ic && dsa->nh>0 && dsa->hd>0;
    int dsa_full = dsa_kidx && dsa->ix_wq && dsa->ix_wp &&
                   dsa->topk_fn && dsa->iscore_host && !dsa->score_off;
    int dsa_reuse = selm && dsa->score_off;   /* riuso temporale: k_idx sì, punteggi no */
    if(dsa && !selm) return 0;
    if(selm && kv_start!=0) return 0;
    if(dsa && dsa->topk_fn && !dsa_full && !dsa_reuse) return 0;
                                                    /* FULL inteso ma prerequisiti carenti:
                                                     * mai degradare a riuso-selezione muto */
    if(dsa_reuse && !dsa_kidx) return 0;  /* il riuso DEVE comunque accodare k_idx a Ic:
                                           * saltarlo corromperebbe le selezioni future */
    /* l'estrazione scrive l'indexer a int8 (fmt 1, scale per riga): le sue tre
     * proiezioni passano dal quant_matmul generico, che accetta fmt 1 e 2 */
    if((dsa_full||dsa_reuse) && !(dsa->ix_wk->fmt==1||dsa->ix_wk->fmt==2)){
        fprintf(stderr,"[CUDA] attn_chain: ix_wk fmt %d\n",dsa->ix_wk->fmt);
        return 0;
    }
    if(dsa_full && !((dsa->ix_wq->fmt==1||dsa->ix_wq->fmt==2)&&
                     (dsa->ix_wp->fmt==1||dsa->ix_wp->fmt==2))){
        fprintf(stderr,"[CUDA] attn_chain: ix fmt %d/%d\n",
                dsa->ix_wq->fmt,dsa->ix_wp->fmt);
        return 0;
    }
    size_t abs_smem=(size_t)(kv_lora+T)*sizeof(float)+8*sizeof(float);
    if(!selm && abs_smem>200u*1024u) return 0;   /* score window no longer smem-resident */
    float *xn  =coli_cuda_pipe_scratch(device,22,(size_t)S*D*4);
    float *qres=coli_cuda_pipe_scratch(device,23,(size_t)q_lora*4);
    float *qQ  =coli_cuda_pipe_scratch(device,24,(size_t)S*H*qh*4);
    float *comp=coli_cuda_pipe_scratch(device,25,(size_t)kva_O*4);
    float *cx  =coli_cuda_pipe_scratch(device,26,(size_t)H*vh*4);
    float *aout=coli_cuda_pipe_scratch(device,27,(size_t)S*D*4);
    if(!xn||!qres||!qQ||!comp||!cx||!aout) return 0;
    float *sc_d=NULL,*qi_d=NULL,*w32_d=NULL; int *dsel_d=NULL;
    if(dsa_full){
        sc_d =coli_cuda_pipe_scratch(device,18,(size_t)S*T*4);
        qi_d =coli_cuda_pipe_scratch(device,19,(size_t)dsa->nh*dsa->hd*4);
        w32_d=coli_cuda_pipe_scratch(device,20,(size_t)dsa->nh*4);
        if(!sc_d||!qi_d||!w32_d){ fprintf(stderr,"[CUDA] attn_chain: dsa scratch\n"); return 0; }
    }
    if(selm){
        dsel_d=(int*)coli_cuda_pipe_scratch(device,21,(size_t)S*dsa->topk*sizeof(int));
        if(!dsel_d){ fprintf(stderr,"[CUDA] attn_chain: sel scratch\n"); return 0; }
    }
    /* COLI_KV_F16: le righe nuove si calcolano in fp32 (staging 37-39), l'host
     * scarica l'fp32 ESATTO, l'ombra riceve la conversione __half */
    int f16=kv_f16_mode();
    __half *hLc=(__half*)d_Lc, *hRc=(__half*)d_Rc;
    float *lrow=NULL,*rrow=NULL,*irow=NULL;
    if(f16){
        lrow=coli_cuda_pipe_scratch(device,37,(size_t)S*kv_lora*4);
        rrow=coli_cuda_pipe_scratch(device,38,(size_t)S*qk_rope*4);
        if(!lrow||!rrow) return 0;
        if(dsa_full||dsa_reuse){
            irow=coli_cuda_pipe_scratch(device,39,(size_t)S*dsa->hd*4);
            if(!irow) return 0;
        }
    }
    size_t smem_D=(size_t)D*sizeof(__half);
    size_t smem_q=(size_t)q_lora*sizeof(__half);
    size_t smem_o=(size_t)(H*vh)*sizeof(__half);
    size_t smem_max=smem_D>smem_o?smem_D:smem_o;
    if(smem_max>48u*1024u &&
       cudaFuncSetAttribute((const void*)gemv_q4,
           cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_max)!=cudaSuccess) return 0;
    if(!selm && abs_smem>48u*1024u &&
       cudaFuncSetAttribute(f16?(const void*)absorption_kernel<__half>
                               :(const void*)absorption_kernel<float>,
           cudaFuncAttributeMaxDynamicSharedMemorySize,(int)abs_smem)!=cudaSuccess) return 0;
    /* loop 1: projections + KV/Ic writes + (FULL layers) selection scoring.
     * Rows>pos in d_Lc/d_Rc are written before the absorb reads them, but the
     * absorb of row s only attends up to pos — same math as the fused order. */
    for(int s=0;s<S;s++){
        int pos=pos_base+s;
        const float *xrow=x_dev+(size_t)s*D;
        float *xns=xn+(size_t)s*D;
        float *qQs=qQ+(size_t)s*H*qh;
        rmsnorm_kernel<<<1,256,256*sizeof(float)>>>(xns,xrow,w_in,D,eps);
        gemv_q4<<<((unsigned)q_lora+7)/8,256,smem_D>>>(qres,xns,
            (const uint8_t*)qa->weights,qa->scales,D,q_lora);
        gemv_q4<<<((unsigned)kva_O+7)/8,256,smem_D>>>(comp,xns,
            (const uint8_t*)kva->weights,kva->scales,D,kva_O);
        { int th=q_lora<256?q_lora:256;
          rmsnorm_kernel<<<1,th,(size_t)th*sizeof(float)>>>(qres,qres,w_qa,q_lora,eps); }
        float *ltgt=f16?lrow+(size_t)s*kv_lora:d_Lc+(size_t)pos*kv_lora;
        float *rtgt=f16?rrow+(size_t)s*qk_rope:d_Rc+(size_t)pos*qk_rope;
        { int th=kv_lora<256?kv_lora:256;
          rmsnorm_kernel<<<1,th,(size_t)th*sizeof(float)>>>(ltgt,comp,w_kva,kv_lora,eps); }
        { int th=qk_rope<256?qk_rope:256;
          memcpy_f32_kernel<<<(qk_rope+th-1)/th,th>>>(rtgt,comp+kv_lora,qk_rope); }
        pipe_rope_rows<<<1,128>>>(rtgt,NULL,pos,
            qk_rope,0,qk_rope,1,theta);
        if(f16){
            cvt_f32_f16<<<((unsigned)kv_lora+255)/256,256>>>(hLc+(size_t)pos*kv_lora,ltgt,kv_lora);
            cvt_f32_f16<<<((unsigned)qk_rope+255)/256,256>>>(hRc+(size_t)pos*qk_rope,rtgt,qk_rope);
        }
        gemv_q4<<<((unsigned)(H*qh)+7)/8,256,smem_q>>>(qQs,qres,
            (const uint8_t*)qb->weights,qb->scales,q_lora,H*qh);
        pipe_rope_rows<<<H,128>>>(qQs,NULL,pos,qh,qk_nope,qk_rope,H,theta);
        if(dsa_full||dsa_reuse){
            int nh=dsa->nh, hd=dsa->hd, nk=pos+1;
            float *icrow=f16?irow+(size_t)s*hd:dsa->d_Ic+(size_t)pos*hd;
            quant_matmul<<<dim3(hd,1),256>>>(icrow,xns,dsa->ix_wk->weights,
                dsa->ix_wk->scales,dsa->ix_wk->fmt,1,D,hd,
                row_bytes(dsa->ix_wk->fmt,D));
            layernorm_kernel<<<1,128>>>(icrow,dsa->knw_dev,dsa->knb_dev,hd,1e-6f);
            pipe_rope_rows<<<1,128>>>(icrow,NULL,pos,hd,0,qk_rope,1,theta);
            if(f16) cvt_f32_f16<<<((unsigned)hd+255)/256,256>>>(
                (__half*)dsa->d_Ic+(size_t)pos*hd,icrow,hd);
            if(!dsa_full) continue;   /* riuso: niente punteggi, sel arriva dal chiamante */
            quant_matmul<<<dim3(nh*hd,1),256>>>(qi_d,qres,dsa->ix_wq->weights,
                dsa->ix_wq->scales,dsa->ix_wq->fmt,1,q_lora,nh*hd,
                row_bytes(dsa->ix_wq->fmt,q_lora));
            pipe_rope_rows<<<nh,128>>>(qi_d,NULL,pos,hd,0,qk_rope,nh,theta);
            quant_matmul<<<dim3(nh,1),256>>>(w32_d,xns,dsa->ix_wp->weights,
                dsa->ix_wp->scales,dsa->ix_wp->fmt,1,D,nh,
                row_bytes(dsa->ix_wp->fmt,D));
            size_t smem_sc=(size_t)(nh*hd+nh)*sizeof(float);
            if(f16) dsa_score_kernel<<<dim3(((unsigned)nk+127)/128,1),128,smem_sc>>>(
                sc_d+(size_t)s*T,qi_d,w32_d,(const __half*)dsa->d_Ic,nk,nh,hd,
                1.f/sqrtf((float)nh),1.f/sqrtf((float)hd),T);
            else dsa_score_kernel<<<dim3(((unsigned)nk+127)/128,1),128,smem_sc>>>(
                sc_d+(size_t)s*T,qi_d,w32_d,dsa->d_Ic,nk,nh,hd,
                1.f/sqrtf((float)nh),1.f/sqrtf((float)hd),T);
        }
    }
    if(dsa_full){
        /* one mid-chain sync: score rows to host, exact CPU top-k, sel back up.
         * t_sync include il drenaggio implicito della pipeline (tutto il lavoro
         * accodato prima della cudaMemcpy sincrona), non solo il download. */
        double t0=dsac_now();
        for(int s=0;s<S;s++)
            if(!cuda_ok(cudaMemcpy(dsa->iscore_host+(size_t)s*T,sc_d+(size_t)s*T,
                (size_t)(pos_base+s+1)*4,cudaMemcpyDeviceToHost),"chain score dl")) return 0;
        double t1=dsac_now();
        if(!dsa->topk_fn(dsa->topk_user,dsa->iscore_host,S,pos_base,dsa->topk,
                         dsa->sel,dsa->nsel)){
            fprintf(stderr,"[CUDA] attn_chain: topk_fn\n"); return 0; }
        coli_dsac_t_sync+=t1-t0; coli_dsac_t_topk+=dsac_now()-t1;
    }
    if(selm){
        for(int s=0;s<S;s++) if(dsa->nsel[s]<1||dsa->nsel[s]>dsa->topk){
            fprintf(stderr,"[CUDA] attn_chain: nsel[%d]=%d\n",s,dsa->nsel[s]); return 0; }
        if(!cuda_ok(cudaMemcpyAsync(dsel_d,dsa->sel,(size_t)S*dsa->topk*sizeof(int),
            cudaMemcpyHostToDevice,0),"chain sel ul")) return 0;
    }
    /* loop 2: absorb + o_proj per row (over the selection when active) */
    for(int s=0;s<S;s++){
        int pos=pos_base+s;
        float *qQs=qQ+(size_t)s*H*qh;
        if(selm){
            int ns=dsa->nsel[s];
            size_t smem_sel=(size_t)(2*kv_lora+ns+256)*sizeof(float);
            if(f16) attention_absorb_sel_kernel<<<H,256,smem_sel>>>(cx,qQs,
                hLc,hRc,dsel_d+(size_t)s*dsa->topk,ns,
                kvb->weights,kvb->scales,kvb->fmt,
                H,qk_nope,qk_rope,vh,kv_lora,attn_scale);
            else attention_absorb_sel_kernel<<<H,256,smem_sel>>>(cx,qQs,
                d_Lc,d_Rc,dsel_d+(size_t)s*dsa->topk,ns,
                kvb->weights,kvb->scales,kvb->fmt,
                H,qk_nope,qk_rope,vh,kv_lora,attn_scale);
        }else if(f16)
            absorption_kernel<<<H,256,abs_smem>>>(cx,qQs,
                (const uint8_t*)kvb->weights,kvb->scales,kv_lora,
                hLc,hRc,H,qk_nope,qk_rope,vh,kv_start,pos,attn_scale);
        else
            absorption_kernel<<<H,256,abs_smem>>>(cx,qQs,
                (const uint8_t*)kvb->weights,kvb->scales,kv_lora,
                d_Lc,d_Rc,H,qk_nope,qk_rope,vh,kv_start,pos,attn_scale);
        gemv_q4<<<((unsigned)D+7)/8,256,smem_o>>>(aout+(size_t)s*D,cx,
            (const uint8_t*)o_proj->weights,o_proj->scales,H*vh,D);
    }
    add_vec_kernel<<<((unsigned)(S*D)+255)/256,256>>>(x_dev,x_dev,aout,S*D);
    for(int s=0;s<S;s++)
        rmsnorm_kernel<<<1,256,256*sizeof(float)>>>(nrm_dev+(size_t)s*D,x_dev+(size_t)s*D,w_post,D,eps);
    /* Optional fused shared expert: launched before the chain sync so it runs
     * during the downloads and the host-side gap, instead of after them. */
    if(shg&&shu&&shd&&sI>0&&shg->fmt==2&&shu->fmt==2&&shd->fmt==2){
        float *sg=coli_cuda_pipe_scratch(device,29,(size_t)sI*4);
        float *su=coli_cuda_pipe_scratch(device,30,(size_t)sI*4);
        size_t smem_sh=(size_t)D*sizeof(__half);
        size_t smem_dn=(size_t)sI*sizeof(__half);
        if(sg&&su){
            for(int s=0;s<S;s++){
                gemv_q4<<<((unsigned)sI+7)/8,256,smem_sh>>>(sg,nrm_dev+(size_t)s*D,
                    (const uint8_t*)shg->weights,shg->scales,D,sI);
                gemv_q4<<<((unsigned)sI+7)/8,256,smem_sh>>>(su,nrm_dev+(size_t)s*D,
                    (const uint8_t*)shu->weights,shu->scales,D,sI);
                silu_mul<<<((unsigned)sI+255)/256,256>>>(sg,su,(size_t)sI);
                gemv_q4<<<((unsigned)D+7)/8,256,smem_dn>>>(aout+(size_t)s*D,sg,
                    (const uint8_t*)shd->weights,shd->scales,sI,D);
            }
            add_vec_kernel<<<((unsigned)(S*D)+255)/256,256>>>(x_dev,x_dev,aout,S*D);
        }
    }
    float *sc=NULL;
    if(d_router&&scores_host&&E>0){
        sc=coli_cuda_pipe_scratch(device,28,(size_t)S*E*4);
        if(sc) for(int s=0;s<S;s++)
            gemv_f32_kernel<<<(unsigned)E,256>>>(sc+(size_t)s*E,nrm_dev+(size_t)s*D,d_router,D,E);
    }
    /* single sync point: the canonical-host downloads */
    if(xn_host&&cudaMemcpy(xn_host,xn,(size_t)S*D*4,cudaMemcpyDeviceToHost)!=cudaSuccess) return 0;
    if(cudaMemcpy(nrm_host,nrm_dev,(size_t)S*D*4,cudaMemcpyDeviceToHost)!=cudaSuccess) return 0;
    if(sc&&cudaMemcpy(scores_host,sc,(size_t)S*E*4,cudaMemcpyDeviceToHost)!=cudaSuccess) sc=NULL;
    if(d_router&&scores_host&&E>0&&!sc) scores_host[0]=NAN;   /* signal: no scores, caller recomputes */
    /* con f16 l'host scarica lo STAGING fp32 (esatto), non l'ombra lossy */
    if(cudaMemcpy(kv_host_L,f16?lrow:d_Lc+(size_t)pos_base*kv_lora,
                  (size_t)S*kv_lora*4,cudaMemcpyDeviceToHost)!=cudaSuccess) return 0;
    if(cudaMemcpy(kv_host_R,f16?rrow:d_Rc+(size_t)pos_base*qk_rope,
                  (size_t)S*qk_rope*4,cudaMemcpyDeviceToHost)!=cudaSuccess) return 0;
    if((dsa_full||dsa_reuse)&&dsa->ic_host&&
       cudaMemcpy(dsa->ic_host,f16?irow:dsa->d_Ic+(size_t)pos_base*dsa->hd,
                  (size_t)S*dsa->hd*4,cudaMemcpyDeviceToHost)!=cudaSuccess) return 0;
    { cudaError_t ce=cudaGetLastError();
      if(ce!=cudaSuccess){ fprintf(stderr,"[CUDA] attn_chain: %s\n",cudaGetErrorString(ce));
                           return 0; }
    }
    return 1;
}

extern "C" int coli_cuda_expert_group_collect(int device,int home_device,float *x_dev,int D){
    DeviceContext *rc=find_ctx(device);
    if(!rc||!rc->accum_pending) return 1;
    DeviceContext *hc=find_ctx(home_device);
    if(!hc||!select_ctx(hc)){ rc->accum_pending=0; return 0; }
    if(!cuda_ok(cudaStreamWaitEvent(0,rc->group_ev,0),"group collect wait")){ rc->accum_pending=0; return 0; }
    const float *src=rc->accum;
    if(device!=home_device){
        float *tmp=coli_cuda_pipe_scratch(home_device,31,(size_t)D*sizeof(float));
        if(!tmp||!cuda_ok(cudaMemcpyPeerAsync(tmp,home_device,rc->accum,device,
                (size_t)D*sizeof(float),0),"group collect peer")){ rc->accum_pending=0; return 0; }
        src=tmp;
        select_ctx(hc);
    }
    add_vec_kernel<<<((unsigned)D+255)/256,256>>>(x_dev,x_dev,src,D);
    rc->accum_pending=0;
    return cuda_ok(cudaGetLastError(),"group collect add");
}

extern "C" void *coli_cuda_pipe_alloc(int device,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return NULL;
    void *p=NULL;
    if(!cuda_ok(cudaMalloc(&p,bytes),"pipe alloc")) return NULL;
    return p;
}
extern "C" void coli_cuda_pipe_free(int device,void *p){
    DeviceContext *ctx=find_ctx(device); if(!p||!select_ctx(ctx)) return;
    cudaFree(p);
}
extern "C" int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice),"pipe upload");
}
extern "C" int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost),"pipe download");
}
extern "C" int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                                      const float *w_dev,int S,int D,float eps){
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,D,D);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm");
}
extern "C" int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                                        const float *w_dev,int S,int D,float eps,
                                        int xstride,int ystride){
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||xstride<D||ystride<D||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,xstride,ystride);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm strided");
}
extern "C" int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,
                                   int rows,int stride,int offset,int R,int heads,
                                   float theta){
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,pos_dev,0,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope");
}
extern "C" int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                                        int stride,int offset,int R,int heads,float theta){
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,NULL,pos_base,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope base");
}
extern "C" int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                                     int spitch,int width,int height){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy2D(dst,(size_t)dpitch*4,src,(size_t)spitch*4,
        (size_t)width*4,height,cudaMemcpyDeviceToDevice),"pipe copy2d");
}
/* ---- scritture nelle ombre KV/Ic nel formato COLI_KV_F16 -------------------
 * dst è il puntatore BASE dell'ombra (opaco per l'host); elem_off/elems sono
 * in ELEMENTI.  Con f16 l'upload host passa da uno staging fp32 e converte;
 * senza, degrada alle memcpy esistenti.  copy2d_kv converte righe strided
 * device->ombra (il writer di prefill). */
extern "C" int coli_cuda_pipe_upload_kv(int device,void *dst,const float *src,
                                        size_t elems,size_t elem_off){
    DeviceContext *ctx=find_ctx(device); if(!elems||!select_ctx(ctx)) return 0;
    if(!kv_f16_mode())
        return cuda_ok(cudaMemcpy((float*)dst+elem_off,src,elems*4,
            cudaMemcpyHostToDevice),"kv upload");
    const size_t CH=4u<<20;                       /* 4M elementi = 16 MB staging */
    size_t done=0;
    while(done<elems){
        size_t n=elems-done; if(n>CH) n=CH;
        if(!reserve(&ctx->cvt,&ctx->cvt_cap,n*4)) return 0;
        if(!cuda_ok(cudaMemcpy(ctx->cvt,src+done,n*4,cudaMemcpyHostToDevice),
            "kv upload stage")) return 0;
        cvt_f32_f16<<<(unsigned)((n+255)/256),256>>>((__half*)dst+elem_off+done,
            ctx->cvt,n);
        if(!cuda_ok(cudaGetLastError(),"kv upload cvt")) return 0;
        done+=n;
    }
    return 1;
}
extern "C" int coli_cuda_pipe_copy2d_kv(int device,void *dst,int dpitch,
        const float *src,int spitch,int width,int height,size_t elem_off){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(!kv_f16_mode())
        return cuda_ok(cudaMemcpy2D((float*)dst+elem_off,(size_t)dpitch*4,
            src,(size_t)spitch*4,(size_t)width*4,height,
            cudaMemcpyDeviceToDevice),"kv copy2d");
    cvt2d_f32_f16<<<dim3((unsigned)((width+255)/256),(unsigned)height),256>>>(
        (__half*)dst+elem_off,dpitch,src,spitch,width);
    return cuda_ok(cudaGetLastError(),"kv copy2d cvt");
}
/* attention batch + fused o_proj with DEVICE-resident q/latent/rope: the whole
 * upstream projection chain stayed on this device, so nothing is uploaded here.
 * Only the final [S,O] projection is downloaded to host. */
/* lancio dell'absorb batch su ombre device: sceglie l'istanza (fp32/__half)
 * secondo COLI_KV_F16 — latent_dev/rope_dev sono SEMPRE puntatori ombra qui */
static int absorb_batch_shadow_launch(DeviceContext *dc,float *ctx,const float *q_dev,
        const float *latent_dev,const float *rope_dev,const ColiCudaTensor *w,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    size_t shared;
    if(kv_f16_mode()){
        if(!absorb_smem_ok(dc,K,T,&shared,(const void*)attention_absorb_batch_kernel<__half>,
                           &dc->absorb_attr_set_h))return 0;
        attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(ctx,q_dev,
            (const __half*)latent_dev,(const __half*)rope_dev,
            w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale);
    }else{
        if(!absorb_smem_ok(dc,K,T,&shared,(const void*)attention_absorb_batch_kernel<float>,
                           &dc->absorb_attr_set))return 0;
        attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(ctx,q_dev,
            latent_dev,rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale);
    }
    return 1;
}
extern "C" int coli_cuda_attention_project_batch_dev(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!proj||!out||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!absorb_batch_shadow_launch(dc,dc->ac,q_dev,latent_dev,rope_dev,
        w,S,H,Q,R,V,K,T,scale))return 0;
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch"))return 0;
    size_t ob=(size_t)S*proj->O*sizeof(float);
    if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I));
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch"))return 0;
    if(!cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"pipe attention download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,
                                       size_t n){
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(gate_dev,up_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe silu mul");
}
extern "C" int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    pipe_add_n<<<(unsigned)((n+255)/256),256>>>(x_dev,t_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe add");
}
extern "C" int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                                       const int *rows_dev,int nrows,int D){
    DeviceContext *ctx=find_ctx(device); if(nrows<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rows_add<<<nrows,256>>>(x_dev,partial_dev,rows_dev,D);
    return cuda_ok(cudaGetLastError(),"pipe rows add");
}
/* GEMM with device-resident activations: same quant_matmul kernel as
 * coli_cuda_matmul, zero host transfers.  For prefill-sized batches the
 * naive block-per-output kernel is bandwidth-tragic: route int4 tensors
 * with S>=16 through the w4a16 wmma tiles instead (same math, fp16
 * activation/weight staging with fp32 accumulate — the same tradeoff as
 * COLI_CUDA_TC_W4A16 on the expert path).  Decode (S<16) stays on the
 * exact fp32 kernel.  COLI_PIPE_TC=0 opts out. */
extern "C" int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,
                                   int S){
    if(!t||S<1) return 0;
    DeviceContext *ctx=find_ctx(t->device); if(!select_ctx(ctx)) return 0;
    static int tc=-1;
    if(tc<0){ const char *e=getenv("COLI_PIPE_TC"); tc=e?atoi(e):1; }
    if(tc&&S>=16&&t->fmt==2&&(t->I&15)==0&&ctx->compute_major>=7){
        dim3 grid((unsigned)((t->O+63)/64),(unsigned)((S+15)/16));
        w4a16_matmul<<<grid,128>>>(y_dev,x_dev,(const uint8_t*)t->weights,t->scales,
            S,t->I,t->O);
        return cuda_ok(cudaGetLastError(),"pipe gemm tc");
    }
    dim3 grid((unsigned)t->O,(unsigned)S);
    quant_matmul<<<grid,256>>>(y_dev,x_dev,t->weights,t->scales,t->fmt,S,t->I,t->O,
        row_bytes(t->fmt,t->I));
    return cuda_ok(cudaGetLastError(),"pipe gemm");
}
/* copia diretta scheda->scheda (P2P se disponibile, altrimenti staging driver) */
extern "C" int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                                        const float *src,size_t bytes){
    if(!dst||!src) return 0;
    if(dst_dev==src_dev){ DeviceContext *c=find_ctx(dst_dev); if(!select_ctx(c)) return 0;
        return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToDevice),"pipe intra copy"); }
    return cuda_ok(cudaMemcpyPeer(dst,dst_dev,src,src_dev,bytes),"pipe peer copy");
}
/* come attention_project_batch_dev ma l'uscita di o_proj RESTA sul device (out_dev). */
extern "C" int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!proj||!out_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!absorb_batch_shadow_launch(dc,dc->ac,q_dev,latent_dev,rope_dev,
        w,S,H,Q,R,V,K,T,scale))return 0;
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch (dev out)"))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(out_dev,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I));
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch (dev out)"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync (dev out)");
}
/* absorb batch con TUTTO su device (q/latent/rope gia' residenti sulla scheda
 * dello shard, ctx resta sul device): il cuore della attention head-shardata
 * dentro il pipeline. Nessun trasferimento host. */
extern "C" int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *w,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!ctx_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    if(!absorb_batch_shadow_launch(dc,ctx_dev,q_dev,latent_dev,rope_dev,
        w,S,H,Q,R,V,K,T,scale))return 0;
    if(!cuda_ok(cudaGetLastError(),"pipe shard attention launch"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe shard attention sync");
}
/* DSA: absorb su una LISTA di posizioni selezionate (top-k dell'indexer).
 * sel[] contiene posizioni ASSOLUTE nella shadow KV del device — latent/rope
 * sono i puntatori base cuda_Lc/cuda_Rc, nessuna finestra st0. Stessa
 * matematica del percorso CPU (score sui selezionati, softmax, gather). */
template<typename KT>
__device__ static void absorb_sel_body(float *ctx,const float *q,
        const KT *latent,const KT *rope,const int *sel,int ns,
        const void *weights,const float *wscale,int fmt,
        int H,int Q,int R,int V,int K,float scale){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+ns;
    const float *qs=q+(size_t)h*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          (fmt?wscale[rbase+d]:1.f);qa[k]=a;}
    __syncthreads();
    for(int j=tid;j<ns;j+=blockDim.x){int t=sel[j];float a=0;
        const KT *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        for(int k=0;k<K;k++)a+=qa[k]*shf(lt+k);
        for(int d=0;d<R;d++)a+=qs[Q+d]*shf(rt+d);scores[j]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int j=tid;j<ns;j+=blockDim.x)local=fmaxf(local,scores[j]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];__syncthreads();          /* barriera anti-race (vedi absorb batch) */
    local=0;for(int j=tid;j<ns;j+=blockDim.x){float e=expf(scores[j]-mx);scores[j]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int j=tid;j<ns;j+=blockDim.x)scores[j]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int j=0;j<ns;j++)
        a+=scores[j]*shf(latent+(size_t)sel[j]*K+k);cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k);
        ctx[(size_t)h*V+v]=a*(fmt?wscale[row]:1.f);}
}
template<typename KT>
__global__ static void attention_absorb_sel_kernel(float *ctx,const float *q,
        const KT *latent,const KT *rope,const int *sel,int ns,
        const void *weights,const float *wscale,int fmt,
        int H,int Q,int R,int V,int K,float scale){
    absorb_sel_body(ctx,q,latent,rope,sel,ns,weights,wscale,fmt,H,Q,R,V,K,scale);
}
/* Batch di righe query, una per blockIdx.y: riga y usa la SUA lista sel
 * (ns fisso = topk, invariante del top-k di prefill).  q [rows,H*(Q+R)],
 * ctx [rows,H*V], sel [rows,topk] posizioni assolute. */
template<typename KT>
__global__ static void attention_absorb_sel_rows_kernel(float *ctx,const float *q,
        const KT *latent,const KT *rope,const int *sel,int topk,
        const void *weights,const float *wscale,int fmt,
        int H,int Q,int R,int V,int K,float scale){
    int y=blockIdx.y;
    absorb_sel_body(ctx+(size_t)y*H*V,q+(size_t)y*H*(Q+R),latent,rope,
                    sel+(size_t)y*topk,topk,weights,wscale,fmt,H,Q,R,V,K,scale);
}
extern "C" int coli_cuda_attention_project_sel(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const float *latent_dev,const float *rope_dev,
        const int *sel,int ns,int H,int Q,int R,int V,int K,float scale){
    if(!w||!proj||!out||!q||!latent_dev||!rope_dev||!sel||ns<1||ns>8192||H<1||Q<1||R<1||
       V<1||K<1||K>512||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),cb=(size_t)H*V*sizeof(float);
    size_t ob=(size_t)proj->O*sizeof(float),sb=(size_t)ns*sizeof(int);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->ac,&dc->ac_cap,cb)||
       !reserve(&dc->y,&dc->y_cap,ob)||
       !reserve_bytes((void**)&dc->qx,&dc->qx_cap,sb))return 0;  /* qx libero in decode (solo W4A4 lo usa) */
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"sel q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->qx,sel,sb,cudaMemcpyHostToDevice,dc->stream),"sel list upload"))return 0;
    size_t shared=(size_t)(2*K+ns+256)*sizeof(float);
    if(kv_f16_mode())
        attention_absorb_sel_kernel<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,
            (const __half*)latent_dev,(const __half*)rope_dev,
            (const int*)dc->qx,ns,w->weights,w->scales,w->fmt,H,Q,R,V,K,scale);
    else
        attention_absorb_sel_kernel<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,latent_dev,
            rope_dev,(const int*)dc->qx,ns,w->weights,w->scales,w->fmt,H,Q,R,V,K,scale);
    if(!cuda_ok(cudaGetLastError(),"sel absorb launch"))return 0;
    quant_matmul<<<dim3(proj->O,1),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,1,proj->I,proj->O,row_bytes(proj->fmt,proj->I));
    if(!cuda_ok(cudaGetLastError(),"sel o_proj launch")||
       !cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"sel out download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"sel absorb sync"))return 0;
    return 1;
}
/* absorb per il DECODE con KV gia' residente: carica solo q (poche KB),
 * latent/rope arrivano dall'ombra device. ctx torna a host (S piccolo). */
extern "C" int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale){
    if(!w||!ctx||!q||!latent_dev||!rope_dev||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"kvdev q upload"))return 0;
    if(!absorb_batch_shadow_launch(dc,dc->ac,dc->aq,latent_dev,rope_dev,
        w,1,H,Q,R,V,K,T,scale))return 0;
    if(!cuda_ok(cudaGetLastError(),"kvdev absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"kvdev ctx download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"kvdev absorb sync"))return 0;
    return 1;
}
/* ---- Prefill attention as five GEMMs (per head) --------------------------
 * Replaces attention_absorb_batch_kernel at prefill sizes, where the naive
 * per-element contraction is ~60x off the FLOP floor.  Same math on tensor
 * cores, one head at a time (scores[S,T] for all 64 heads would be GBs):
 *   1. qabs[S,K]   = q_nope[S,Q] @ (ws*Wk)[Q,K]           int4 weights, NN
 *   2. scores[S,T] = qabs @ Lc^T + q_rope @ Rc^T           fp16 TC, NT
 *   3. causal online-softmax rows (scale applied here; tail zeroed so the
 *      step-4 reduction can run over the full T)
 *   4. ctxL[S,K]   = P[S,T] @ Lc[T,K]                      fp16 TC, NN
 *   5. ctx[:,hV:]  = ctxL @ Wv^T ;  out = ctx @ Wo^T       int4 weights, NT
 * Weight nibbles are SIGNED (upload converts); decode matches w4a16_matmul. */
template<typename AT,typename BT>
__device__ static void gemm_f16_tc_body(float *C,const AT *A,const BT *B,
        int M,int N,int K,int lda,int ldb,int ldc,int transB,int beta){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){
            int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?shh(A+(size_t)gm*lda+gk):__float2half(0.f);
        }
        if(transB){                                    /* B[N,K] righe: C=A@B^T */
            for(int z=lane;z<256;z+=32){int n=z/16,gk=k0+(z%16),gn=n0+n;
                bh[warp][z]=(gn<N&&gk<K)?shh(B+(size_t)gn*ldb+gk):__float2half(0.f);}
        }else{                                         /* B[K,N] righe: C=A@B */
            for(int z=lane;z<256;z+=32){int k=z/16,gn=n0+(z%16),gk=k0+k;
                bh[warp][z]=(gn<N&&gk<K)?shh(B+(size_t)gk*ldb+gn):__float2half(0.f);}
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::load_matrix_sync(af,ah,16);
        if(transB){
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
            wmma::load_matrix_sync(bf,bh[warp],16);wmma::mma_sync(acc,af,bf,acc);
        }else{
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bf;
            wmma::load_matrix_sync(bf,bh[warp],16);wmma::mma_sync(acc,af,bf,acc);
        }
        __syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N){size_t i=(size_t)(m0+m)*ldc+n0+n;C[i]=beta?C[i]+out[warp][z]:out[warp][z];}}
#endif
}
template<typename BT>
__global__ static void gemm_f16_tc(float *C,const float *A,const BT *B,
        int M,int N,int K,int lda,int ldb,int ldc,int transB,int beta){
    gemm_f16_tc_body(C,A,B,M,N,K,lda,ldb,ldc,transB,beta);
}
/* batch su blockIdx.z: problema z usa A/B/C spostati di z*stride (elementi).
 * Serve alla fase B DSA: una GEMM per riga query (teste = dimensione M). */
template<typename BT>
__global__ static void gemm_f16_tc_zb(float *C,const float *A,const BT *B,
        int M,int N,int K,int lda,int ldb,int ldc,int transB,int beta,
        size_t sA,size_t sB,size_t sC){
    size_t z=blockIdx.z;
    gemm_f16_tc_body(C+z*sC,A+z*sA,B+z*sB,M,N,K,lda,ldb,ldc,transB,beta);
}
/* y[M,N] = x[M,K] @ dec(w)[K,N] con scala per riga K (l'assorbimento q@Wk:
 * la riga del tensore e' la dimensione di riduzione, scala inclusa in B). */
__device__ static void w4a16_nn_scaled_body(float *y,const float *x,const uint8_t *w,
        const float *ws,int M,int N,int K,int lda,size_t wrb,int ldy){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*lda+gk]):__float2half(0.f);}
        for(int z=lane;z<256;z+=32){int k=z/16,n=z%16,gk=k0+k,gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gk*wrb+(gn>>1)];int a=(gn&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*ws[gk];}
            bh[warp][z]=__float2half(v);}              /* [Ktile,Ntile] row-major */
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*ldy+n0+n]=out[warp][z];}
#endif
}
__global__ static void w4a16_nn_scaled(float *y,const float *x,const uint8_t *w,
        const float *ws,int M,int N,int K,int lda,size_t wrb,int ldy){
    w4a16_nn_scaled_body(y,x,w,ws,M,N,K,lda,wrb,ldy);
}
/* block-diagonale per testa (blockIdx.z=h): x avanza di xz elementi, i pesi
 * di wrowz righe di tensore (+wrow0), l'uscita di yz elementi.  Fase B DSA:
 * qabs[r,h,:] = q_nope[r,h,:] @ Wk_h con M=righe del chunk. */
__global__ static void w4a16_nn_scaled_bd(float *y,const float *x,const uint8_t *w,
        const float *ws,int M,int N,int K,int lda,size_t wrb,int ldy,
        int xz,int wrowz,int wrow0,int yz){
    size_t z=blockIdx.z,wr=z*wrowz+wrow0;
    w4a16_nn_scaled_body(y+z*yz,x+z*xz,w+wr*wrb,ws+wr,M,N,K,lda,wrb,ldy);
}
/* w4a16_matmul con passo d'uscita ldy: serve per scrivere la slice di testa
 * dentro ctx[S,H*V] (e riusato per o_proj con ldy==N). x [M,K] passo lda. */
__device__ static void w4a16_nt_ld_body(float *y,const float *x,const uint8_t *w,
        const float *scale,int M,int K,int N,int lda,int ldy){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    size_t rb=(size_t)(K+1)/2;
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){
            int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*lda+gk]):__float2half(0.f);
        }
        for(int z=lane;z<256;z+=32){
            int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}
            bh[warp][z]=__float2half(v);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*ldy+n0+n]=out[warp][z];}
#endif
}
__global__ static void w4a16_nt_ld(float *y,const float *x,const uint8_t *w,
        const float *scale,int M,int K,int N,int ldy){
    w4a16_nt_ld_body(y,x,w,scale,M,K,N,K,ldy);
}
/* block-diagonale per testa (blockIdx.z=h), vedi w4a16_nn_scaled_bd.  Fase B
 * DSA: ctx[r,h*V..] = ctxL[r,h,:] @ Wv_h^T (righe peso = uscita, NT). */
__global__ static void w4a16_nt_bd(float *y,const float *x,const uint8_t *w,
        const float *scale,int M,int K,int N,int lda,int ldy,
        int xz,int wrowz,int wrow0,int yz){
    size_t z=blockIdx.z,wr=z*wrowz+wrow0,rb=(size_t)(K+1)/2;
    w4a16_nt_ld_body(y+z*yz,x+z*xz,w+wr*rb,scale+wr,M,K,N,lda,ldy);
}
/* softmax causale in-place su scores[S,T]: riga s vede nt=T-S+s+1 chiavi.
 * La coda [nt,T) viene azzerata: il GEMM del passo 4 riduce su tutto T. */
__global__ static void causal_softmax_rows(float *scores,int S,int T,float scale){
    int s=blockIdx.x,tid=threadIdx.x; if(s>=S)return;
    int nt=T-S+s+1; float *row=scores+(size_t)s*T;
    __shared__ float red[256];
    float local=-3.402823466e+38F;
    for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,row[t]*scale);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];__syncthreads();
    local=0;
    for(int t=tid;t<nt;t+=blockDim.x){float e=expf(row[t]*scale-mx);row[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];
    for(int t=tid;t<nt;t+=blockDim.x)row[t]*=inv;
    for(int t=nt+tid;t<T;t+=blockDim.x)row[t]=0.f;
}
/* Fase B DSA in GEMM: raccoglie le righe Lc/Rc selezionate dalla riga query
 * z in buffer contigui — pagata UNA volta per riga invece dei load sparsi
 * ripetuti per ognuna delle 64 teste. */
template<typename KT>
__global__ static void dsa_gather_sel(KT *LcSel,KT *RcSel,
        const KT *latent,const KT *rope,const int *sel,int topk,int K,int R,
        int T){
    size_t z=blockIdx.y,j=blockIdx.x;
    int t=sel[z*topk+j];
    if(t<0)t=0; if(t>=T)t=T-1;   /* un indice corrotto non deve toccare VA selvaggia */
    const KT *ls=latent+(size_t)t*K,*rs=rope+(size_t)t*R;
    KT *ld=LcSel+(z*topk+j)*K,*rd=RcSel+(z*topk+j)*R;
    for(int k=threadIdx.x;k<K;k+=blockDim.x)ld[k]=ls[k];
    for(int r=threadIdx.x;r<R;r+=blockDim.x)rd[r]=rs[r];
}
/* softmax in-place su righe piene (nessuna causalita': ogni riga di fase B
 * vede esattamente topk chiavi selezionate). */
__global__ static void softmax_rows_flat(float *rows,int width,float scale){
    int tid=threadIdx.x;float *row=rows+(size_t)blockIdx.x*width;
    __shared__ float red[256];
    float local=-3.402823466e+38F;
    for(int t=tid;t<width;t+=blockDim.x)local=fmaxf(local,row[t]*scale);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];__syncthreads();
    local=0;
    for(int t=tid;t<width;t+=blockDim.x){float e=expf(row[t]*scale-mx);row[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];
    for(int t=tid;t<width;t+=blockDim.x)row[t]*=inv;
}
/* DSA prefill: k_idx per S righe nuove nell'ombra Ic device + punteggi della
 * selezione per le righe di fase B (pos+1 > topk), scaricati sull'host per il
 * top-k esatto.  xn_dev = righe in_ln-normate [S,D], qres_dev = residuo q
 * normato [S,q_lora].  ic_host riceve le S righe k_idx (host canonico),
 * iscore_host le righe punteggio [S-sB0, pos_base+S].  Kernel sullo stream
 * legacy (ordinato con i pipe_*); i download sincronizzano. */
extern "C" int coli_cuda_prefill_dsa_select(int device,ColiCudaDsaChain *dsa,
        const float *xn_dev,const float *qres_dev,
        int S,int pos_base,int sB0,int D,int q_lora,int qk_rope,float theta){
    /* COLI_DSA_DEVTOPK (default 1): il top-k della selezione gira sul device
     * (dsa_topk_rows, semantica bit-identica all'host) a CHUNK di righe —
     * spariscono il download S_b*T dei punteggi, il buffer host gigante e lo
     * scratch device S_b*T che oltre ~64k non allocherebbe.  Scende solo la
     * selezione (S_b*topk int, indipendente da T).  Con =0 resta il percorso
     * host esatto (iscore_host + top-k del chiamante). */
    static int devtopk=-1;
    if(devtopk<0){ const char *e=getenv("COLI_DSA_DEVTOPK"); devtopk=e?(atoi(e)!=0):1; }
    if(!dsa||!dsa->ix_wq||!dsa->ix_wk||!dsa->ix_wp||!dsa->knw_dev||!dsa->knb_dev||
       !dsa->d_Ic||!dsa->ic_host||!xn_dev||!qres_dev||
       S<1||pos_base<0||sB0<0||sB0>S||dsa->nh<1||dsa->hd<1) return 0;
    if(devtopk ? !(dsa->sel&&dsa->nsel&&dsa->topk>0) : !dsa->iscore_host) return 0;
    int nh=dsa->nh,hd=dsa->hd,T=pos_base+S,S_b=S-sB0;
    if((dsa->ix_wq->fmt!=1&&dsa->ix_wq->fmt!=2)||
       (dsa->ix_wk->fmt!=1&&dsa->ix_wk->fmt!=2)||
       (dsa->ix_wp->fmt!=1&&dsa->ix_wp->fmt!=2)) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(!ctx||!select_ctx(ctx)) return 0;
    /* COLI_KV_F16: k_idx in staging fp32 (slot 39), ombra riceve la conversione
     * PRIMA dei punteggi (che leggono d_Ic fino a T incluse le righe nuove);
     * ic_host scarica lo staging esatto. */
    int f16=kv_f16_mode();
    float *icrows;
    if(f16){
        icrows=coli_cuda_pipe_scratch(device,39,(size_t)S*hd*4);
        if(!icrows){ fprintf(stderr,"[CUDA] prefill dsa: ic stage\n"); return 0; }
    }else icrows=dsa->d_Ic+(size_t)pos_base*hd;
    quant_matmul<<<dim3(hd,S),256>>>(icrows,xn_dev,dsa->ix_wk->weights,
        dsa->ix_wk->scales,dsa->ix_wk->fmt,S,D,hd,row_bytes(dsa->ix_wk->fmt,D));
    layernorm_kernel<<<S,128>>>(icrows,dsa->knw_dev,dsa->knb_dev,hd,1e-6f);
    pipe_rope_rows<<<S,128>>>(icrows,NULL,pos_base,hd,0,qk_rope,1,theta);
    if(f16) cvt_f32_f16<<<(unsigned)(((size_t)S*hd+255)/256),256>>>(
        (__half*)dsa->d_Ic+(size_t)pos_base*hd,icrows,(size_t)S*hd);
    if(S_b>0){
        int RB=devtopk?(S_b<512?S_b:512):S_b;      /* chunk di righe in modalita' device */
        float *isc =coli_cuda_pipe_scratch(device,18,(size_t)RB*T*4);
        float *qi_d=coli_cuda_pipe_scratch(device,19,(size_t)S_b*nh*hd*4);
        float *w32d=coli_cuda_pipe_scratch(device,20,(size_t)S_b*nh*4);
        int *dsel=devtopk?(int*)coli_cuda_pipe_scratch(device,40,
            (size_t)RB*dsa->topk*sizeof(int)):NULL;
        if(!isc||!qi_d||!w32d||(devtopk&&!dsel)){
            fprintf(stderr,"[CUDA] prefill dsa: scratch\n"); return 0; }
        quant_matmul<<<dim3(nh*hd,S_b),256>>>(qi_d,qres_dev+(size_t)sB0*q_lora,
            dsa->ix_wq->weights,dsa->ix_wq->scales,dsa->ix_wq->fmt,S_b,q_lora,nh*hd,
            row_bytes(dsa->ix_wq->fmt,q_lora));
        pipe_rope_rows<<<S_b*nh,128>>>(qi_d,NULL,pos_base+sB0,hd,0,qk_rope,nh,theta);
        quant_matmul<<<dim3(nh,S_b),256>>>(w32d,xn_dev+(size_t)sB0*D,
            dsa->ix_wp->weights,dsa->ix_wp->scales,dsa->ix_wp->fmt,S_b,D,nh,
            row_bytes(dsa->ix_wp->fmt,D));
        size_t smem_sc=(size_t)(nh*hd+nh)*sizeof(float);
        for(int c0=0;c0<S_b;c0+=RB){
            int rn=S_b-c0; if(rn>RB)rn=RB;
            int nk0=pos_base+sB0+c0+1;
            if(f16) dsa_score_kernel<<<dim3(((unsigned)T+127)/128,(unsigned)rn),128,smem_sc>>>(
                isc,qi_d+(size_t)c0*nh*hd,w32d+(size_t)c0*nh,
                (const __half*)dsa->d_Ic,nk0,nh,hd,
                1.f/sqrtf((float)nh),1.f/sqrtf((float)hd),T);
            else dsa_score_kernel<<<dim3(((unsigned)T+127)/128,(unsigned)rn),128,smem_sc>>>(
                isc,qi_d+(size_t)c0*nh*hd,w32d+(size_t)c0*nh,
                dsa->d_Ic,nk0,nh,hd,
                1.f/sqrtf((float)nh),1.f/sqrtf((float)hd),T);
            if(devtopk){
                dsa_topk_rows<<<(unsigned)rn,256>>>(dsel,isc,T,nk0,dsa->topk);
                if(!cuda_ok(cudaGetLastError(),"prefill dsa topk launch")) return 0;
                if(!cuda_ok(cudaMemcpy(dsa->sel+(size_t)(sB0+c0)*dsa->topk,dsel,
                    (size_t)rn*dsa->topk*sizeof(int),cudaMemcpyDeviceToHost),
                    "prefill dsa sel dl")) return 0;
            }else{
                if(!cuda_ok(cudaGetLastError(),"prefill dsa launch")) return 0;
                if(!cuda_ok(cudaMemcpy(dsa->iscore_host+(size_t)c0*T,isc,
                    (size_t)rn*T*4,cudaMemcpyDeviceToHost),"prefill dsa score dl")) return 0;
            }
        }
        if(devtopk) for(int r=0;r<S_b;r++){
            int nk=pos_base+sB0+r+1;
            dsa->nsel[sB0+r]=nk<dsa->topk?nk:dsa->topk;
        }
    }
    if(!cuda_ok(cudaGetLastError(),"prefill dsa kidx launch")) return 0;
    return cuda_ok(cudaMemcpy(dsa->ic_host,icrows,(size_t)S*hd*4,
        cudaMemcpyDeviceToHost),"prefill dsa kidx dl");
}
extern "C" int coli_cuda_prefill_attn_gemm(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale,
        const int *sel_host,int sB0,int sel_topk){
    if(!w||!proj||!out_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>131072||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V||w->fmt!=2||proj->fmt!=2||
       (Q&15)||(K&15)||(R&15)||(V&15)||(proj->I&15))return 0;
    /* fase B (selezione DSA): righe [sB0,S) assorbono sulla PROPRIA lista sel
     * (posizioni assolute, kv_start==0); fase A resta sul percorso GEMM causale */
    int S_b=0;
    if(sel_host){
        if(sB0<0||sB0>=S||sel_topk<1||sel_topk>T) return 0;
        S_b=S-sB0;
    } else sB0=S;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    if(dc->compute_major<7)return 0;                  /* wmma fp16 */
    /* COLI_PREFILL_GEMM=2: lancia sullo stream legacy (debug ordering) */
    const char *pgenv=getenv("COLI_PREFILL_GEMM");
    cudaStream_t st=(pgenv&&atoi(pgenv)==2)?0:dc->stream;
    int S_a=sB0, T_a=T-S+sB0;
    if(!reserve(&dc->ac,&dc->ac_cap,(size_t)S*H*V*sizeof(float)))return 0;
    if(S_a>0){
        if(!reserve(&dc->pf_q,&dc->pf_q_cap,(size_t)S_a*K*sizeof(float)))return 0;
        if(!reserve(&dc->pf_c,&dc->pf_c_cap,(size_t)S_a*K*sizeof(float)))return 0;
        if(!reserve(&dc->pf_s,&dc->pf_s_cap,(size_t)S_a*T_a*sizeof(float)))return 0;
    }
    int *dsel=NULL;
    if(S_b>0){
        dsel=(int*)coli_cuda_pipe_scratch(w->device,31,(size_t)S_b*sel_topk*sizeof(int));
        if(!dsel) return 0;
        if(!cuda_ok(cudaMemcpyAsync(dsel,sel_host+(size_t)sB0*sel_topk,
            (size_t)S_b*sel_topk*sizeof(int),cudaMemcpyHostToDevice,st),
            "prefill sel upload")) return 0;
    }
    size_t rb=row_bytes(2,K);
    const uint8_t *wb=(const uint8_t*)w->weights;
    const float *wsc=w->scales;
    int f16=kv_f16_mode();
    const __half *hL=(const __half*)latent_dev,*hR=(const __half*)rope_dev;
    if(S_a>0){
        dim3 gq((unsigned)((K+63)/64),(unsigned)((S_a+15)/16));
        dim3 gs((unsigned)((T_a+63)/64),(unsigned)((S_a+15)/16));
        dim3 gv((unsigned)((V+63)/64),(unsigned)((S_a+15)/16));
        for(int h=0;h<H;h++){
            size_t rbase=(size_t)h*(Q+V);
            const float *qh=q_dev+(size_t)h*(Q+R);
            w4a16_nn_scaled<<<gq,128,0,st>>>(dc->pf_q,qh,wb+rbase*rb,
                wsc+rbase,S_a,K,Q,H*(Q+R),rb,K);
            if(f16){
                gemm_f16_tc<<<gs,128,0,st>>>(dc->pf_s,dc->pf_q,hL,
                    S_a,T_a,K,K,K,T_a,1,0);
                gemm_f16_tc<<<gs,128,0,st>>>(dc->pf_s,qh+Q,hR,
                    S_a,T_a,R,H*(Q+R),R,T_a,1,1);
            }else{
                gemm_f16_tc<<<gs,128,0,st>>>(dc->pf_s,dc->pf_q,latent_dev,
                    S_a,T_a,K,K,K,T_a,1,0);
                gemm_f16_tc<<<gs,128,0,st>>>(dc->pf_s,qh+Q,rope_dev,
                    S_a,T_a,R,H*(Q+R),R,T_a,1,1);
            }
            causal_softmax_rows<<<S_a,256,0,st>>>(dc->pf_s,S_a,T_a,scale);
            if(f16) gemm_f16_tc<<<gq,128,0,st>>>(dc->pf_c,dc->pf_s,hL,
                S_a,K,T_a,T_a,K,K,0,0);
            else gemm_f16_tc<<<gq,128,0,st>>>(dc->pf_c,dc->pf_s,latent_dev,
                S_a,K,T_a,T_a,K,K,0,0);
            w4a16_nt_ld<<<gv,128,0,st>>>(dc->ac+(size_t)h*V,dc->pf_c,
                wb+(rbase+Q)*rb,wsc+rbase+Q,S_a,K,V,H*V);
        }
    }
    if(S_b>0){
        /* TC gather (default): raccogli le righe selezionate in buffer contigui
         * e assorbi con GEMM fp16 batched (teste = M).  Il kernel scalare resta
         * come fallback (COLI_DSA_TCGATHER=0 o scratch esaurito) per l'A/B. */
        const char *tg=getenv("COLI_DSA_TCGATHER");
        int tcg=(!tg||atoi(tg))&&dc->compute_major>=7;
        float *LcSel=NULL,*RcSel=NULL,*qabs=NULL,*scb=NULL,*ctxL=NULL;
        int RB=32; if(RB>S_b)RB=S_b;
        size_t esz=f16?sizeof(__half):sizeof(float);   /* LcSel/RcSel nel formato ombra */
        if(tcg){
            LcSel=coli_cuda_pipe_scratch(w->device,32,(size_t)RB*sel_topk*K*esz);
            RcSel=coli_cuda_pipe_scratch(w->device,33,(size_t)RB*sel_topk*R*esz);
            qabs =coli_cuda_pipe_scratch(w->device,34,(size_t)RB*H*K*sizeof(float));
            scb  =coli_cuda_pipe_scratch(w->device,35,(size_t)RB*H*sel_topk*sizeof(float));
            ctxL =coli_cuda_pipe_scratch(w->device,36,(size_t)RB*H*K*sizeof(float));
            if(!LcSel||!RcSel||!qabs||!scb||!ctxL)tcg=0;
        }
        if(tcg)for(int c0=0;c0<S_b;c0+=RB){
            int rn=S_b-c0; if(rn>RB)rn=RB;
            int row0=sB0+c0;
            if(f16) dsa_gather_sel<<<dim3((unsigned)sel_topk,(unsigned)rn),128,0,st>>>(
                (__half*)LcSel,(__half*)RcSel,hL,hR,dsel+(size_t)c0*sel_topk,sel_topk,K,R,T);
            else dsa_gather_sel<<<dim3((unsigned)sel_topk,(unsigned)rn),128,0,st>>>(
                LcSel,RcSel,latent_dev,rope_dev,dsel+(size_t)c0*sel_topk,sel_topk,K,R,T);
            /* qabs[r,h,:] = q_nope[r,h,:] @ (ws*Wk_h) */
            w4a16_nn_scaled_bd<<<dim3((unsigned)((K+63)/64),(unsigned)((rn+15)/16),(unsigned)H),128,0,st>>>(
                qabs,q_dev+(size_t)row0*H*(Q+R),wb,wsc,rn,K,Q,H*(Q+R),rb,H*K,
                Q+R,Q+V,0,K);
            /* scores[r,h,:] = qabs[r,h,:] @ LcSel[r]^T + q_rope[r,h,:] @ RcSel[r]^T */
            if(f16){
                gemm_f16_tc_zb<<<dim3((unsigned)((sel_topk+63)/64),(unsigned)((H+15)/16),(unsigned)rn),128,0,st>>>(
                    scb,qabs,(const __half*)LcSel,H,sel_topk,K,K,K,sel_topk,1,0,
                    (size_t)H*K,(size_t)sel_topk*K,(size_t)H*sel_topk);
                gemm_f16_tc_zb<<<dim3((unsigned)((sel_topk+63)/64),(unsigned)((H+15)/16),(unsigned)rn),128,0,st>>>(
                    scb,q_dev+(size_t)row0*H*(Q+R)+Q,(const __half*)RcSel,H,sel_topk,R,Q+R,R,sel_topk,1,1,
                    (size_t)H*(Q+R),(size_t)sel_topk*R,(size_t)H*sel_topk);
            }else{
                gemm_f16_tc_zb<<<dim3((unsigned)((sel_topk+63)/64),(unsigned)((H+15)/16),(unsigned)rn),128,0,st>>>(
                    scb,qabs,LcSel,H,sel_topk,K,K,K,sel_topk,1,0,
                    (size_t)H*K,(size_t)sel_topk*K,(size_t)H*sel_topk);
                gemm_f16_tc_zb<<<dim3((unsigned)((sel_topk+63)/64),(unsigned)((H+15)/16),(unsigned)rn),128,0,st>>>(
                    scb,q_dev+(size_t)row0*H*(Q+R)+Q,RcSel,H,sel_topk,R,Q+R,R,sel_topk,1,1,
                    (size_t)H*(Q+R),(size_t)sel_topk*R,(size_t)H*sel_topk);
            }
            softmax_rows_flat<<<rn*H,256,0,st>>>(scb,sel_topk,scale);
            /* ctxL[r,h,:] = P[r,h,:] @ LcSel[r] */
            if(f16) gemm_f16_tc_zb<<<dim3((unsigned)((K+63)/64),(unsigned)((H+15)/16),(unsigned)rn),128,0,st>>>(
                ctxL,scb,(const __half*)LcSel,H,K,sel_topk,sel_topk,K,K,0,0,
                (size_t)H*sel_topk,(size_t)sel_topk*K,(size_t)H*K);
            else gemm_f16_tc_zb<<<dim3((unsigned)((K+63)/64),(unsigned)((H+15)/16),(unsigned)rn),128,0,st>>>(
                ctxL,scb,LcSel,H,K,sel_topk,sel_topk,K,K,0,0,
                (size_t)H*sel_topk,(size_t)sel_topk*K,(size_t)H*K);
            /* ctx[r,h*V..] = ctxL[r,h,:] @ (Wv_h)^T */
            w4a16_nt_bd<<<dim3((unsigned)((V+63)/64),(unsigned)((rn+15)/16),(unsigned)H),128,0,st>>>(
                dc->ac+(size_t)row0*H*V,ctxL,wb,wsc,rn,K,V,H*K,H*V,
                K,Q+V,Q,V);
        }
        if(!tcg){
            size_t smem_sel=(size_t)(2*K+sel_topk+256)*sizeof(float);
            if(f16) attention_absorb_sel_rows_kernel<<<dim3((unsigned)H,(unsigned)S_b),256,smem_sel,st>>>(
                dc->ac+(size_t)sB0*H*V,q_dev+(size_t)sB0*H*(Q+R),hL,hR,
                dsel,sel_topk,w->weights,wsc,w->fmt,H,Q,R,V,K,scale);
            else attention_absorb_sel_rows_kernel<<<dim3((unsigned)H,(unsigned)S_b),256,smem_sel,st>>>(
                dc->ac+(size_t)sB0*H*V,q_dev+(size_t)sB0*H*(Q+R),latent_dev,rope_dev,
                dsel,sel_topk,w->weights,wsc,w->fmt,H,Q,R,V,K,scale);
        }
    }
    if(!cuda_ok(cudaGetLastError(),"prefill gemm attention launch"))return 0;
    dim3 go((unsigned)((proj->O+63)/64),(unsigned)((S+15)/16));
    w4a16_nt_ld<<<go,128,0,st>>>(out_dev,dc->ac,
        (const uint8_t*)proj->weights,proj->scales,S,proj->I,proj->O,proj->O);
    if(!cuda_ok(cudaGetLastError(),"prefill gemm o_proj launch"))return 0;
    return cuda_ok(cudaStreamSynchronize(st),"prefill gemm attention sync");
}
extern "C" int coli_cuda_pipe_sync(int device){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaDeviceSynchronize(),"pipe sync");
}
