/* backend_loader.c — Windows runtime loader for coli_cuda.dll.
 *
 * Why this exists: the engine is built with MinGW-w64 (gcc), but CUDA kernels
 * must be compiled with MSVC + nvcc. We cannot link a CUDA .o into a gcc binary
 * reliably across the MSVC/MinGW ABI, and nvcc requires cl.exe as its host
 * compiler. The clean cross-toolchain split is: build the CUDA backend into a
 * standalone coli_cuda.dll with nvcc+MSVC, then load it here at runtime via
 * LoadLibrary/GetProcAddress. The host (glm.exe) never links cudart directly.
 *
 * On Linux this file is not compiled (the Makefile links backend_cuda.o
 * directly). On Windows, when COLI_CUDA is defined, glm.c calls the
 * coli_cuda_* wrappers below, which forward through function pointers resolved
 * from the DLL at first use. If the DLL is absent, every call safely returns
 * the "not initialized" sentinel (0 / no-op) and the engine falls back to CPU.
 *
 * ABI note: ColiCudaTensor* is opaque to the host (it stores the pointer,
 * never dereferences it), so the MSVC-allocated struct is safe to pass across
 * the boundary as an opaque handle. All scalar types (int, size_t, pointers)
 * agree between MSVC and MinGW-w64 on x86-64.
 */
#ifdef _WIN32

#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <windows.h>

#include "backend_cuda.h"

/* Function-pointer typedefs matching each exported symbol. */
typedef int            (*fn_init)(const int *devices, int count);
typedef void           (*fn_shutdown)(void);
typedef int            (*fn_device_count)(void);
typedef int            (*fn_device_at)(int index);
typedef int            (*fn_mem_info)(int device, size_t *free_bytes, size_t *total_bytes);
typedef int            (*fn_device_is_integrated)(int device);
typedef void           (*fn_stats)(int device, size_t *tensor_count, size_t *tensor_bytes);
typedef void           (*fn_group_stats)(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                         double *h2d_ms, double *kernel_ms, double *d2h_ms);
typedef int            (*fn_expert_mlp)(ColiCudaTensor *gate, ColiCudaTensor *up,
                                        ColiCudaTensor *down, float *y, const float *x, int S);
typedef int            (*fn_expert_group)(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
                                          ColiCudaTensor *const *downs, const int *rows, int count,
                                          float *y, const float *x, const float *wrow,
                                          const int *tokrow, int S_tok, int accum_ok,
                                          int x_device,float *out_device);
typedef int            (*fn_expert_group_collect)(int device, int home_device, float *x_dev, int D);
typedef int            (*fn_expert_group_issue)(ColiCudaTensor *const *gates,
                                                ColiCudaTensor *const *ups,
                                                ColiCudaTensor *const *downs,
                                                const int *rows, int count, const float *x);
typedef const float *  (*fn_expert_group_take)(int device);
typedef int            (*fn_attention_absorb)(ColiCudaTensor *kv_b, float *ctx, const float *q,
                                              const float *latent, const float *rope, int H, int Q,
                                              int R, int V, int K, int T, float attention_scale);
typedef int            (*fn_tensor_upload)(ColiCudaTensor **tensor, const void *weights,
                                           const float *scales, int fmt, int I, int O, int device);
typedef int            (*fn_tensor_upload_g)(ColiCudaTensor **tensor, const void *weights,
                                             const float *scales, int fmt, int I, int O,
                                             int device, int gs);
typedef int            (*fn_tensor_wrap_host)(ColiCudaTensor **tensor, const void *weights,
                                              const float *scales, int fmt, int I, int O, int device);
typedef int            (*fn_tensor_upload_nvfp4)(ColiCudaTensor **tensor, const uint8_t *weights,
                                                 const uint8_t *block_scales, float tensor_scale,
                                                 float input_scale, int scale_layout,
                                                 int I, int O, int device);
typedef int            (*fn_matmul)(ColiCudaTensor **tensor, float *y, const float *x,
                                    const void *weights, const float *scales,
                                    int fmt, int S, int I, int O, int device, int gs);
typedef int            (*fn_matmul_nvfp4)(ColiCudaTensor **tensor, float *y, const float *x,
                                          const uint8_t *weights, const uint8_t *block_scales,
                                          float tensor_scale, float input_scale, int scale_layout,
                                          int S, int I, int O, int device);
typedef void           (*fn_nvfp4_stats)(uint64_t *native_calls, uint64_t *generic_calls,
                                         uint64_t *fallback_native_unavailable, uint64_t *failures);
typedef void           (*fn_nvfp4_fallback_stats)(uint64_t *shape,uint64_t *rejected,uint64_t *launch);
typedef void           (*fn_nvfp4_grouped_stats)(uint64_t *calls,uint64_t *problems,uint64_t *fallbacks);
typedef int            (*fn_nvfp4_native_capable)(int device);
typedef void           (*fn_tensor_free)(ColiCudaTensor *tensor);
typedef size_t         (*fn_tensor_bytes)(const ColiCudaTensor *tensor);
typedef int            (*fn_tensor_device)(const ColiCudaTensor *tensor);

/* --- #111 GPU resident pipeline additions (matched to backend_cuda.h) --- */


/* --- #111 GPU resident pipeline additions (matched to backend_cuda.h) --- */
typedef int (*fn_attention_absorb_batch)(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale);
typedef int (*fn_attention_absorb_batch_dev)(ColiCudaTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_attention_absorb_kvdev)(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,const float *latent_scale,const float *rope_scale,int H,int Q,int R,int V,int K,int T, float scale);
typedef int (*fn_attention_project_batch)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale);
typedef int (*fn_attention_project_ragged)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale);
typedef int (*fn_attention_project_batch_dev)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,const float *latent_scale,const float *rope_scale, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_attention_project_batch_dev_out)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,const float *latent_scale,const float *rope_scale, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_prefill_attn_gemm)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,const float *latent_scale,const float *rope_scale,int S,int H,int Q,int R,int V,int K,int T,float scale, const int *sel_host,int sB0,int sel_topk);
typedef int (*fn_prefill_dsa_select)(int device,ColiCudaDsaChain *dsa, const float *xn_dev,const float *qres_dev, int S,int pos_base,int sB0,int D,int q_lora,int qk_rope,float theta);
typedef void (*fn_dsac_times)(double *sync_s, double *topk_s);
typedef void (*fn_dsac_phase_times)(double out[3]);
typedef void (*fn_dsac_tcg_stats)(uint64_t *rows,uint64_t *fallbacks);
typedef int (*fn_kv_f16)(void);
typedef int (*fn_kv_dtype)(void);
typedef void (*fn_kv_fp8_stats)(uint64_t*,uint64_t*,uint64_t*);
typedef int (*fn_pipe_upload_kv)(int device,void *dst,const float *src,size_t elems,size_t elem_off);
typedef int (*fn_pipe_copy2d_kv)(int device,void *dst,int dpitch,const float *src,int spitch,int width,int height,size_t elem_off);
typedef int (*fn_pipe_upload_kv_rows)(int device,void *dst,float *scale_dst,const float *src,int rows,int width,int row_off);
typedef int (*fn_pipe_copy2d_kv_rows)(int device,void *dst,float *scale_dst,const float *src,int spitch,int rows,int width,int row_off);
typedef int (*fn_attention_project_sel)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q,const float *latent_dev,const float *rope_dev,const float *latent_scale,const float *rope_scale,const int *sel,int ns,int H,int Q,int R,int V,int K,float scale);
typedef int (*fn_pipe_add)(int device,float *x_dev,const float *t_dev,size_t n);
typedef void * (*fn_pipe_alloc)(int device,size_t bytes);
typedef int (*fn_pipe_copy2d)(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height);
typedef int (*fn_pipe_download)(int device,const void *src,void *dst,size_t bytes);
typedef void (*fn_pipe_free)(int device,void *p);
typedef int (*fn_pipe_gemm)(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S);
typedef int (*fn_pipe_attn_chain_v2)(int device,
        float *x_dev, float *nrm_dev, float *nrm_host,
        float *kv_host_L, float *kv_host_R,
        const ColiCudaTensor *qa, const ColiCudaTensor *qb,
        const ColiCudaTensor *kva, const ColiCudaTensor *kvb,
        const ColiCudaTensor *o_proj,
        const float *w_in, const float *w_qa, const float *w_kva, const float *w_post,
        float *d_Lc, float *d_Rc, float *d_Lc_scale, float *d_Rc_scale,
        int D, int H, int q_lora, int kv_lora,
        int qk_nope, int qk_rope, int vh,
        int S, int pos_base, int kv_start,
        float eps, float theta, float attn_scale,
        const float *d_router, int E, float *scores_host,
        const ColiCudaTensor *shg, const ColiCudaTensor *shu,
        const ColiCudaTensor *shd, int sI, float *xn_host,
        ColiCudaDsaChain *dsa);
typedef int (*fn_pipe_peer_copy)(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes);
typedef int (*fn_pipe_rmsnorm)(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps);
typedef int (*fn_pipe_rmsnorm_s)(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride);
typedef int (*fn_pipe_rope)(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta);
typedef int (*fn_pipe_rope_base)(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta);
typedef int (*fn_pipe_rows_add)(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D);
typedef float * (*fn_pipe_scratch)(int device,int slot,size_t bytes);
typedef int (*fn_pipe_silu_mul)(int device,float *gate_dev,const float *up_dev,size_t n);
typedef int (*fn_pipe_sync)(int device);
typedef int (*fn_pipe_upload)(int device,void *dst,const void *src,size_t bytes);
typedef int (*fn_shared_mlp_w4a16)(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S);
typedef int (*fn_tensor_update)(ColiCudaTensor *tensor, const void *weights, const float *scales);
typedef int (*fn_tensor_update_nvfp4)(ColiCudaTensor *tensor, const uint8_t *weights,
                                      const uint8_t *block_scales, float tensor_scale,
                                      float input_scale);

/* Resolved pointers, plus a flag so we attempt the load at most once. */
static struct {
    int loaded;        /* 1 = load attempted (success or fail), 0 = not yet */
    int available;     /* 1 = DLL loaded and all symbols resolved */
    HMODULE dll;
    fn_init            init;
    fn_shutdown        shutdown;
    fn_device_count    device_count;
    fn_device_at       device_at;
    fn_mem_info        mem_info;
    fn_device_is_integrated device_is_integrated;
    fn_stats           stats;
    fn_group_stats     group_stats;
    fn_expert_mlp      expert_mlp;
    fn_expert_group    expert_group;
    fn_expert_group_collect expert_group_collect;
    fn_expert_group_issue expert_group_issue;
    fn_expert_group_take expert_group_take;
    fn_attention_absorb attention_absorb;
    fn_tensor_upload   tensor_upload;
    fn_tensor_upload_g tensor_upload_g;
    fn_tensor_wrap_host tensor_wrap_host;
    fn_tensor_upload_nvfp4 tensor_upload_nvfp4;
    fn_tensor_upload_nvfp4 tensor_wrap_host_nvfp4;
    fn_matmul          matmul;
    fn_matmul_nvfp4    matmul_nvfp4;
    fn_nvfp4_stats     nvfp4_stats;
    fn_nvfp4_fallback_stats nvfp4_fallback_stats;
    fn_nvfp4_grouped_stats nvfp4_grouped_stats;
    fn_nvfp4_native_capable nvfp4_native_capable;
    fn_tensor_free     tensor_free;
    fn_tensor_bytes    tensor_bytes;
    fn_tensor_device   tensor_device;

    fn_attention_absorb_batch attention_absorb_batch;
    fn_attention_absorb_batch_dev attention_absorb_batch_dev;
    fn_attention_absorb_kvdev attention_absorb_kvdev;
    fn_attention_project_batch attention_project_batch;
    fn_attention_project_ragged attention_project_ragged;
    fn_attention_project_batch_dev attention_project_batch_dev;
    fn_attention_project_batch_dev_out attention_project_batch_dev_out;
    fn_prefill_attn_gemm prefill_attn_gemm;
    fn_prefill_dsa_select prefill_dsa_select;
    fn_dsac_times      dsac_times;
    fn_dsac_phase_times dsac_phase_times;
    fn_dsac_tcg_stats  dsac_tcg_stats;
    fn_kv_f16          kv_f16;
    fn_kv_dtype        kv_dtype;
    fn_kv_fp8_stats    kv_fp8_stats;
    fn_pipe_upload_kv  pipe_upload_kv;
    fn_pipe_copy2d_kv  pipe_copy2d_kv;
    fn_pipe_upload_kv_rows pipe_upload_kv_rows;
    fn_pipe_copy2d_kv_rows pipe_copy2d_kv_rows;
    fn_attention_project_sel attention_project_sel;
    fn_pipe_add pipe_add;
    fn_pipe_alloc pipe_alloc;
    fn_pipe_copy2d pipe_copy2d;
    fn_pipe_download pipe_download;
    fn_pipe_free pipe_free;
    fn_pipe_gemm pipe_gemm;
    fn_pipe_attn_chain_v2 pipe_attn_chain_v2;
    fn_pipe_peer_copy pipe_peer_copy;
    fn_pipe_rmsnorm pipe_rmsnorm;
    fn_pipe_rmsnorm_s pipe_rmsnorm_s;
    fn_pipe_rope pipe_rope;
    fn_pipe_rope_base pipe_rope_base;
    fn_pipe_rows_add pipe_rows_add;
    fn_pipe_scratch pipe_scratch;
    fn_pipe_silu_mul pipe_silu_mul;
    fn_pipe_sync pipe_sync;
    fn_pipe_upload pipe_upload;
    fn_shared_mlp_w4a16 shared_mlp_w4a16;
    fn_tensor_update tensor_update;
    fn_tensor_update_nvfp4 tensor_update_nvfp4;
} g_cuda;

/* Resolve the DLL and all 11 symbols. Returns 1 on success, 0 otherwise.
 * Idempotent: the first call (success or fail) sticks; later calls are no-ops
 * that return the cached result. The engine treats a 0 return as "CUDA
 * unavailable" and falls back to the CPU path without aborting. */
static int coli_cuda_load(void){
    if(g_cuda.loaded) return g_cuda.available;
    g_cuda.loaded = 1;

    /* Load coli_cuda.dll from the engine's OWN directory, by absolute path —
     * never a bare name. LoadLibraryA("coli_cuda.dll") searches the current
     * working directory (and, without SafeDllSearchMode, other writable dirs):
     * an attacker who drops a coli_cuda.dll where the user launches glm.exe (or
     * inside a downloaded model directory the user cd's into) would get their
     * DllMain run at load — classic DLL hijacking -> arbitrary code execution.
     * Resolving the path next to glm.exe and loading THAT specific file with
     * LOAD_WITH_ALTERED_SEARCH_PATH anchors both the DLL and its dependency
     * search to the trusted install directory instead of the CWD. */
    char dllpath[MAX_PATH];
    DWORD mn = GetModuleFileNameA(NULL, dllpath, (DWORD)sizeof(dllpath));
    if(mn > 0 && mn < sizeof(dllpath)){
        char *slash = strrchr(dllpath, '\\');
        if(slash && (size_t)(slash + 1 - dllpath) + sizeof("coli_cuda.dll") <= sizeof(dllpath)){
            strcpy(slash + 1, "coli_cuda.dll");
            g_cuda.dll = LoadLibraryExA(dllpath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
    }
    if(!g_cuda.dll){
        /* fallback (GetModuleFileNameA praticamente non fallisce): cerca solo
         * nella dir dell'applicazione e in System32, MAI la CWD. */
        g_cuda.dll = LoadLibraryExA("coli_cuda.dll", NULL,
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    }
    if(!g_cuda.dll){
        fprintf(stderr, "[CUDA] coli_cuda.dll not found; GPU tier disabled "
                        "(CPU path remains active).\n");
        return 0;
    }

    #define RESOLVE(name, type) \
        /* GetProcAddress returns FARPROC (void(*)(void)); casting it to a   \
         * specific function-pointer type is the standard LoadLibrary idiom. \
         * -Wcast-function-type flags it but it is safe: the DLL exported     \
         * the symbol with extern "C" and the exact signature we expect. */   \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"") \
        g_cuda.name = (type)GetProcAddress(g_cuda.dll, "coli_cuda_" #name); \
        _Pragma("GCC diagnostic pop") \
        if(!g_cuda.name){ \
            fprintf(stderr, "[CUDA] coli_cuda.dll missing symbol coli_cuda_" #name "\n"); \
            FreeLibrary(g_cuda.dll); g_cuda.dll=NULL; return 0; }

    RESOLVE(init,           fn_init)
    RESOLVE(shutdown,       fn_shutdown)
    RESOLVE(device_count,   fn_device_count)
    RESOLVE(device_at,      fn_device_at)
    RESOLVE(mem_info,       fn_mem_info)
    RESOLVE(device_is_integrated, fn_device_is_integrated)
    RESOLVE(stats,          fn_stats)
    RESOLVE(group_stats,    fn_group_stats)
    RESOLVE(expert_mlp,     fn_expert_mlp)
    RESOLVE(expert_group,   fn_expert_group)
    RESOLVE(expert_group_collect, fn_expert_group_collect)
    RESOLVE(expert_group_issue, fn_expert_group_issue)
    RESOLVE(expert_group_take, fn_expert_group_take)
    RESOLVE(attention_absorb, fn_attention_absorb)
    RESOLVE(tensor_upload,  fn_tensor_upload)
    RESOLVE(tensor_upload_g, fn_tensor_upload_g)
    RESOLVE(tensor_wrap_host, fn_tensor_wrap_host)
    RESOLVE(tensor_upload_nvfp4, fn_tensor_upload_nvfp4)
    RESOLVE(tensor_wrap_host_nvfp4, fn_tensor_upload_nvfp4)
    RESOLVE(matmul,         fn_matmul)
    RESOLVE(matmul_nvfp4,   fn_matmul_nvfp4)
    RESOLVE(nvfp4_stats,    fn_nvfp4_stats)
    RESOLVE(nvfp4_fallback_stats, fn_nvfp4_fallback_stats)
    RESOLVE(nvfp4_grouped_stats, fn_nvfp4_grouped_stats)
    RESOLVE(nvfp4_native_capable, fn_nvfp4_native_capable)
    RESOLVE(tensor_free,    fn_tensor_free)
    RESOLVE(tensor_bytes,   fn_tensor_bytes)
    RESOLVE(tensor_device,  fn_tensor_device)

    RESOLVE(attention_absorb_batch, fn_attention_absorb_batch)
    RESOLVE(attention_absorb_batch_dev, fn_attention_absorb_batch_dev)
    RESOLVE(attention_absorb_kvdev, fn_attention_absorb_kvdev)
    RESOLVE(attention_project_batch, fn_attention_project_batch)
    RESOLVE(attention_project_ragged, fn_attention_project_ragged)
    RESOLVE(attention_project_batch_dev, fn_attention_project_batch_dev)
    RESOLVE(attention_project_batch_dev_out, fn_attention_project_batch_dev_out)
    RESOLVE(prefill_attn_gemm, fn_prefill_attn_gemm)
    RESOLVE(prefill_dsa_select, fn_prefill_dsa_select)
    RESOLVE(dsac_times,     fn_dsac_times)
    RESOLVE(dsac_phase_times, fn_dsac_phase_times)
    RESOLVE(dsac_tcg_stats, fn_dsac_tcg_stats)
    RESOLVE(kv_f16,         fn_kv_f16)
    RESOLVE(kv_dtype,       fn_kv_dtype)
    RESOLVE(kv_fp8_stats,   fn_kv_fp8_stats)
    RESOLVE(pipe_upload_kv, fn_pipe_upload_kv)
    RESOLVE(pipe_copy2d_kv, fn_pipe_copy2d_kv)
    RESOLVE(pipe_upload_kv_rows, fn_pipe_upload_kv_rows)
    RESOLVE(pipe_copy2d_kv_rows, fn_pipe_copy2d_kv_rows)
    RESOLVE(attention_project_sel, fn_attention_project_sel)
    RESOLVE(pipe_add, fn_pipe_add)
    RESOLVE(pipe_alloc, fn_pipe_alloc)
    RESOLVE(pipe_copy2d, fn_pipe_copy2d)
    RESOLVE(pipe_download, fn_pipe_download)
    RESOLVE(pipe_free, fn_pipe_free)
    RESOLVE(pipe_gemm, fn_pipe_gemm)
    RESOLVE(pipe_attn_chain_v2, fn_pipe_attn_chain_v2)
    RESOLVE(pipe_peer_copy, fn_pipe_peer_copy)
    RESOLVE(pipe_rmsnorm, fn_pipe_rmsnorm)
    RESOLVE(pipe_rmsnorm_s, fn_pipe_rmsnorm_s)
    RESOLVE(pipe_rope, fn_pipe_rope)
    RESOLVE(pipe_rope_base, fn_pipe_rope_base)
    RESOLVE(pipe_rows_add, fn_pipe_rows_add)
    RESOLVE(pipe_scratch, fn_pipe_scratch)
    RESOLVE(pipe_silu_mul, fn_pipe_silu_mul)
    RESOLVE(pipe_sync, fn_pipe_sync)
    RESOLVE(pipe_upload, fn_pipe_upload)
    RESOLVE(shared_mlp_w4a16, fn_shared_mlp_w4a16)
    RESOLVE(tensor_update, fn_tensor_update)
    RESOLVE(tensor_update_nvfp4, fn_tensor_update_nvfp4)
    #undef RESOLVE

    g_cuda.available = 1;
    return 1;
}

/* ---- Public wrappers: match backend_cuda.h signatures exactly.
 * Each forwards to the resolved pointer; if the DLL never loaded, return the
 * "not initialized" result the engine already handles (init returns 0, matmul
 * returns 0 so the caller marks the tensor cuda_failed and uses CPU). ---- */

int coli_cuda_init(const int *devices, int count){
    if(!coli_cuda_load()) return 0;
    return g_cuda.init(devices, count);
}

void coli_cuda_shutdown(void){
    if(g_cuda.available && g_cuda.shutdown) g_cuda.shutdown();
}

int coli_cuda_device_count(void){
    if(!g_cuda.available) return 0;
    return g_cuda.device_count();
}

int coli_cuda_device_at(int index){
    if(!g_cuda.available) return -1;
    return g_cuda.device_at(index);
}

int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes){
    if(!g_cuda.available){ if(free_bytes)*free_bytes=0; if(total_bytes)*total_bytes=0; return 0; }
    return g_cuda.mem_info(device, free_bytes, total_bytes);
}

int coli_cuda_device_is_integrated(int device){
    if(!g_cuda.available){ return 0; }
    return g_cuda.device_is_integrated(device);
}

void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes){
    if(!g_cuda.available){ if(tensor_count)*tensor_count=0; if(tensor_bytes)*tensor_bytes=0; return; }
    g_cuda.stats(device, tensor_count, tensor_bytes);
}

void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms){
    if(!g_cuda.available){
        if(calls)*calls=0; if(experts)*experts=0; if(rows)*rows=0;
        if(h2d_ms)*h2d_ms=0; if(kernel_ms)*kernel_ms=0; if(d2h_ms)*d2h_ms=0;
        return;
    }
    g_cuda.group_stats(calls, experts, rows, h2d_ms, kernel_ms, d2h_ms);
}

int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_mlp(gate, up, down, y, x, S);
}

int coli_cuda_expert_group(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs, const int *rows, int count,
                           float *y, const float *x, const float *wrow,
                           const int *tokrow, int S_tok, int accum_ok,
                           int x_device,float *out_device){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_group(gates,ups,downs,rows,count,y,x,wrow,tokrow,S_tok,accum_ok,
                               x_device,out_device);
}

int coli_cuda_expert_group_collect(int device,int home_device,float *x_dev,int D){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_group_collect(device,home_device,x_dev,D);
}

int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
        ColiCudaTensor *const *ups, ColiCudaTensor *const *downs,
        const int *rows, int count, const float *x){
    if(!g_cuda.available || !g_cuda.expert_group_issue) return 0;
    return g_cuda.expert_group_issue(gates,ups,downs,rows,count,x);
}

const float *coli_cuda_expert_group_take(int device){
    if(!g_cuda.available || !g_cuda.expert_group_take) return NULL;
    return g_cuda.expert_group_take(device);
}

int coli_cuda_attention_absorb(ColiCudaTensor *kv_b, float *ctx, const float *q,
                               const float *latent, const float *rope, int H, int Q,
                               int R, int V, int K, int T, float attention_scale){
    if(!g_cuda.available) return 0;
    return g_cuda.attention_absorb(kv_b, ctx, q, latent, rope, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_tensor_upload(ColiCudaTensor **tensor, const void *weights,
                            const float *scales, int fmt, int I, int O, int device){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_upload(tensor, weights, scales, fmt, I, O, device);
}

int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor, const void *weights,
        const float *scales, int fmt, int I, int O, int device, int gs){
    if(!g_cuda.available || !g_cuda.tensor_upload_g) return 0;
    return g_cuda.tensor_upload_g(tensor,weights,scales,fmt,I,O,device,gs);
}

int coli_cuda_tensor_wrap_host(ColiCudaTensor **tensor, const void *weights,
                               const float *scales, int fmt, int I, int O, int device){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_wrap_host(tensor, weights, scales, fmt, I, O, device);
}

int coli_cuda_tensor_upload_nvfp4(ColiCudaTensor **tensor, const uint8_t *weights,
                                  const uint8_t *block_scales, float tensor_scale,
                                  float input_scale, int scale_layout,
                                  int I, int O, int device){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_upload_nvfp4(tensor,weights,block_scales,tensor_scale,input_scale,
                                      scale_layout,I,O,device);
}

int coli_cuda_tensor_wrap_host_nvfp4(ColiCudaTensor **tensor, const uint8_t *weights,
                                     const uint8_t *block_scales, float tensor_scale,
                                     float input_scale, int scale_layout,
                                     int I, int O, int device){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_wrap_host_nvfp4(tensor,weights,block_scales,tensor_scale,input_scale,
                                         scale_layout,I,O,device);
}

int coli_cuda_matmul(ColiCudaTensor **tensor, float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device, int gs){
    if(!g_cuda.available) return 0;
    return g_cuda.matmul(tensor, y, x, weights, scales, fmt, S, I, O, device, gs);
}

int coli_cuda_matmul_nvfp4(ColiCudaTensor **tensor, float *y, const float *x,
                           const uint8_t *weights, const uint8_t *block_scales,
                           float tensor_scale, float input_scale, int scale_layout,
                           int S, int I, int O, int device){
    if(!g_cuda.available) return 0;
    return g_cuda.matmul_nvfp4(tensor,y,x,weights,block_scales,tensor_scale,input_scale,
                               scale_layout,S,I,O,device);
}

void coli_cuda_nvfp4_stats(uint64_t *native_calls, uint64_t *generic_calls,
                           uint64_t *fallback_native_unavailable, uint64_t *failures){
    if(!g_cuda.available){
        if(native_calls)*native_calls=0; if(generic_calls)*generic_calls=0;
        if(fallback_native_unavailable)*fallback_native_unavailable=0;
        if(failures)*failures=0; return;
    }
    g_cuda.nvfp4_stats(native_calls,generic_calls,fallback_native_unavailable,failures);
}

void coli_cuda_nvfp4_fallback_stats(uint64_t *shape,uint64_t *rejected,uint64_t *launch){
    if(!g_cuda.available){if(shape)*shape=0;if(rejected)*rejected=0;if(launch)*launch=0;return;}
    g_cuda.nvfp4_fallback_stats(shape,rejected,launch);
}

void coli_cuda_nvfp4_grouped_stats(uint64_t *calls,uint64_t *problems,uint64_t *fallbacks){
    if(!g_cuda.available){if(calls)*calls=0;if(problems)*problems=0;if(fallbacks)*fallbacks=0;return;}
    g_cuda.nvfp4_grouped_stats(calls,problems,fallbacks);
}

int coli_cuda_nvfp4_native_capable(int device){
    return g_cuda.available&&g_cuda.nvfp4_native_capable(device);
}

void coli_cuda_tensor_free(ColiCudaTensor *tensor){
    if(g_cuda.available && g_cuda.tensor_free) g_cuda.tensor_free(tensor);
}

size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_bytes(tensor);
}

int coli_cuda_tensor_device(const ColiCudaTensor *tensor){
    if(!g_cuda.available) return -1;
    return g_cuda.tensor_device(tensor);
}

/* ---- #111 pipeline wrappers ---- */


/* ---- #111 pipeline wrappers (see header for semantics) ---- */

int coli_cuda_attention_absorb_batch(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_batch(kv_b, ctx, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_batch_dev(kv_b_shard, ctx_dev, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,const float *latent_scale,const float *rope_scale,int H,int Q,int R,int V,int K,int T, float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_kvdev(kv_b, ctx, q, latent_dev, rope_dev, latent_scale, rope_scale, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_project_batch(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch(kv_b, o_proj, out, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_attention_project_ragged(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale){
    if(!coli_cuda_load()) return 0;
    return g_cuda.attention_project_ragged(kv_b,o_proj,out,q,latent,rope,lengths,
        S,H,Q,R,V,K,max_t,attention_scale);
}

int coli_cuda_attention_project_batch_dev(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,const float *latent_scale,const float *rope_scale, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch_dev(kv_b, o_proj, out, q_dev, latent_dev, rope_dev, latent_scale, rope_scale, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,const float *latent_scale,const float *rope_scale, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch_dev_out(kv_b, o_proj, out_dev, q_dev, latent_dev, rope_dev, latent_scale, rope_scale, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_prefill_attn_gemm(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,const float *latent_scale,const float *rope_scale,int S,int H,int Q,int R,int V,int K,int T,float scale, const int *sel_host,int sB0,int sel_topk){
    if(!g_cuda.available){ return 0; }
    return g_cuda.prefill_attn_gemm(kv_b, o_proj, out_dev, q_dev, latent_dev, rope_dev,latent_scale,rope_scale,S, H, Q, R, V, K, T, scale, sel_host, sB0, sel_topk);
}

int coli_cuda_prefill_dsa_select(int device,ColiCudaDsaChain *dsa, const float *xn_dev,const float *qres_dev, int S,int pos_base,int sB0,int D,int q_lora,int qk_rope,float theta){
    if(!g_cuda.available){ return 0; }
    return g_cuda.prefill_dsa_select(device, dsa, xn_dev, qres_dev, S, pos_base, sB0, D, q_lora, qk_rope, theta);
}

void coli_cuda_dsac_times(double *sync_s, double *topk_s){
    if(!g_cuda.available){ if(sync_s)*sync_s=0; if(topk_s)*topk_s=0; return; }
    g_cuda.dsac_times(sync_s, topk_s);
}

int coli_cuda_kv_f16(void){
    if(!g_cuda.available){ return 0; }
    return g_cuda.kv_f16();
}
int coli_cuda_kv_dtype(void){
    if(!g_cuda.available){ return 0; }
    return g_cuda.kv_dtype();
}
void coli_cuda_kv_fp8_stats(uint64_t *q,uint64_t *r,uint64_t *f){
    if(!g_cuda.available){if(q)*q=0;if(r)*r=0;if(f)*f=0;return;}
    g_cuda.kv_fp8_stats(q,r,f);
}

void coli_cuda_dsac_phase_times(double out[3]){
    if(!g_cuda.available){ out[0]=out[1]=out[2]=0; return; }
    g_cuda.dsac_phase_times(out);
}

void coli_cuda_dsac_tcg_stats(uint64_t *rows,uint64_t *fallbacks){
    if(!g_cuda.available){if(rows)*rows=0;if(fallbacks)*fallbacks=0;return;}
    g_cuda.dsac_tcg_stats(rows,fallbacks);
}

int coli_cuda_pipe_upload_kv(int device,void *dst,const float *src,size_t elems,size_t elem_off){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_upload_kv(device, dst, src, elems, elem_off);
}

int coli_cuda_pipe_copy2d_kv(int device,void *dst,int dpitch,const float *src,int spitch,int width,int height,size_t elem_off){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_copy2d_kv(device, dst, dpitch, src, spitch, width, height, elem_off);
}
int coli_cuda_pipe_upload_kv_rows(int device,void *dst,float *scale_dst,
        const float *src,int rows,int width,int row_off){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_upload_kv_rows(device,dst,scale_dst,src,rows,width,row_off);
}
int coli_cuda_pipe_copy2d_kv_rows(int device,void *dst,float *scale_dst,
        const float *src,int spitch,int rows,int width,int row_off){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_copy2d_kv_rows(device,dst,scale_dst,src,spitch,rows,width,row_off);
}

int coli_cuda_attention_project_sel(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q,const float *latent_dev,const float *rope_dev,const float *latent_scale,const float *rope_scale,const int *sel,int ns,int H,int Q,int R,int V,int K,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_sel(kv_b, o_proj, out, q, latent_dev, rope_dev,latent_scale,rope_scale,sel, ns, H, Q, R, V, K, scale);
}

int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_add(device, x_dev, t_dev, n);
}

void * coli_cuda_pipe_alloc(int device,size_t bytes){
    if(!g_cuda.available){ return NULL; }
    return g_cuda.pipe_alloc(device, bytes);
}

int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_copy2d(device, dst, dpitch, src, spitch, width, height);
}

int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_download(device, src, dst, bytes);
}

void coli_cuda_pipe_free(int device,void *p){
    if(!g_cuda.available){ return; }
    g_cuda.pipe_free(device, p);
}

int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_gemm(t, y_dev, x_dev, S);
}

int coli_cuda_pipe_attn_chain_v2(int device,
        float *x_dev, float *nrm_dev, float *nrm_host,
        float *kv_host_L, float *kv_host_R,
        const ColiCudaTensor *qa, const ColiCudaTensor *qb,
        const ColiCudaTensor *kva, const ColiCudaTensor *kvb,
        const ColiCudaTensor *o_proj,
        const float *w_in, const float *w_qa, const float *w_kva, const float *w_post,
        float *d_Lc, float *d_Rc,float *d_Lc_scale,float *d_Rc_scale,
        int D, int H, int q_lora, int kv_lora,
        int qk_nope, int qk_rope, int vh,
        int S, int pos_base, int kv_start,
        float eps, float theta, float attn_scale,
        const float *d_router, int E, float *scores_host,
        const ColiCudaTensor *shg, const ColiCudaTensor *shu,
        const ColiCudaTensor *shd, int sI, float *xn_host,
        ColiCudaDsaChain *dsa){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_attn_chain_v2(device,x_dev,nrm_dev,nrm_host,kv_host_L,kv_host_R,
        qa,qb,kva,kvb,o_proj,w_in,w_qa,w_kva,w_post,d_Lc,d_Rc,d_Lc_scale,d_Rc_scale,D,H,q_lora,kv_lora,
        qk_nope,qk_rope,vh,S,pos_base,kv_start,eps,theta,attn_scale,
        d_router,E,scores_host,shg,shu,shd,sI,xn_host,dsa);
}

int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_peer_copy(dst_dev, dst, src_dev, src, bytes);
}

int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rmsnorm(device, y_dev, x_dev, w_dev, S, D, eps);
}

int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rmsnorm_s(device, y_dev, x_dev, w_dev, S, D, eps, xstride, ystride);
}

int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rope(device, v_dev, pos_dev, rows, stride, offset, R, heads, theta);
}

int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rope_base(device, v_dev, pos_base, rows, stride, offset, R, heads, theta);
}

int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rows_add(device, x_dev, partial_dev, rows_dev, nrows, D);
}

float * coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    if(!g_cuda.available){ return NULL; }
    return g_cuda.pipe_scratch(device, slot, bytes);
}

int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_silu_mul(device, gate_dev, up_dev, n);
}

int coli_cuda_pipe_sync(int device){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_sync(device);
}

int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_upload(device, dst, src, bytes);
}

int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S){
    if(!g_cuda.available){ return 0; }
    return g_cuda.shared_mlp_w4a16(gate, up, down, y, x, S);
}

int coli_cuda_tensor_update(ColiCudaTensor *tensor, const void *weights, const float *scales){
    if(!g_cuda.available){ return 0; }
    return g_cuda.tensor_update(tensor, weights, scales);
}

int coli_cuda_tensor_update_nvfp4(ColiCudaTensor *tensor, const uint8_t *weights,
                                  const uint8_t *block_scales, float tensor_scale,
                                  float input_scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.tensor_update_nvfp4(tensor,weights,block_scales,tensor_scale,input_scale);
}

#endif /* _WIN32 */
