#ifndef COLIBRI_BACKEND_CUDA_H
#define COLIBRI_BACKEND_CUDA_H

#include <stddef.h>
#include <stdint.h>
#include "tensor_format.h"

/* COLI_CUDA_DLLEXPORT marks functions exported from coli_cuda.dll on Windows.
 * Define COLI_CUDA_BUILDING_DLL when compiling the .cu into the DLL (so the
 * functions are __declspec(dllexport)); the host loader does NOT include this
 * header's declarations — it resolves symbols at runtime via GetProcAddress. */
#if defined(_WIN32) && defined(COLI_CUDA_BUILDING_DLL)
#define COLI_CUDA_DLLEXPORT __declspec(dllexport)
#else
#define COLI_CUDA_DLLEXPORT
#endif


#ifdef __cplusplus
extern "C" {
#endif

#define COLI_CUDA_MAX_DEVICES 16

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct ColiCudaTensor ColiCudaTensor;

/* Devices are CUDA ordinals, not positions in the input list. */
COLI_CUDA_DLLEXPORT int coli_cuda_init(const int *devices, int count);
COLI_CUDA_DLLEXPORT void coli_cuda_shutdown(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_count(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_at(int index);
COLI_CUDA_DLLEXPORT int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
/* 1 when cudaMalloc storage shares physical system RAM, 0 for discrete VRAM. */
COLI_CUDA_DLLEXPORT int coli_cuda_device_is_integrated(int device);
/* device < 0 returns aggregate statistics for all configured devices. */
COLI_CUDA_DLLEXPORT void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms);

/* Upload without executing, so capacity failures happen during model startup. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
        const void *weights, const float *scales,
        int fmt, int I, int O, int device, int gs);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);
/* Wrap pageable host memory for direct kernel reads on coherent CUDA systems.
 * No allocation or copy is performed; the caller retains ownership and must
 * keep weights/scales alive until coli_cuda_tensor_free(). */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_wrap_host(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload_nvfp4(ColiCudaTensor **tensor,
                            const uint8_t *weights, const uint8_t *block_scales,
                            float tensor_scale, float input_scale, int scale_layout,
                            int I, int O, int device);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_wrap_host_nvfp4(ColiCudaTensor **tensor,
                            const uint8_t *weights, const uint8_t *block_scales,
                            float tensor_scale, float input_scale, int scale_layout,
                            int I, int O, int device);

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt is a ColiTensorFormat from tensor_format.h.
 * gs is the group size for fmt=4 (0 for all other formats).
 * The first successful call uploads W and its scales; later calls reuse it.
 * Returns 1 on success and 0 when CUDA is not initialized or the format is invalid.
 */
COLI_CUDA_DLLEXPORT int coli_cuda_matmul(ColiCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device, int gs);
COLI_CUDA_DLLEXPORT int coli_cuda_matmul_nvfp4(ColiCudaTensor **tensor,
                     float *y, const float *x, const uint8_t *weights,
                     const uint8_t *block_scales, float tensor_scale,
                     float input_scale, int scale_layout,
                     int S, int I, int O, int device);
COLI_CUDA_DLLEXPORT void coli_cuda_nvfp4_stats(uint64_t *native_calls,
                     uint64_t *generic_calls, uint64_t *fallback_native_unavailable,
                     uint64_t *failures);
COLI_CUDA_DLLEXPORT void coli_cuda_nvfp4_fallback_stats(uint64_t *shape,
                     uint64_t *rejected, uint64_t *launch);
COLI_CUDA_DLLEXPORT void coli_cuda_nvfp4_grouped_stats(uint64_t *calls,
                     uint64_t *problems, uint64_t *fallbacks);
COLI_CUDA_DLLEXPORT int coli_cuda_nvfp4_native_capable(int device);

/* Fused expert pipeline: y = down(silu(gate(x)) * up(x)).  All three tensors
 * must already be resident on one device.  Activations cross PCIe once in
 * each direction instead of once per matrix. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S);

/* Prefill-oriented shared expert path.  INT4 weights stay packed in global
 * memory, activations are converted to FP16 per tile, and Tensor Cores
 * accumulate into FP32.  Unlike COLI_CUDA_TC_INT4 this does not quantize the
 * activation to INT4. */
COLI_CUDA_DLLEXPORT int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate, ColiCudaTensor *up,
                               ColiCudaTensor *down, float *y,
                               const float *x, int S);

/* Packed group of same-shaped experts. Inputs and outputs contain sum(rows)
 * consecutive [D] rows in call order. */
/* Async issue/take split of the group call below (Inc.4): issue launches on the
 * device stream and returns; take syncs and returns the pinned result rows (valid
 * until the next issue on that device). Small totals only (<=8 rows); one
 * outstanding issue per device. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                               ColiCudaTensor *const *ups,
                               ColiCudaTensor *const *downs,
                               const int *rows, int count, const float *x);
COLI_CUDA_DLLEXPORT const float *coli_cuda_expert_group_take(int device);

COLI_CUDA_DLLEXPORT int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                           ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x,
        const float *wrow, const int *tokrow, int S_tok, int accum_ok,
        int x_device, float *out_device);
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_collect(int device,int home_device,float *x_dev,int D);

/* Decode-only MLA weight-absorption core for one token. kv_b is [H*(Q+V),K]. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb(ColiCudaTensor *kv_b,float *ctx,const float *q,
                               const float *latent,const float *rope,int H,int Q,
                               int R,int V,int K,int T,float attention_scale);

/* Causal MLA absorption for S contiguous rows from one sequence.  The KV
 * arrays contain T rows ending at the final query; query s attends T-S+s+1
 * rows.  One transfer and one launch replace S host round-trips. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_batch(ColiCudaTensor *kv_b,float *ctx,const float *q,
                                     const float *latent,const float *rope,int S,
                                     int H,int Q,int R,int V,int K,int T,
                                     float attention_scale);

/* Same attention batch followed immediately by resident o_proj on the same
 * device.  Only the final [S,D] tensor crosses back to the host. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
                                      float *out,const float *q,const float *latent,
                                      const float *rope,int S,int H,int Q,int R,
                                      int V,int K,int T,float attention_scale);

COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_ragged(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale);

COLI_CUDA_DLLEXPORT void coli_cuda_tensor_free(ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_device(const ColiCudaTensor *tensor);

/* Replace a resident tensor's contents without reallocating its device slot. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                            const void *weights, const float *scales);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_update_nvfp4(ColiCudaTensor *tensor,
                            const uint8_t *weights, const uint8_t *block_scales,
                            float tensor_scale, float input_scale);

/* ---- resident-pipeline primitives (Inc.0): device-pointer entry points ---- */
COLI_CUDA_DLLEXPORT float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes);
/* COLI_KV_F16 (default 1): the device KV/Ic shadows are stored as fp16.  The
 * host canonical caches stay exact fp32.  Shadow pointers remain float* in
 * every signature (opaque to the host); ALL element offsets into a shadow must
 * go through these helpers or be byte-scaled by the element size (2 or 4). */
/* Effective shadow dtype after COLI_KV_DTYPE precedence and legacy fallback:
 * 0=fp32, 1=fp16, 2=fp8 E4M3. Invalid settings make CUDA initialization fail. */
COLI_CUDA_DLLEXPORT int coli_cuda_kv_dtype(void);
COLI_CUDA_DLLEXPORT int coli_cuda_kv_f16(void); /* compatibility: dtype==fp16 */
COLI_CUDA_DLLEXPORT void coli_cuda_kv_fp8_stats(uint64_t *quantized_rows,
                            uint64_t *reader_rows,uint64_t *fallbacks);
/* Upload host fp32 rows into a shadow at element offset, converting if f16. */
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_upload_kv(int device,void *dst,const float *src,
                            size_t elems,size_t elem_off);
/* Strided device fp32 rows -> shadow rows (the prefill KV writer). */
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_copy2d_kv(int device,void *dst,int dpitch,
                            const float *src,int spitch,int width,int height,
                            size_t elem_off);
/* FP8-aware row writers. scale_dst is required only for fp8 and contains one
 * fp32 scale per row. Values and scales are completed on the same stream. */
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_upload_kv_rows(int device,void *dst,
                            float *scale_dst,const float *src,int rows,int width,
                            int row_off);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_copy2d_kv_rows(int device,void *dst,
                            float *scale_dst,const float *src,int spitch,
                            int rows,int width,int row_off);
/* DSA selection support for the fused chain (phase 2 inc.2).  All-or-nothing:
 * pass NULL to keep the dense absorb.  On indexer-FULL layers the ix_* tensors
 * are set and the chain computes the new k_idx row + selection scores on the
 * device Ic shadow, downloads the score rows and calls topk_fn (host, exact
 * CPU top-k) to refresh sel/nsel before the absorb; 'shared' layers pass the
 * previous FULL layer's sel/nsel with ix_*==NULL.  sel entries are absolute
 * KV positions (kv_start must be 0). */
typedef struct {
    const ColiCudaTensor *ix_wq, *ix_wk, *ix_wp;  /* NULL on 'shared' layers */
    const float *knw_dev, *knb_dev;   /* k_norm LayerNorm weight/bias, device [hd] */
    float *d_Ic;                      /* device index-key shadow [max_t, hd] */
    float *d_Ic_scale;                /* fp8 only: one scale per Ic row */
    int nh, hd, topk;
    int *sel, *nsel;                  /* host [S*topk] / [S]; refreshed on FULL layers */
    float *iscore_host;               /* host staging [S*(pos_base+S)] for score rows */
    int (*topk_fn)(void *u, const float *scores, int S, int pos_base, int topk,
                   int *sel, int *nsel);
    void *topk_user;
    float *ic_host;                   /* out: the S new k_idx rows (host Ic canonical) */
    int score_off;                    /* COLI_DSA_REFRESH reuse token: append k_idx to the
                                       * Ic shadow but skip scoring/top-k; sel/nsel arrive
                                       * prefilled by the caller (topk_fn stays NULL) */
} ColiCudaDsaChain;
/* Cumulative wall-time of the chain's mid-sync score download (includes the
 * implicit pipeline drain) and of the host top-k callback, for COLI_DBG_DSACHAIN. */
COLI_CUDA_DLLEXPORT void coli_cuda_dsac_times(double *sync_s, double *topk_s);
/* COLI_DBG_DSACHAIN=2: cumulative GPU time of the chain's three phases
 * (loop1 projections+KV writes+scoring, loop2 absorb+o_proj, tail). */
COLI_CUDA_DLLEXPORT void coli_cuda_dsac_phase_times(double out[3]);
/* Decode selected-attention tensor-core gather engagement. */
COLI_CUDA_DLLEXPORT void coli_cuda_dsac_tcg_stats(uint64_t *rows, uint64_t *fallbacks);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_attn_chain_v2(int device,
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
COLI_CUDA_DLLEXPORT void *coli_cuda_pipe_alloc(int device,size_t bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_pipe_free(int device,void *p);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                           const float *w_dev,int S,int D,float eps);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows,
                        int stride,int offset,int R,int heads,float theta);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                            const int *rows_dev,int nrows,int D);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                             const float *w_dev,int S,int D,float eps,
                             int xstride,int ystride);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                             int stride,int offset,int R,int heads,float theta);
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_resident_issue(ColiCudaTensor *const *gates,
        ColiCudaTensor *const *ups, ColiCudaTensor *const *downs,
        const float *weights, int count,
        int home_device, const float *x_src_dev, float *partial_slot_dev);
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_resident_take(int home_device,const int *devices,
        int n_issued,float *slots_dev,float *acc_dev,int D);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_router(int device,const float *x_dev,
        const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,
        float topp,int norm_topk,float routed_scale,
        int *idx_host,float *w_host,int *keff_host);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                          int spitch,int width,int height);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch_dev(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        const float *latent_scale,const float *rope_scale,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *kv_b_shard,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *kv_b,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,const float *latent_scale,
        const float *rope_scale,int H,int Q,int R,int V,int K,int T,
        float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_sel(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const float *latent_dev,const float *rope_dev,
        const float *latent_scale,const float *rope_scale,
        const int *sel,int ns,int H,int Q,int R,int V,int K,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                             const float *src,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        const float *latent_scale,const float *rope_scale,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
/* sel_host!=NULL: DSA prefill phase split — rows [sB0,S) absorb over their own
 * top-`sel_topk` list (absolute positions, kv_start must be 0), rows [0,sB0)
 * keep the causal GEMM path.  sel_host=NULL: dense causal prefill (sB0/sel_topk
 * ignored). */
COLI_CUDA_DLLEXPORT int coli_cuda_prefill_attn_gemm(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        const float *latent_scale,const float *rope_scale,
        int S,int H,int Q,int R,int V,int K,int T,float scale,
        const int *sel_host,int sB0,int sel_topk);
/* DSA prefill indexer pass: k_idx for the S new rows into the device Ic shadow
 * (downloaded to dsa->ic_host, host canonical) + selection scores for rows
 * [sB0,S) downloaded to dsa->iscore_host [S-sB0, pos_base+S] for the exact
 * host top-k.  Uses dsa->{ix_*, knw/knb_dev, d_Ic, nh, hd}. */
COLI_CUDA_DLLEXPORT int coli_cuda_prefill_dsa_select(int device,ColiCudaDsaChain *dsa,
        const float *xn_dev,const float *qres_dev,
        int S,int pos_base,int sB0,int D,int q_lora,int qk_rope,float theta);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_sync(int device);

#ifdef __cplusplus
}
#endif

#endif
