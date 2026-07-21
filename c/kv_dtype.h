#ifndef COLI_KV_DTYPE_H
#define COLI_KV_DTYPE_H

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nvfp4.h"

typedef enum {
    COLI_KV_FP32 = 0,
    COLI_KV_FP16 = 1,
    COLI_KV_FP8_E4M3 = 2
} ColiKVDtype;

/* Returns -1 for an invalid new setting. The legacy flag is consulted only
 * when COLI_KV_DTYPE is absent; default remains fp16. */
static inline int coli_kv_dtype_parse(const char *dtype, const char *legacy_f16) {
    if (dtype && *dtype) {
        if (!strcmp(dtype,"fp32")) return COLI_KV_FP32;
        if (!strcmp(dtype,"fp16")) return COLI_KV_FP16;
        if (!strcmp(dtype,"fp8")) return COLI_KV_FP8_E4M3;
        return -1;
    }
    return legacy_f16 ? (atoi(legacy_f16)!=0 ? COLI_KV_FP16 : COLI_KV_FP32)
                      : COLI_KV_FP16;
}

/* Reference encoder: nearest finite E4M3FN value, ties-to-even code. This is
 * intentionally scalar and deterministic; CUDA uses its native conversion. */
static inline uint8_t coli_f32_e4m3fn_ref(float value) {
    if (isnan(value)) return 0x7f;
    int negative=signbit(value)!=0;float magnitude=fabsf(value);
    if (isinf(magnitude)||magnitude>=448.0f)return (uint8_t)((negative?0x80:0)|0x7e);
    int best=0;float distance=INFINITY;
    for(int code=0;code<=0x7e;code++){
        float candidate=coli_e4m3fn_f32((uint8_t)code);
        if(candidate<0||isnan(candidate))continue;
        float d=fabsf(candidate-magnitude);
        if(d<distance||(d==distance&&((code&1)==0&&(best&1)))){distance=d;best=code;}
    }
    return (uint8_t)(best|(negative?0x80:0));
}

static inline float coli_kv_fp8_quantize_row(uint8_t *dst,const float *src,size_t n) {
    float amax=0.0f;
    for(size_t i=0;i<n;i++){float a=fabsf(src[i]);if(!isfinite(a))return NAN;if(a>amax)amax=a;}
    float scale=amax/448.0f;
    if(scale<FLT_MIN)scale=FLT_MIN;
    for(size_t i=0;i<n;i++)dst[i]=coli_f32_e4m3fn_ref(src[i]/scale);
    return scale;
}

static inline int coli_kv_fp8_dequantize_row(float *dst,const uint8_t *src,
                                              size_t n,float scale) {
    if(!isfinite(scale)||scale<=0)return 0;
    for(size_t i=0;i<n;i++){
        float value=coli_e4m3fn_f32(src[i]);if(!isfinite(value))return 0;
        dst[i]=value*scale;
    }
    return 1;
}

#endif
