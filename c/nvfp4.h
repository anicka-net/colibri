#ifndef COLI_NVFP4_H
#define COLI_NVFP4_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "tensor_format.h"

static inline float coli_bf16_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline float coli_e4m3fn_f32(uint8_t value) {
    int sign = value & 0x80 ? -1 : 1;
    int exp = (value >> 3) & 15, mant = value & 7;
    if (exp == 15 && mant == 7) return NAN;
    float magnitude = exp ? ldexpf(1.0f + (float)mant / 8.0f, exp - 7)
                          : ldexpf((float)mant, -9);
    return sign * magnitude;
}

static inline float coli_e2m1_f32(uint8_t code) {
    static const float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return lut[code & 15];
}

static inline size_t coli_nvfp4_cutlass_scale_bytes(int O, int I) {
    return (size_t)((O + 127) / 128) * (size_t)((I + 63) / 64) * 512;
}

static inline size_t coli_nvfp4_cutlass_scale_offset(int o, int group, int I) {
    int ktiles = (I + 63) / 64;
    int om = o / 128, oi = o % 128, kg = group / 4, gi = group % 4;
    return ((size_t)om * (size_t)ktiles + (size_t)kg) * 512
         + (size_t)(oi % 32) * 16 + (size_t)(oi / 32) * 4 + (size_t)gi;
}

static inline uint8_t coli_nvfp4_scale_raw(const uint8_t *scales,
                                            int layout, int o, int group,
                                            int I) {
    if (layout == COLI_SCALE_ROW_MAJOR_G16)
        return scales[(size_t)o * (size_t)((I + 15) / 16) + (size_t)group];
    if (layout == COLI_SCALE_CUTLASS_SM1XX_128X4)
        return scales[coli_nvfp4_cutlass_scale_offset(o, group, I)];
    return 0x7f; /* E4M3FN NaN: malformed layout propagates visibly. */
}

/* Correctness fallbacks/oracles. These intentionally accumulate in FP32. */
static inline void coli_matmul_bf16_ref(float *y, const float *x,
                                         const uint16_t *w, int S, int I, int O) {
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) {
            float acc = 0.0f;
            for (int i = 0; i < I; i++)
                acc += x[(size_t)s * I + i] * coli_bf16_f32(w[(size_t)o * I + i]);
            y[(size_t)s * O + o] = acc;
        }
}

static inline int coli_matmul_nvfp4_w4a32_ref(
    float *y, const float *x, const uint8_t *packed,
    const uint8_t *block_scales, float tensor_scale, int scale_layout,
    int S, int I, int O) {
    if (!y || !x || !packed || !block_scales || S < 0 || I <= 0 || O <= 0 ||
        !isfinite(tensor_scale) || tensor_scale <= 0.0f || tensor_scale >= 1.0f ||
        (scale_layout != COLI_SCALE_ROW_MAJOR_G16 &&
         scale_layout != COLI_SCALE_CUTLASS_SM1XX_128X4)) return 0;
    int rb = (I + 1) / 2;
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) {
            float acc = 0.0f;
            for (int i = 0; i < I; i++) {
                uint8_t byte = packed[(size_t)o * rb + (size_t)i / 2];
                uint8_t code = i & 1 ? byte >> 4 : byte & 15;
                float scale = coli_e4m3fn_f32(coli_nvfp4_scale_raw(
                    block_scales, scale_layout, o, i / COLI_NVFP4_GROUP_SIZE, I));
                if (!isfinite(scale) || scale <= 0.0f) return 0;
                acc += x[(size_t)s * I + i] * coli_e2m1_f32(code) * scale;
            }
            y[(size_t)s * O + o] = acc * tensor_scale;
        }
    return 1;
}

#endif
