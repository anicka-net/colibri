#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../nvfp4.h"

static int near(float a, float b) { return fabsf(a - b) < 1e-6f; }

int main(void) {
    for (int i = 0; i < 16; i++) {
        static const float expected[16] = {
            0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6
        };
        if (!near(coli_e2m1_f32((uint8_t)i), expected[i])) return 1;
    }
    if (!near(coli_e4m3fn_f32(0x01), ldexpf(1.0f,-9)) ||
        !near(coli_e4m3fn_f32(0x7e), 448.0f) ||
        !isnan(coli_e4m3fn_f32(0x7f))) return 2;
    if (!coli_e4m3fn_raw_is_finite(0x00) ||
        !coli_e4m3fn_raw_is_finite(0x7e) ||
        coli_e4m3fn_raw_is_finite(0x7f) ||
        coli_e4m3fn_raw_is_finite(0xff) ||
        coli_e4m3fn_raw_is_positive(0x00) ||
        !coli_e4m3fn_raw_is_positive(0x01) ||
        !coli_e4m3fn_raw_is_positive(0x7e) ||
        coli_e4m3fn_raw_is_positive(0x81)) return 15;

    uint16_t bf16[] = {0x3f80, 0x4000, 0x4040, 0x4080};
    float bx[] = {2, -1}, by[2];
    coli_matmul_bf16_ref(by, bx, bf16, 1, 2, 2);
    if (!near(by[0], 0.0f) || !near(by[1], 2.0f)) return 3;

    enum { O = 131, I = 17, RB = 9, GROUPS = 2 };
    uint8_t packed[O * RB], row_scales[O * GROUPS];
    uint8_t cutlass_scales[1024];
    float x[I], row_y[O], cutlass_y[O], w4a4_y[O];
    memset(packed, 0, sizeof(packed));
    memset(cutlass_scales, 0, sizeof(cutlass_scales));
    for (int i = 0; i < I; i++) x[i] = (float)(i - 8) / 8.0f;
    for (int o = 0; o < O; o++) {
        for (int i = 0; i < I; i++) {
            uint8_t code = (uint8_t)((o + i) & 15);
            packed[o * RB + i / 2] |= (uint8_t)(code << ((i & 1) * 4));
        }
        row_scales[o * GROUPS] = 0x38;     /* 1 */
        row_scales[o * GROUPS + 1] = 0x40; /* 2 */
        for (int g = 0; g < GROUPS; g++)
            cutlass_scales[coli_nvfp4_cutlass_scale_offset(o,g,I)] = row_scales[o*GROUPS+g];
    }
    if (coli_nvfp4_cutlass_scale_bytes(O,I) != 1024) return 4;
    if (!coli_matmul_nvfp4_w4a32_ref(row_y,x,packed,row_scales,.25f,
                                     COLI_SCALE_ROW_MAJOR_G16,1,I,O) ||
        !coli_matmul_nvfp4_w4a32_ref(cutlass_y,x,packed,cutlass_scales,.25f,
                                     COLI_SCALE_CUTLASS_SM1XX_128X4,1,I,O)) return 5;
    for (int o = 0; o < O; o++) if (!near(row_y[o], cutlass_y[o])) return 6;
    if (!coli_matmul_nvfp4_w4a4_ref(w4a4_y,x,packed,cutlass_scales,.25f,.5f,
                                     COLI_SCALE_CUTLASS_SM1XX_128X4,1,I,O)) return 7;
    for(int o=0;o<O;o++)if(!isfinite(w4a4_y[o]))return 8;
    row_scales[0] = 0;
    if (coli_matmul_nvfp4_w4a32_ref(row_y,x,packed,row_scales,.25f,
                                    COLI_SCALE_ROW_MAJOR_G16,1,I,O)) return 9;
    puts("nvfp4 reference: ok");
    return 0;
}
