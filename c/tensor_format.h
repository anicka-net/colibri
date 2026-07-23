#ifndef COLI_TENSOR_FORMAT_H
#define COLI_TENSOR_FORMAT_H

/* Stable values are part of the C/CUDA/Windows ABI and snapshot manifest. */
typedef enum {
    COLI_TENSOR_F32 = 0,
    COLI_TENSOR_INT8_ROW = 1,
    COLI_TENSOR_INT4_ROW = 2,
    COLI_TENSOR_INT2_ROW = 3,
    COLI_TENSOR_INT4_GROUP = 4,
    COLI_TENSOR_BF16 = 5,
    COLI_TENSOR_MODELOPT_NVFP4 = 6,
    COLI_TENSOR_INT3_GROUP = 7,
    COLI_TENSOR_E8 = 8
} ColiTensorFormat;

typedef enum {
    COLI_SCALE_NONE = 0,
    COLI_SCALE_ROW_MAJOR_G16 = 1,
    COLI_SCALE_CUTLASS_SM1XX_128X4 = 2
} ColiScaleLayout;

#define COLI_NVFP4_GROUP_SIZE 16

#endif
