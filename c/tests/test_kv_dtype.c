#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../kv_dtype.h"

int main(void){
    if(coli_kv_dtype_parse(NULL,NULL)!=COLI_KV_FP16||
       coli_kv_dtype_parse(NULL,"0")!=COLI_KV_FP32||
       coli_kv_dtype_parse("fp8","0")!=COLI_KV_FP8_E4M3||
       coli_kv_dtype_parse("bad",NULL)!=-1)return 1;
    float zero[17]={0},roundtrip[17];uint8_t encoded[17];
    float scale=coli_kv_fp8_quantize_row(encoded,zero,17);
    if(scale!=FLT_MIN||!coli_kv_fp8_dequantize_row(roundtrip,encoded,17,scale))return 2;
    for(int i=0;i<17;i++)if(roundtrip[i]!=0)return 3;
    float row[]={-448,-3,-1,-.1f,0,.1f,1,3,448};
    scale=coli_kv_fp8_quantize_row(encoded,row,9);
    if(fabsf(scale-1.0f)>1e-7f||!coli_kv_fp8_dequantize_row(roundtrip,encoded,9,scale))return 4;
    if(roundtrip[0]!=-448||roundtrip[8]!=448||roundtrip[4]!=0)return 5;
    float bad[]={NAN};if(!isnan(coli_kv_fp8_quantize_row(encoded,bad,1)))return 6;
    if(coli_kv_fp8_dequantize_row(roundtrip,encoded,1,0))return 7;
    /* Long-context shadow oracle: values and their per-row scales remain
     * finite across 4k/32k boundaries, including an exact rewind overwrite. */
    const int width=17,max_rows=32768;
    uint8_t *values=malloc((size_t)max_rows*width);float *scales=malloc((size_t)max_rows*4);
    float *src=malloc((size_t)width*4),*dst=malloc((size_t)width*4);
    if(!values||!scales||!src||!dst)return 8;
    for(int r=0;r<max_rows;r++){
        for(int i=0;i<width;i++)src[i]=sinf((float)(r*19+i)*0.013f)*(1.f+(r%31));
        scales[r]=coli_kv_fp8_quantize_row(values+(size_t)r*width,src,width);
        if(!isfinite(scales[r])||scales[r]<=0||
           !coli_kv_fp8_dequantize_row(dst,values+(size_t)r*width,width,scales[r]))return 9;
        float tol=scales[r]*32.f+1e-6f;
        for(int i=0;i<width;i++)if(!isfinite(dst[i])||fabsf(dst[i]-src[i])>tol)return 10;
    }
    int rewind=4096;
    for(int r=rewind;r<rewind+7;r++){
        for(int i=0;i<width;i++)src[i]=(float)(r-rewind+1)*(i-8);
        scales[r]=coli_kv_fp8_quantize_row(values+(size_t)r*width,src,width);
        if(!coli_kv_fp8_dequantize_row(dst,values+(size_t)r*width,width,scales[r]))return 11;
        for(int i=0;i<width;i++)if(!isfinite(dst[i]))return 12;
    }
    free(values);free(scales);free(src);free(dst);
    puts("kv dtype: ok");return 0;
}
