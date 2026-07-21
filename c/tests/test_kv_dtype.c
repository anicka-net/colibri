#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

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
    puts("kv dtype: ok");return 0;
}
