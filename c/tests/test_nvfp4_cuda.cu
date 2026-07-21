#include "../backend_cuda.h"
#include "../nvfp4.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int run_shape(int device,int I,int O){
    const int rb=(I+1)/2,groups=(I+15)/16,sample_o=16;
    std::vector<uint8_t> w((size_t)O*rb),sf(coli_nvfp4_cutlass_scale_bytes(O,I),0);
    std::vector<uint8_t> sample_w((size_t)sample_o*rb),sample_sf((size_t)sample_o*groups,0x38);
    for(int o=0;o<O;o++){
        for(int b=0;b<rb;b++)w[(size_t)o*rb+b]=(uint8_t)(((o+b+1)&15)|(((o*3+b*5+7)&15)<<4));
        for(int g=0;g<groups;g++)sf[coli_nvfp4_cutlass_scale_offset(o,g,I)]=0x38;
    }
    for(int o=0;o<sample_o;o++)
        std::memcpy(sample_w.data()+(size_t)o*rb,w.data()+(size_t)o*rb,rb);
    ColiCudaTensor *tensor=nullptr;
    for(int S: {1,2,8,16,64}){
        std::vector<float> x((size_t)S*I),got((size_t)S*O),ref((size_t)S*sample_o);
        for(size_t i=0;i<x.size();i++)x[i]=std::sin((float)(i+1)*.013f)*2.f;
        if(!coli_cuda_matmul_nvfp4(&tensor,got.data(),x.data(),w.data(),sf.data(),.25f,.01f,
              COLI_SCALE_CUTLASS_SM1XX_128X4,S,I,O,device))return 1;
        if(!coli_matmul_nvfp4_w4a4_ref(ref.data(),x.data(),sample_w.data(),sample_sf.data(),
              .25f,.01f,COLI_SCALE_ROW_MAJOR_G16,S,I,sample_o))return 2;
        double err=0,base=0;
        for(int s=0;s<S;s++)for(int o=0;o<sample_o;o++){
            double d=got[(size_t)s*O+o]-ref[(size_t)s*sample_o+o];err+=d*d;
            base+=(double)ref[(size_t)s*sample_o+o]*ref[(size_t)s*sample_o+o];
        }
        double rms=std::sqrt(err/(base+1e-30));
        if(rms>3e-3){std::fprintf(stderr,"NVFP4 (%d,%d) S=%d RMS %.6g\n",O,I,S,rms);return 3;}
    }
    coli_cuda_tensor_free(tensor);return 0;
}

int main(int argc,char **argv){
    int device=argc>1?std::atoi(argv[1]):0;
    if(!coli_cuda_init(&device,1))return 77;
    if(!coli_cuda_nvfp4_native_capable(device))return 77;
    int rc=run_shape(device,6144,2048);if(!rc)rc=run_shape(device,2048,6144);
    uint64_t native=0,generic=0,unavailable=0,failures=0;
    coli_cuda_nvfp4_stats(&native,&generic,&unavailable,&failures);
    if(!rc&&(native!=10||generic||unavailable||failures))rc=4;
    coli_cuda_shutdown();
    if(!rc)std::puts("GB10 CUTLASS NVFP4 GLM-shape oracle: ok");
    return rc;
}
