#include "../backend_cuda.h"
#include "../nvfp4.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

static double bench(ColiCudaTensor **tensor,const std::vector<float>& x,
        std::vector<float>& y,const std::vector<uint8_t>& w,
        const std::vector<uint8_t>& sf,int S,int I,int O,int device,int native){
    char value[2]={(char)('0'+native),0};
    setenv("COLI_NVFP4_NATIVE",value,1);
    int reps=S<=2?20:S<=16?8:3;
    for(int i=0;i<2;i++)
        if(!coli_cuda_matmul_nvfp4(tensor,y.data(),x.data(),w.data(),sf.data(),
                .25f,.01f,COLI_SCALE_CUTLASS_SM1XX_128X4,S,I,O,device))return -1;
    auto begin=std::chrono::steady_clock::now();
    for(int i=0;i<reps;i++)
        if(!coli_cuda_matmul_nvfp4(tensor,y.data(),x.data(),w.data(),sf.data(),
                .25f,.01f,COLI_SCALE_CUTLASS_SM1XX_128X4,S,I,O,device))return -1;
    auto end=std::chrono::steady_clock::now();
    return std::chrono::duration<double,std::milli>(end-begin).count()/reps;
}

static double time_one(ColiCudaTensor **tensor,const std::vector<float>& x,
        std::vector<float>& y,const std::vector<uint8_t>& w,
        const std::vector<uint8_t>& sf,int S,int I,int O,int device){
    setenv("COLI_NVFP4_NATIVE","1",1);
    auto begin=std::chrono::steady_clock::now();
    if(!coli_cuda_matmul_nvfp4(tensor,y.data(),x.data(),w.data(),sf.data(),
            .25f,.01f,COLI_SCALE_CUTLASS_SM1XX_128X4,S,I,O,device))return -1;
    auto end=std::chrono::steady_clock::now();
    return std::chrono::duration<double,std::milli>(end-begin).count();
}

static int run_shape(int device,int I,int O){
    int rb=(I+1)/2,groups=(I+15)/16;
    std::vector<uint8_t> w((size_t)O*rb),sf(coli_nvfp4_cutlass_scale_bytes(O,I),0);
    for(int o=0;o<O;o++){
        for(int b=0;b<rb;b++)w[(size_t)o*rb+b]=(uint8_t)(((o+b+1)&15)|(((o*3+b*5+7)&15)<<4));
        for(int g=0;g<groups;g++)sf[coli_nvfp4_cutlass_scale_offset(o,g,I)]=0x38;
    }
    ColiCudaTensor *tensor=nullptr;
    for(int S: {1,2,4,8,16,32,64}){
        std::vector<float> x((size_t)S*I,.125f),y((size_t)S*O);
        double generic=bench(&tensor,x,y,w,sf,S,I,O,device,0);
        double native=bench(&tensor,x,y,w,sf,S,I,O,device,1);
        if(generic<0||native<0){coli_cuda_tensor_free(tensor);return 1;}
        std::printf("I=%d O=%d S=%d generic_ms=%.3f native_ms=%.3f speedup=%.2fx\n",
                    I,O,S,generic,native,generic/native);
    }
    /* The streaming engine executes immutable pageable slabs directly on
     * unified-memory GB10. Measure that storage class separately from the
     * device-resident tensor above; kernel timing alone can otherwise hide
     * host-page mapping/coherency costs. Use one projection (not an MLP,
     * whose gate/up/down shapes differ) through the ordinary matmul entry. */
    {
        int S=8;std::vector<float> x((size_t)S*I,.125f),y((size_t)S*O);
        ColiCudaTensor *pageable=nullptr;
        if(!coli_cuda_tensor_wrap_host_nvfp4(&pageable,w.data(),sf.data(),.25f,.01f,
                COLI_SCALE_CUTLASS_SM1XX_128X4,I,O,device))return 1;
        double page_cold=time_one(&pageable,x,y,w,sf,S,I,O,device);
        double page_ms=bench(&pageable,x,y,w,sf,S,I,O,device,1);
        coli_cuda_tensor_free(pageable);
        cudaError_t wr=cudaHostRegister(w.data(),w.size(),cudaHostRegisterDefault);
        cudaError_t sr=cudaHostRegister(sf.data(),sf.size(),cudaHostRegisterDefault);
        if(wr==cudaSuccess&&sr==cudaSuccess){
            ColiCudaTensor *registered=nullptr;
            if(!coli_cuda_tensor_wrap_host_nvfp4(&registered,w.data(),sf.data(),.25f,.01f,
                    COLI_SCALE_CUTLASS_SM1XX_128X4,I,O,device))return 1;
            double registered_cold=time_one(&registered,x,y,w,sf,S,I,O,device);
            double registered_ms=bench(&registered,x,y,w,sf,S,I,O,device,1);
            std::printf("I=%d O=%d S=%d host_pageable_cold_ms=%.3f steady_ms=%.3f "
                        "host_registered_cold_ms=%.3f steady_ms=%.3f cold_speedup=%.2fx\n",
                        I,O,S,page_cold,page_ms,registered_cold,registered_ms,
                        page_cold/registered_cold);
            coli_cuda_tensor_free(registered);
        }else{
            std::printf("I=%d O=%d S=%d host_pageable_ms=%.3f host_register=unavailable\n",
                        I,O,S,page_ms);
        }
        if(sr==cudaSuccess)cudaHostUnregister(sf.data());
        if(wr==cudaSuccess)cudaHostUnregister(w.data());
    }
    coli_cuda_tensor_free(tensor);return 0;
}

int main(int argc,char **argv){
    int device=argc>1?std::atoi(argv[1]):0;
    if(!coli_cuda_init(&device,1))return 77;
    if(!coli_cuda_nvfp4_native_capable(device))return 77;
    int rc=run_shape(device,6144,2048);if(!rc)rc=run_shape(device,2048,6144);
    coli_cuda_shutdown();return rc;
}
