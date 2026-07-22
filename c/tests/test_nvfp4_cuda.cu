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

static void fill_tensor(std::vector<uint8_t>& w,std::vector<uint8_t>& sf,int I,int O){
    int rb=(I+1)/2,groups=(I+15)/16;
    for(int o=0;o<O;o++){
        for(int b=0;b<rb;b++)w[(size_t)o*rb+b]=(uint8_t)(((o+b+1)&15)|(((o*3+b*5+7)&15)<<4));
        for(int g=0;g<groups;g++)sf[coli_nvfp4_cutlass_scale_offset(o,g,I)]=0x38;
    }
}

static int run_group_shape(int device,int D,int I){
    std::vector<uint8_t> wi((size_t)I*((D+1)/2)),si(coli_nvfp4_cutlass_scale_bytes(I,D),0);
    std::vector<uint8_t> wd((size_t)D*((I+1)/2)),sd(coli_nvfp4_cutlass_scale_bytes(D,I),0);
    fill_tensor(wi,si,D,I);fill_tensor(wd,sd,I,D);
    ColiCudaTensor *g=nullptr,*u=nullptr,*d=nullptr;
    if(!coli_cuda_tensor_upload_nvfp4(&g,wi.data(),si.data(),.25f,.01f,COLI_SCALE_CUTLASS_SM1XX_128X4,D,I,device)||
       !coli_cuda_tensor_upload_nvfp4(&u,wi.data(),si.data(),.25f,.01f,COLI_SCALE_CUTLASS_SM1XX_128X4,D,I,device)||
       !coli_cuda_tensor_upload_nvfp4(&d,wd.data(),sd.data(),.25f,.01f,COLI_SCALE_CUTLASS_SM1XX_128X4,I,D,device))return 1;
    ColiCudaTensor *gs[2]={g,g},*us[2]={u,u},*ds[2]={d,d};
    for(int S: {1,2,8,16,64}){
        int rows[2]={S,S};std::vector<float> x((size_t)2*S*D),got((size_t)2*S*D),ref((size_t)2*S*D);
        for(size_t z=0;z<x.size();z++)x[z]=std::sin((float)(z+1)*.013f)*2.f;
        if(!coli_cuda_expert_mlp(g,u,d,ref.data(),x.data(),S)||
           !coli_cuda_expert_mlp(g,u,d,ref.data()+(size_t)S*D,x.data()+(size_t)S*D,S)||
           !coli_cuda_expert_group(gs,us,ds,rows,2,got.data(),x.data(),nullptr,nullptr,0,0,0,nullptr))return 2;
        double err=0,base=0;for(size_t z=0;z<got.size();z++){double q=got[z]-ref[z];err+=q*q;base+=(double)ref[z]*ref[z];}
        if(std::sqrt(err/(base+1e-30))>3e-3)return 3;
    }
    coli_cuda_tensor_free(g);coli_cuda_tensor_free(u);coli_cuda_tensor_free(d);return 0;
}

int main(int argc,char **argv){
    int device=argc>1?std::atoi(argv[1]):0;
    if(!coli_cuda_init(&device,1))return 77;
    if(!coli_cuda_nvfp4_native_capable(device))return 77;
    int rc=run_shape(device,6144,2048);if(!rc)rc=run_shape(device,2048,6144);
    if(!rc)rc=run_group_shape(device,6144,2048);
    uint64_t native=0,generic=0,unavailable=0,failures=0;
    coli_cuda_nvfp4_stats(&native,&generic,&unavailable,&failures);
    uint64_t grouped=0,problems=0,group_fallbacks=0;
    coli_cuda_nvfp4_grouped_stats(&grouped,&problems,&group_fallbacks);
    if(!rc&&(native!=70||generic||unavailable||failures||grouped!=15||problems!=30||group_fallbacks))rc=4;
    coli_cuda_shutdown();
    if(!rc)std::puts("GB10 CUTLASS NVFP4 GLM-shape oracle: ok");
    return rc;
}
