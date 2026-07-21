#ifndef COLI_BACKEND_CUDA_NVFP4_CUTLASS_CUH
#define COLI_BACKEND_CUDA_NVFP4_CUTLASS_CUH

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace coli_nvfp4_native {
using namespace cute;
using ElementA=cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB=cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC=float; using ElementD=float; using Acc=float;
using Arch=cutlass::arch::Sm120;
using Op=cutlass::arch::OpClassBlockScaledTensorOp;
using Tile=Shape<_128,_128,_128>; using Cluster=Shape<_1,_1,_1>;
using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch,Op,Tile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,
    Acc,Acc,ElementC,cutlass::layout::RowMajor,4,
    ElementD,cutlass::layout::RowMajor,4,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
    Arch,Op,ElementA,cutlass::layout::RowMajor,32,
    ElementB,cutlass::layout::ColumnMajor,32,Acc,Tile,Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename Epilogue::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue,void>;
using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using StrideA=typename Kernel::StrideA; using StrideB=typename Kernel::StrideB;
using StrideC=typename Kernel::StrideC; using StrideD=typename Kernel::StrideD;
using Scale=typename ElementA::ScaleFactorType;
using ScaleConfig=typename Mainloop::Sm1xxBlkScaledConfig;

template<class Layout>
__global__ void quantize(uint8_t *packed,Scale *sf,const float *x,Layout layout,float input_scale,int M,int N,int K){
    int m=blockIdx.x,g=blockIdx.y;if(m>=M||g*16>=K)return;
    __shared__ float red[32];float a=0.f;int i=g*16+threadIdx.x;
    if(threadIdx.x<16&&i<K)a=fabsf(x[(size_t)m*K+i]);red[threadIdx.x]=a;__syncthreads();
    for(int n=8;n;n>>=1){if(threadIdx.x<n)red[threadIdx.x]=fmaxf(red[threadIdx.x],red[threadIdx.x+n]);__syncthreads();}
    float scale=red[0]>0.f?red[0]/(6.f*input_scale):1.f;
    if(!threadIdx.x)sf[layout(m,g*16,0)]=Scale(scale);
    if(threadIdx.x<8){int i0=g*16+threadIdx.x*2;uint8_t lo=0,hi=0;
        const float levels[8]={0,.5f,1,1.5f,2,3,4,6};
        if(i0<K){float v=x[(size_t)m*K+i0]/(scale*input_scale),a=fabsf(v);int q=0;for(int z=1;z<8;z++)if(fabsf(a-levels[z])<fabsf(a-levels[q]))q=z;lo=q|(signbit(v)?8:0);}
        if(i0+1<K){float v=x[(size_t)m*K+i0+1]/(scale*input_scale),a=fabsf(v);int q=0;for(int z=1;z<8;z++)if(fabsf(a-levels[z])<fabsf(a-levels[q]))q=z;hi=q|(signbit(v)?8:0);}
        packed[(size_t)m*((K+1)/2)+i0/2]=(uint8_t)(lo|(hi<<4));}
}

static size_t activation_bytes(int M,int K){return (size_t)M*((K+1)/2);}
static auto scale_layout(int M,int N,int K){return ScaleConfig::tile_atom_to_shape_SFA(make_shape(M,N,K,1));}
static size_t scale_bytes(int M,int N,int K){return (size_t)size(filter_zeros(scale_layout(M,N,K)))*sizeof(Scale);}

static int run(float *d,const float *x,uint8_t *qa,void *sfa,
               const uint8_t *b,const uint8_t *sfb,float alpha,float input_scale,
               int M,int N,int K,cudaStream_t stream,int *reason){
    if((K&31)||(N&7)){if(reason)*reason=1;return 0;}
    auto sa=cutlass::make_cute_packed_stride(StrideA{},make_shape(M,K,1));
    auto sb=cutlass::make_cute_packed_stride(StrideB{},make_shape(N,K,1));
    auto sc=cutlass::make_cute_packed_stride(StrideC{},make_shape(M,N,1));
    auto sd=cutlass::make_cute_packed_stride(StrideD{},make_shape(M,N,1));
    auto la=scale_layout(M,N,K);auto lb=ScaleConfig::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
    quantize<<<dim3(M,(K+15)/16),32,0,stream>>>(qa,(Scale*)sfa,x,la,input_scale,M,N,K);
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},
        {reinterpret_cast<typename ElementA::DataType*>(qa),sa,
         reinterpret_cast<typename ElementB::DataType*>(const_cast<uint8_t*>(b)),sb,
         (Scale*)sfa,la,(Scale*)const_cast<uint8_t*>(sfb),lb},
        {{alpha*input_scale,0.f},d,sc,d,sd}};
    Gemm gemm;auto st=gemm.can_implement(args);if(st!=cutlass::Status::kSuccess){if(reason)*reason=2;return 0;}
    size_t ws=Gemm::get_workspace_size(args);void *workspace=nullptr;
    if(ws&&cudaMalloc(&workspace,ws)!=cudaSuccess){if(reason)*reason=3;return 0;}
    st=gemm.initialize(args,workspace,stream);if(st==cutlass::Status::kSuccess)st=gemm.run(stream);
    if(workspace)cudaFree(workspace);if(st!=cutlass::Status::kSuccess){if(reason)*reason=4;return 0;}
    return 1;
}
}
#endif
