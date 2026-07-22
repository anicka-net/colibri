#ifndef COLI_BACKEND_CUDA_NVFP4_CUTLASS_CUH
#define COLI_BACKEND_CUDA_NVFP4_CUTLASS_CUH

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <vector>

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

/* Pointer-array grouped GEMM. The descriptor storage is deliberately local to
 * this adapter: cudaMallocAsync/cudaFreeAsync preserve stream ordering, while
 * the immutable expert weight and scale allocations remain independently
 * addressable. */
using GroupShape=cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
using GroupEpilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch,Op,Tile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,
    Acc,Acc,ElementC,cutlass::layout::RowMajor*,4,
    ElementD,cutlass::layout::RowMajor*,4,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using GroupMainloop=typename cutlass::gemm::collective::CollectiveBuilder<
    Arch,Op,ElementA,cutlass::layout::RowMajor*,32,
    ElementB,cutlass::layout::ColumnMajor*,32,Acc,Tile,Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<(int)sizeof(typename GroupEpilogue::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GroupKernel=cutlass::gemm::kernel::GemmUniversal<GroupShape,GroupMainloop,GroupEpilogue>;
using GroupGemm=cutlass::gemm::device::GemmUniversalAdapter<GroupKernel>;
using GroupStrideA=typename GroupKernel::InternalStrideA;
using GroupStrideB=typename GroupKernel::InternalStrideB;
using GroupStrideC=typename GroupKernel::InternalStrideC;
using GroupStrideD=typename GroupKernel::InternalStrideD;
using GroupLayoutSFA=typename GroupMainloop::InternalLayoutSFA;
using GroupLayoutSFB=typename GroupMainloop::InternalLayoutSFB;

static int run_group(float *d,const float *x,uint8_t *qa,void *sfa,
        const uint8_t *const *b,const uint8_t *const *sfb,const float *alpha,
        const int *rows,const int *offsets,int groups,int N,int K,
        const float *input_scales,cudaStream_t stream,int *reason){
    if(groups<1||(K&31)||(N&7)){if(reason)*reason=1;return 0;}
    using Shape3=typename GroupShape::UnderlyingProblemShape;
    std::vector<Shape3> shapes;std::vector<const typename ElementA::DataType*> ap;
    std::vector<const typename ElementB::DataType*> bp;std::vector<const Scale*> sap,sbp;
    std::vector<const float*> cp;std::vector<float*> dp;
    std::vector<GroupStrideA> as;std::vector<GroupStrideB> bs;
    std::vector<GroupStrideC> cs;std::vector<GroupStrideD> ds;
    std::vector<GroupLayoutSFA> sal;std::vector<GroupLayoutSFB> sbl;
    size_t qoff=0,soff=0;
    for(int g=0;g<groups;g++){
        int M=rows[g];auto la=scale_layout(M,N,K);
        shapes.push_back({M,N,K});
        ap.push_back(reinterpret_cast<typename ElementA::DataType*>(qa+qoff));
        bp.push_back(reinterpret_cast<const typename ElementB::DataType*>(b[g]));
        sap.push_back((Scale*)((uint8_t*)sfa+soff));sbp.push_back((const Scale*)sfb[g]);
        cp.push_back(d+(size_t)offsets[g]*N);dp.push_back(d+(size_t)offsets[g]*N);
        as.push_back(cutlass::make_cute_packed_stride(GroupStrideA{},make_shape(M,K,1)));
        bs.push_back(cutlass::make_cute_packed_stride(GroupStrideB{},make_shape(N,K,1)));
        cs.push_back(cutlass::make_cute_packed_stride(GroupStrideC{},make_shape(M,N,1)));
        ds.push_back(cutlass::make_cute_packed_stride(GroupStrideD{},make_shape(M,N,1)));
        sal.push_back(la);sbl.push_back(ScaleConfig::tile_atom_to_shape_SFB(make_shape(M,N,K,1)));
        quantize<<<dim3(M,(K+15)/16),32,0,stream>>>(qa+qoff,(Scale*)((uint8_t*)sfa+soff),
            x+(size_t)offsets[g]*K,la,input_scales[g],M,N,K);
        qoff+=activation_bytes(M,K);soff+=scale_bytes(M,N,K);
    }
#define COLI_GCOPY(name,vec) decltype(vec)::value_type *name=nullptr; \
    if(cudaMallocAsync(&name,sizeof(*name)*(vec).size(),stream)!=cudaSuccess)return 0; \
    cudaMemcpyAsync(name,(vec).data(),sizeof(*name)*(vec).size(),cudaMemcpyHostToDevice,stream)
    COLI_GCOPY(d_shapes,shapes);COLI_GCOPY(d_ap,ap);COLI_GCOPY(d_bp,bp);
    COLI_GCOPY(d_sap,sap);COLI_GCOPY(d_sbp,sbp);COLI_GCOPY(d_cp,cp);COLI_GCOPY(d_dp,dp);
    COLI_GCOPY(d_as,as);COLI_GCOPY(d_bs,bs);COLI_GCOPY(d_cs,cs);COLI_GCOPY(d_ds,ds);
    COLI_GCOPY(d_sal,sal);COLI_GCOPY(d_sbl,sbl);
    float *d_av=nullptr,**d_alpha=nullptr;std::vector<float*> alpha_ptr(groups);
    cudaMallocAsync(&d_av,sizeof(float)*groups,stream);cudaMemcpyAsync(d_av,alpha,sizeof(float)*groups,cudaMemcpyHostToDevice,stream);
    for(int g=0;g<groups;g++)alpha_ptr[g]=d_av+g;
    cudaMallocAsync(&d_alpha,sizeof(float*)*groups,stream);cudaMemcpyAsync(d_alpha,alpha_ptr.data(),sizeof(float*)*groups,cudaMemcpyHostToDevice,stream);
    decltype(typename GroupGemm::Arguments{}.epilogue.thread) fusion{};
    fusion.alpha=0;fusion.alpha_ptr_array=d_alpha;fusion.dAlpha={_0{},_0{},1};
    fusion.beta=0;fusion.beta_ptr_array=nullptr;fusion.dBeta={_0{},_0{},0};
    cutlass::KernelHardwareInfo hw{};hw.device_id=0;hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename GroupGemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
        {groups,d_shapes,shapes.data()},{d_ap,d_as,d_bp,d_bs,d_sap,d_sal,d_sbp,d_sbl},
        {fusion,d_cp,d_cs,d_dp,d_ds},hw};
    GroupGemm gemm;auto st=gemm.can_implement(args);if(st!=cutlass::Status::kSuccess){if(reason)*reason=2;return 0;}
    size_t wsz=GroupGemm::get_workspace_size(args);void *ws=nullptr;if(wsz)cudaMallocAsync(&ws,wsz,stream);
    st=gemm.initialize(args,ws,stream);if(st==cutlass::Status::kSuccess)st=gemm.run(stream);
    if(ws)cudaFreeAsync(ws,stream);
    cudaFreeAsync(d_shapes,stream);cudaFreeAsync(d_ap,stream);cudaFreeAsync(d_bp,stream);
    cudaFreeAsync(d_sap,stream);cudaFreeAsync(d_sbp,stream);cudaFreeAsync(d_cp,stream);cudaFreeAsync(d_dp,stream);
    cudaFreeAsync(d_as,stream);cudaFreeAsync(d_bs,stream);cudaFreeAsync(d_cs,stream);cudaFreeAsync(d_ds,stream);
    cudaFreeAsync(d_sal,stream);cudaFreeAsync(d_sbl,stream);cudaFreeAsync(d_av,stream);cudaFreeAsync(d_alpha,stream);
#undef COLI_GCOPY
    if(st!=cutlass::Status::kSuccess){if(reason)*reason=4;return 0;}return 1;
}
}
#endif
