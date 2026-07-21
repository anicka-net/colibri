#include "../nvfp4.h"
#include "../backend_cuda_nvfp4_cutlass.cuh"

#include <cstdio>

static int check(int m,int n,int k){
    auto a=coli_nvfp4_native::scale_layout(m,n,k);
    using Config=coli_nvfp4_native::ScaleConfig;
    auto b=Config::tile_atom_to_shape_SFB(cute::make_shape(m,n,k,1));
    if(coli_nvfp4_native::scale_bytes(m,n,k)!=(size_t)cute::size(cute::filter_zeros(a)))return 1;
    for(int row=0;row<n;row++)for(int group=0;group<(k+15)/16;group++){
        size_t got=(size_t)b(row,group*16,0);
        size_t want=coli_nvfp4_cutlass_scale_offset(row,group,k);
        if(got!=want){
            std::fprintf(stderr,"SFB offset mismatch shape (%d,%d,%d), row %d group %d: %zu != %zu\n",
                         m,n,k,row,group,got,want);return 2;
        }
    }
    /* SFA is the same atom over M rather than N. */
    for(int row=0;row<m;row++)for(int group=0;group<(k+15)/16;group++){
        size_t got=(size_t)a(row,group*16,0);
        size_t want=coli_nvfp4_cutlass_scale_offset(row,group,k);
        if(got!=want){
            std::fprintf(stderr,"SFA offset mismatch shape (%d,%d,%d), row %d group %d: %zu != %zu\n",
                         m,n,k,row,group,got,want);return 3;
        }
    }
    return 0;
}

int main(){
    const int shapes[][3]={{1,32,32},{17,131,65},{128,128,64},{129,257,129}};
    for(const auto &s:shapes){int rc=check(s[0],s[1],s[2]);if(rc)return rc;}
    std::puts("CUTLASS NVFP4 SFA/SFB layout parity: ok");return 0;
}
