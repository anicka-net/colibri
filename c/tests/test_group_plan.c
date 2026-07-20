#include <assert.h>
#include <stdio.h>
#include "../group_plan.h"

static void verify(const int *rows,const int *dev,int n,int device,int cap){
    int ns=coli_group_plan(rows,dev,n,device,cap,NULL,0);
    assert(ns>=0);
    ColiGroupSeg seg[256];assert(ns<=(int)(sizeof(seg)/sizeof(*seg)));
    assert(coli_group_plan(rows,dev,n,device,cap,seg,256)==ns);
    int seen[64]={0};
    for(int i=0;i<ns;i++){
        assert(seg[i].group>=0&&seg[i].group<n);
        assert(dev[seg[i].group]==device);
        assert(seg[i].begin==seen[seg[i].group]);
        assert(seg[i].rows>0&&seg[i].rows<=cap);
        seen[seg[i].group]+=seg[i].rows;
    }
    for(int q=0;q<n;q++)assert(seen[q]==(dev[q]==device?rows[q]:0));
}

int main(void){
    int rows[]={0,1,16384,16385,262144,8192};
    int dev []={0,0,0,0,0,1};
    verify(rows,dev,6,0,16384);
    verify(rows,dev,6,1,16384);
    assert(coli_group_plan(rows,dev,6,0,0,NULL,0)<0);
    ColiGroupSeg short_out[1];
    assert(coli_group_plan(rows,dev,6,0,16384,short_out,1)<0);
    puts("OK bounded CUDA expert group planner");
    return 0;
}
