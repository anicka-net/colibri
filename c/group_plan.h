#ifndef COLI_GROUP_PLAN_H
#define COLI_GROUP_PLAN_H

typedef struct { int group, begin, rows; } ColiGroupSeg;

static int coli_group_plan(const int *rows,const int *devices,int count,int device,
                           int row_cap,ColiGroupSeg *out,int out_cap){
    if(!rows||!devices||count<0||row_cap<1)return -1;
    int n=0;
    for(int q=0;q<count;q++)if(devices[q]==device){
        if(rows[q]<0)return -1;
        for(int b=0;b<rows[q];b+=row_cap){
            if(out){
                if(n>=out_cap)return -1;
                int nr=rows[q]-b<row_cap?rows[q]-b:row_cap;
                out[n]=(ColiGroupSeg){q,b,nr};
            }
            n++;
        }
    }
    return n;
}

#endif
