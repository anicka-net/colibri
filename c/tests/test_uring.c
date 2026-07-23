#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int fail(const char *s){ fprintf(stderr,"FAIL: %s\n",s); return 1; }

static int test_expert_layout(int fd){
    Model m={0}; ESlot slot={0}; UringBatch batch={0};
    m.c.hidden=4; m.c.moe_inter=3; m.ebits=8;
    m.S.n=6; m.S.cap=6; m.S.t=calloc(6,sizeof(st_tensor));
    if(!m.S.t) return fail("tensor metadata allocation");
    const char *proj[3]={"gate_proj","up_proj","down_proj"};
    int wbytes[3]={12,12,12}, sbytes[3]={12,12,16};
    unsigned char data[76];
    for(int i=0;i<36;i++) data[i]=(unsigned char)(i+1);
    float scales[10]; for(int i=0;i<10;i++) scales[i]=(float)i+0.5f;
    memcpy(data+36,scales,sizeof(scales));
    if(pwrite(fd,data,sizeof(data),0)!=(ssize_t)sizeof(data)){ free(m.S.t); return fail("expert fixture write"); }
    int64_t wo=0,so=36;
    for(int k=0;k<3;k++){
        char name[300];
        snprintf(name,sizeof(name),"model.layers.1.mlp.experts.7.%s.weight",proj[k]);
        m.S.t[k]=(st_tensor){strdup(name),fd,wo,wbytes[k],3,wbytes[k]}; wo+=wbytes[k];
        size_t n=strlen(name); memcpy(name+n,".qs",4);
        m.S.t[3+k]=(st_tensor){strdup(name),fd,so,sbytes[k],2,sbytes[k]/4}; so+=sbytes[k];
    }
    if(uring_batch_init(&batch)){ free(m.S.t); return fail("expert ring init"); }
    uring_batch_reset(&batch);
    int li=uring_load_add(&batch,&m,1,7,&slot,1);
    if(li!=0 || uring_submit_batch(&batch) || uring_finalize_load(&batch,li,1)){
        coli_uring_close(&batch.ring); free(m.S.t); return fail("expert batch load");
    }
    int bad=slot.eid!=7 || slot.g.fmt!=1 || slot.u.fmt!=1 || slot.d.fmt!=1
        || memcmp(slot.g.q8,data,12) || memcmp(slot.u.q8,data+12,12) || memcmp(slot.d.q8,data+24,12)
        || memcmp(slot.g.s,scales,12) || memcmp(slot.u.s,scales+3,12) || memcmp(slot.d.s,scales+6,16);
    coli_uring_close(&batch.ring);
    compat_aligned_free(slot.slab); free(slot.fslab);
    if(bad){ for(int i=0;i<m.S.n;i++) free(m.S.t[i].name); free(m.S.t); return fail("expert tensor views"); }

    m.c.n_experts=8; m.c.n_layers=2; m.ecap=2;
    m.pin=calloc(3,sizeof(ESlot*)); m.npin=calloc(3,sizeof(int));
    m.ecache=calloc(3,sizeof(ESlot*)); m.ecn=calloc(3,sizeof(int));
    m.ecache[1]=calloc(2,sizeof(ESlot));
    if(!m.pin||!m.npin||!m.ecache||!m.ecn||!m.ecache[1])
        return fail("pilot fixture allocation");
    if(uring_batch_init(&g_ub_pilot)) return fail("pilot ring init");
    memset(g_pilot_inflight,0,sizeof(g_pilot_inflight));
    atomic_store(&g_cur_moe_layer,-1); atomic_store(&g_pilot_loads,0); atomic_store(&g_pilot_drops,0);
    pilot_r=0; pilot_w=1; pilot_q[0].l=1; pilot_q[0].e=7;
    pilot_uring_batch(&m);
    bad=m.ecn[1]!=1 || m.ecache[1][0].eid!=7 || g_pilot_inflight[1]!=0
        || atomic_load(&g_pilot_loads)!=1 || atomic_load(&g_pilot_drops)!=0;
    coli_uring_close(&g_ub_pilot.ring);
    compat_aligned_free(m.ecache[1][0].slab); free(m.ecache[1][0].fslab);
    free(m.ecache[1]);
    free(m.pin); free(m.npin); free(m.ecache); free(m.ecn);
    for(int i=0;i<m.S.n;i++) free(m.S.t[i].name);
    free(m.S.t);
    return bad?fail("pilot uring publication"):0;
}

static int test_native_component_layout(int fd){
    Model m={0};ESlot slot={0};UringBatch batch={0};
    m.c.hidden=4;m.c.moe_inter=3;m.manifest.present=1;m.manifest.component_alignment=16;
    m.S.n=12;m.S.cap=12;m.S.t=calloc(12,sizeof(st_tensor));
    if(!m.S.t)return fail("native metadata allocation");
    const char *proj[3]={"gate_proj","up_proj","down_proj"};
    int O[3]={3,3,4},I[3]={4,4,3};int n=0;off_t cursor=4096;
    for(int k=0;k<3;k++){
        char base[300],name[340];snprintf(base,sizeof(base),
            "model.layers.1.mlp.experts.7.%s.weight",proj[k]);
        int64_t wb=(int64_t)O[k]*((I[k]+1)/2);
        int64_t sb=(int64_t)coli_nvfp4_cutlass_scale_bytes(O[k],I[k]);
        cursor=(cursor+15)&~15;unsigned char *w=malloc((size_t)wb);memset(w,0x11+k,(size_t)wb);
        if(pwrite(fd,w,(size_t)wb,cursor)!=wb)return fail("native weight fixture");
        m.S.t[n++]=(st_tensor){strdup(base),fd,cursor,wb,3,wb};free(w);cursor+=wb;
        cursor=(cursor+15)&~15;unsigned char *s=malloc((size_t)sb);memset(s,0x38,(size_t)sb);
        snprintf(name,sizeof(name),"%s.nvfp4_scale",base);
        if(pwrite(fd,s,(size_t)sb,cursor)!=sb)return fail("native scale fixture");
        m.S.t[n++]=(st_tensor){strdup(name),fd,cursor,sb,3,sb};free(s);cursor+=sb;
        float ts=.5f,is=1.25f;
        cursor=(cursor+15)&~15;snprintf(name,sizeof(name),"%s.nvfp4_tensor_scale",base);
        if(pwrite(fd,&ts,4,cursor)!=4)return fail("native tensor scale fixture");
        m.S.t[n++]=(st_tensor){strdup(name),fd,cursor,4,2,1};cursor+=4;
        cursor=(cursor+15)&~15;snprintf(name,sizeof(name),"%s.nvfp4_input_scale",base);
        if(pwrite(fd,&is,4,cursor)!=4)return fail("native input scale fixture");
        m.S.t[n++]=(st_tensor){strdup(name),fd,cursor,4,2,1};cursor+=4;
    }
    if(ftruncate(fd,(cursor+4095)&~4095))return fail("native fixture truncate");
    if(uring_batch_init(&batch))return fail("native ring init");
    uring_batch_reset(&batch);int li=uring_load_add(&batch,&m,1,7,&slot,1);
    if(li!=0||!batch.load[li].native_direct||batch.load[li].iobuf||
       uring_submit_batch(&batch)||uring_finalize_load(&batch,li,1))
        return fail("native direct record load");
    int bad=slot.eid!=7||slot.g.fmt!=COLI_TENSOR_MODELOPT_NVFP4||
        ((uintptr_t)slot.g.q4&15)||((uintptr_t)slot.g.block_scales&15)||
        ((uintptr_t)slot.u.q4&15)||((uintptr_t)slot.u.block_scales&15)||
        ((uintptr_t)slot.d.q4&15)||((uintptr_t)slot.d.block_scales&15)||
        slot.g.tensor_scale!=.5f||slot.g.input_scale!=1.25f;
    coli_uring_close(&batch.ring);compat_aligned_free(slot.slab);free(slot.fslab);
    UringBatch malformed={0};ESlot rejected={0};m.S.t[4].off+=4;
    if(uring_batch_init(&malformed))return fail("malformed native ring init");
    uring_batch_reset(&malformed);li=uring_load_add(&malformed,&m,1,7,&rejected,0);
    bad|=li!=0||malformed.load[li].error!=EINVAL||rejected.slab!=NULL;
    coli_uring_close(&malformed.ring);
    for(int i=0;i<m.S.n;i++)free(m.S.t[i].name);free(m.S.t);
    return bad?fail("native direct tensor views"):0;
}

int main(void){
    char path[]="/tmp/coli-uring-XXXXXX";
    int fd=mkstemp(path); if(fd<0) return fail("mkstemp");
    unlink(path);
    enum { N=4, SZ=4096 };
    unsigned char src[N][SZ],dst[N][SZ];
    for(int i=0;i<N;i++){
        memset(src[i],0,sizeof(src[i])); memset(dst[i],0,sizeof(dst[i]));
        for(int j=0;j<SZ;j++) src[i][j]=(unsigned char)(i*37+j);
        if(pwrite(fd,src[i],SZ,(off_t)i*SZ)!=SZ){ close(fd); return fail("pwrite fixture"); }
    }
    ColiUring ring;
    if(coli_uring_init(&ring,8)){
        if(errno==EPERM || errno==ENOSYS || errno==EACCES){
            printf("test_uring: skipped (%s)\n",strerror(errno)); close(fd); return 0;
        }
        close(fd); return fail("io_uring_setup");
    }
    if(coli_uring_set_workers(&ring,4)){
        coli_uring_close(&ring); close(fd); return fail("io-wq worker limit");
    }
    for(int i=0;i<N;i++) if(coli_uring_prep_read(&ring,fd,dst[i],SZ,(off_t)i*SZ,(uint64_t)i+1)){
        coli_uring_close(&ring); close(fd); return fail("prepare read");
    }
    if(coli_uring_enter(&ring,0)<0){ coli_uring_close(&ring); close(fd); return fail("submit"); }
    int seen[N]={0},complete=0;
    while(complete<N){
        struct io_uring_cqe cqe; int reaped=0;
        while(coli_uring_peek(&ring,&cqe)){
            reaped=1;
            if(cqe.user_data<1 || cqe.user_data>N || cqe.res!=SZ){
                coli_uring_close(&ring); close(fd); return fail("bad completion");
            }
            int i=(int)cqe.user_data-1;
            if(seen[i]++){ coli_uring_close(&ring); close(fd); return fail("duplicate completion"); }
            complete++;
        }
        if(!reaped && complete<N && coli_uring_enter(&ring,1)<0){
            coli_uring_close(&ring); close(fd); return fail("completion wait");
        }
    }
    for(int i=0;i<N;i++) if(memcmp(src[i],dst[i],SZ)){
        coli_uring_close(&ring); close(fd); return fail("read data mismatch");
    }
    coli_uring_close(&ring);
    if(ftruncate(fd,0) || test_expert_layout(fd) ||
       ftruncate(fd,0) || test_native_component_layout(fd)){ close(fd); return 1; }
    close(fd);
    puts("test_uring: ok"); return 0;
}
