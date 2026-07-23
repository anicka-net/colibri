#define _GNU_SOURCE
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static void add_tensor(shards *s,int *index,const char *name,int fd,int64_t off,
                       int64_t bytes,int dtype){
    s->t[*index]=(st_tensor){strdup(name),fd,off,bytes,dtype,
                             dtype==2?bytes/4:bytes};
    (*index)++;
}

static void write_projection(Model *m,FILE *file,int *index,const char *base,
                             int O,int I,uint8_t seed){
    int64_t wb=(int64_t)O*((I+1)/2);
    int64_t sb=(int64_t)coli_nvfp4_cutlass_scale_bytes(O,I);
    uint8_t *weights=malloc((size_t)wb),*scales=calloc(1,(size_t)sb);
    assert(weights&&scales);
    for(int64_t z=0;z<wb;z++)weights[z]=(uint8_t)(seed+z*17);
    for(int o=0;o<O;o++)for(int g=0;g<(I+15)/16;g++)
        scales[coli_nvfp4_cutlass_scale_offset(o,g,I)]=0x38; /* E4M3 1.0 */
    int fd=fileno(file);int64_t off=ftello(file);
    assert(fwrite(weights,1,(size_t)wb,file)==(size_t)wb);
    add_tensor(&m->S,index,base,fd,off,wb,3);
    char name[384];off=ftello(file);
    assert(fwrite(scales,1,(size_t)sb,file)==(size_t)sb);
    snprintf(name,sizeof(name),"%s.nvfp4_scale",base);
    add_tensor(&m->S,index,name,fd,off,sb,3);
    const float tensor_scale=0.5f,input_scale=1.25f;
    off=ftello(file);assert(fwrite(&tensor_scale,4,1,file)==1);
    snprintf(name,sizeof(name),"%s.nvfp4_tensor_scale",base);
    add_tensor(&m->S,index,name,fd,off,4,2);
    off=ftello(file);assert(fwrite(&input_scale,4,1,file)==1);
    snprintf(name,sizeof(name),"%s.nvfp4_input_scale",base);
    add_tensor(&m->S,index,name,fd,off,4,2);
    free(weights);free(scales);
}

static void assert_projection(const QT *q,int O,int I){
    assert(q->fmt==COLI_TENSOR_MODELOPT_NVFP4);
    assert(q->O==O&&q->I==I&&q->gs==16);
    assert(q->q4&&q->block_scales&&!q->s);
    assert(q->tensor_scale==0.5f&&q->input_scale==1.25f);
    assert(q->scale_layout==COLI_SCALE_CUTLASS_SM1XX_128X4);
    float x[16]={0},y[16]={0};for(int i=0;i<I;i++)x[i]=(float)(i+1)/8.0f;
    assert(coli_matmul_nvfp4_w4a32_ref(y,x,q->q4,q->block_scales,q->tensor_scale,
                                       q->scale_layout,1,I,O));
    for(int o=0;o<O;o++)assert(isfinite(y[o]));
}

int main(void){
    /* Faithful snapshots keep the token boundary in BF16.  Decode it
     * explicitly instead of falling through to the legacy packed-INT2 case. */
    {
        uint16_t rows[6]={0x3f80,0xc000,0x3f00,0x4040,0x0000,0xbf80};
        Model em={0};em.c.hidden=3;em.embed.O=2;
        em.embed.fmt=COLI_TENSOR_BF16;em.embed.bf16=rows;
        float out[3]={0};embed_row(&em,1,out);
        assert(out[0]==3.0f&&out[1]==0.0f&&out[2]==-1.0f);
    }
    char path[]="test_nvfp4_loader_XXXXXX";int fd=mkstemp(path);assert(fd>=0);
    FILE *file=fdopen(fd,"w+b");assert(file);
    Model m={0};m.c.hidden=5;m.c.moe_inter=7;m.c.n_layers=2;m.c.n_experts=4;
    m.ebits=4;m.manifest.present=1;
    m.native_valid=calloc((size_t)(m.c.n_layers+1)*m.c.n_experts,sizeof(*m.native_valid));
    assert(m.native_valid);
    m.S.n=m.S.cap=12;m.S.t=calloc(12,sizeof(st_tensor));assert(m.S.t);
    int n=0;char base[288];const char *p[3]={"gate_proj","up_proj","down_proj"};
    for(int k=0;k<3;k++){
        snprintf(base,sizeof(base),"model.layers.2.mlp.experts.3.%s.weight",p[k]);
        write_projection(&m,file,&n,base,k<2?7:5,k<2?5:7,(uint8_t)(3+k));
    }
    assert(n==12);fflush(file);
    ESlot slot={.eid=-1};
    assert(expert_load_impl(&m,2,3,&slot,0,1)==0&&slot.eid==3);
    assert(atomic_load(&m.native_valid[2*m.c.n_experts+3])==1);
    assert_projection(&slot.g,7,5);assert_projection(&slot.u,7,5);assert_projection(&slot.d,5,7);
    int64_t logical=qt_bytes(&slot.g)+qt_bytes(&slot.u)+qt_bytes(&slot.d);
    assert(logical>0&&slot.slab_cap>=4096&&slot.fslab_cap==6);
    int64_t released=expert_lru_release(&slot);assert(released>=4096&&slot.eid==-1&&!slot.slab);

    /* Native records use the same validated pread path in direct/mmap modes;
     * neither flag may reinterpret a native tensor as a legacy mapped record. */
    g_direct=1;g_mmap=1;
    assert(expert_load_impl(&m,2,3,&slot,0,1)==0);
    assert_projection(&slot.g,7,5);expert_lru_release(&slot);g_direct=g_mmap=0;

    /* Metadata corruption is rejected before allocation. */
    m.S.t[1].dtype=2;
    assert(expert_load_impl(&m,2,3,&slot,0,1)==-1&&!slot.slab);
    m.S.t[1].dtype=3;

    /* A shard shortened underneath live descriptors must report failure and
     * never publish the expert id or a partially initialized QT. */
    assert(ftruncate(fd,m.S.t[11].off+2)==0);
    assert(expert_load_impl(&m,2,3,&slot,0,1)==-1&&slot.eid==-1);
    assert(!slot.g.q4&&!slot.u.q4&&!slot.d.q4);
    expert_lru_release(&slot);
    for(int i=0;i<m.S.n;i++)free(m.S.t[i].name);free(m.S.t);free(m.native_valid);
    fclose(file);unlink(path);
    puts("native NVFP4 expert load/evict/reload: ok");
    return 0;
}
