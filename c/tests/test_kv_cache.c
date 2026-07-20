#include <assert.h>

#define main coli_glm_main_unused
#include "../glm.c"
#undef main

static void fill_kv(Model *m, int len, float base){
    for(int p=0;p<len;p++) for(int l=0;l<m->c.n_layers;l++){
        for(int j=0;j<m->c.kv_lora;j++)
            m->Lc[l][(int64_t)p*m->c.kv_lora+j]=base+100*l+10*p+j;
        for(int j=0;j<m->c.qk_rope;j++)
            m->Rc[l][(int64_t)p*m->c.qk_rope+j]=base+1000+100*l+10*p+j;
    }
}

static void cleanup_dir(const char *dir){
    DIR *d=opendir(dir); struct dirent *e;
    if(d){ while((e=readdir(d))) if(strcmp(e->d_name,".") && strcmp(e->d_name,"..")){
        char p[2048]; snprintf(p,sizeof(p),"%s/%s",dir,e->d_name); remove(p);
    } closedir(d); }
    rmdir(dir);
}

int main(void){
    char dir[256]; snprintf(dir,sizeof(dir),".kv-cache-%ld",(long)getpid());
    cleanup_dir(dir); assert(kv_cache_mkdir(dir));

    static Model m; ServeCtx s={0};
    m.c.n_layers=2; m.c.kv_lora=3; m.c.qk_rope=2; m.c.vocab=128;
    s.kv.kv_start=calloc((size_t)m.c.n_layers+1,sizeof(int));
    s.hist=malloc(8*sizeof(int)); m.kv=&s.kv; kv_alloc(&m,8);
    g_kvsave=1; g_kv_cache.enabled=1; g_kv_cache.budget=1LL<<30;
    snprintf(g_kv_cache.dir,sizeof(g_kv_cache.dir),"%s",dir);

    int first_hist[]={1,2,3,4};
    memcpy(s.hist,first_hist,sizeof(first_hist)); s.len=4; fill_kv(&m,4,10);
    assert(kv_cache_checkpoint(&m,&s));
    char first_tok[2048],first_kv[2048];
    strcpy(first_tok,s.source_tok); assert(kv_cache_pair_path(first_tok,first_kv,sizeof(first_kv)));

    memset(m.Lc[0],0,(size_t)8*m.c.kv_lora*4);
    memset(m.Rc[0],0,(size_t)8*m.c.qk_rope*4);
    int loaded[8]={0};
    assert(kv_disk_load_prefix(&m,first_kv,loaded,8,3,first_hist,0)==3);
    assert(!memcmp(loaded,first_hist,3*sizeof(int)));
    assert(m.Lc[0][2*m.c.kv_lora+1]==31);
    assert(m.Rc[0][2*m.c.qk_rope+1]==1031);

    int second_hist[]={1,2,9,10};
    memcpy(s.hist,second_hist,sizeof(second_hist)); s.len=4;
    fill_kv(&m,4,20); assert(kv_cache_checkpoint(&m,&s));
    char second_tok[2048],second_kv[2048];
    strcpy(second_tok,s.source_tok); assert(kv_cache_pair_path(second_tok,second_kv,sizeof(second_kv)));
    assert(access(first_tok,F_OK)==0 && access(first_kv,F_OK)==0); /* branch kept */
    int query[]={1,2,9,11}; char best[2048]={0};

    char bad_kv[2048],bad_tok[2048];
    snprintf(bad_kv,sizeof(bad_kv),"%s/corrupt.kv",dir);
    snprintf(bad_tok,sizeof(bad_tok),"%s/corrupt.tok",dir);
    assert(kv_cache_write_data(&m,bad_kv,second_hist,4));
    assert(kv_cache_write_meta(&m,bad_tok,query,4));
    FILE *bad=fopen(bad_kv,"r+b"); assert(bad); fputc('X',bad); fclose(bad);
    assert(kv_cache_find_best(&m,query,4,best,sizeof(best))==3);
    assert(!strcmp(best,second_tok));

    char mismatch_kv[2048],mismatch_tok[2048];
    snprintf(mismatch_kv,sizeof(mismatch_kv),"%s/mismatch.kv",dir);
    snprintf(mismatch_tok,sizeof(mismatch_tok),"%s/mismatch.tok",dir);
    assert(kv_cache_write_data(&m,mismatch_kv,second_hist,4));
    assert(kv_cache_write_meta(&m,mismatch_tok,query,4));
    assert(kv_cache_extend_data(&m,mismatch_kv,second_hist,3,4)); /* metadata may lag data after a crash */
    memcpy(s.hist,first_hist,sizeof(first_hist)); s.len=4;
    strcpy(s.source_tok,first_tok); fill_kv(&m,4,10);
    assert(kv_cache_restore_best(&m,&s,query,4,8)==2);
    assert(s.len==2 && !strcmp(s.source_tok,first_tok));
    kv_cache_remove_pair(mismatch_tok);
    assert(kv_cache_restore_best(&m,&s,query,4,8)==3);
    assert(s.len==3 && s.hist[2]==9 && !strcmp(s.source_tok,second_tok));
    assert(m.Lc[0][2*m.c.kv_lora+1]==41);

    int extended[]={1,2,9,10,12};
    memcpy(s.hist,extended,sizeof(extended)); s.len=5; fill_kv(&m,5,30);
    assert(kv_cache_checkpoint(&m,&s));
    char newest_tok[2048]; strcpy(newest_tok,s.source_tok);
    assert(access(second_tok,F_OK)!=0 && access(second_kv,F_OK)!=0);

    char newest_kv[2048]; struct stat ks,ts;
    assert(kv_cache_pair_path(newest_tok,newest_kv,sizeof(newest_kv)));
    assert(!stat(newest_kv,&ks) && !stat(newest_tok,&ts));
    g_kv_cache.budget=(int64_t)ks.st_size+ts.st_size;
    kv_cache_evict(newest_tok);
    assert(access(first_tok,F_OK)!=0 && access(first_kv,F_OK)!=0);
    assert(access(newest_tok,F_OK)==0 && access(newest_kv,F_OK)==0);

    serve_ctx_free(&m,&s);
    cleanup_dir(dir);
    puts("OK bounded prefix KV cache");
    return 0;
}
