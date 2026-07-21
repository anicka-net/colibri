/* kv_alloc must survive re-allocation on the same KVState: every free path is
 * guarded by if(k->Lc) precisely so callers (context resize, slot re-init) can
 * call it again. A stale duplicate free block frees every Lc[i]/Rc[i] and both
 * arrays twice on the second call -> allocator abort. No model file needed:
 * the CPU path of kv_alloc only reads c->n_layers/kv_lora/qk_rope. */
#include <assert.h>
#define main coli_glm_main_unused
#include "../glm.c"
#undef main

typedef struct {
    Model *m;
    _Atomic int started;
    _Atomic int done;
} RepinProbe;

static void *repin_probe(void *arg){
    RepinProbe *p=arg;
    atomic_store_explicit(&p->started,1,memory_order_release);
    repin_adapt(p->m,1);
    atomic_store_explicit(&p->done,1,memory_order_release);
    return NULL;
}

int main(void){
    static Model m;
    m.c.n_layers=2; m.c.kv_lora=8; m.c.qk_rope=4;
    m.kv=calloc(1,sizeof(KVState));
    kv_alloc(&m,16);
    for(int i=0;i<m.c.n_layers+1;i++){ m.Lc[i][0]=1.0f; m.Rc[i][0]=1.0f; }
    kv_alloc(&m,32);                       /* the re-allocation path under test */
    for(int i=0;i<m.c.n_layers+1;i++){
        m.Lc[i][(int64_t)32*m.c.kv_lora-1]=2.0f;
        m.Rc[i][(int64_t)32*m.c.qk_rope-1]=2.0f;
    }
    m.ecap_alloc=100;
    g_adaptive_cap_floor=17; g_adaptive_cap_maxctx=100; g_adaptive_cap_margin=10;
    g_adaptive_cap_kv_token_b=2; g_adaptive_cap_scratch_token_b=1;
    g_adaptive_cap_slot_b=10; g_adaptive_cap_highwater=0;
    assert(adaptive_cap_target(&m,10)==41);
    g_adaptive_cap_highwater=90;
    assert(adaptive_cap_target(&m,10)==27);
    assert(cuda_ic_shadow_bytes_for(2,128,131072,1,4,0)==134217728.0);
    assert(cuda_ic_shadow_bytes_for(2,128,131072,3,4,0)==402653184.0);
    assert(cuda_ic_shadow_bytes_for(2,128,131072,1,1,4)==34603008.0);
    assert(cuda_ic_shadow_layer_eligible(1,1,1)); /* dense and sparse resident layers */
    assert(!cuda_ic_shadow_layer_eligible(0,1,1)); /* SHARED indexer */
    assert(!cuda_ic_shadow_layer_eligible(1,0,1)); /* resident path unavailable */
    assert(!cuda_ic_shadow_layer_eligible(1,1,0)); /* discrete CUDA memory */
    assert(rss_guard_target(42,1.0e9,2.0e9)==41);
    assert(rss_guard_target(17,40.0e9,2.0e9)==1);
    m.c.n_layers=0; m.ecap=2;
    m.ecache=calloc(1,sizeof(*m.ecache)); m.ecn=calloc(1,sizeof(*m.ecn));
    m.ecache[0]=calloc(2,sizeof(ESlot)); m.ecn[0]=2;
    for(int z=0;z<2;z++){
        ESlot *s=&m.ecache[0][z]; s->eid=10+z; s->used=1+z;
        if(z==0){
            QT *q[3]={&s->g,&s->u,&s->d};
            for(int k=0;k<3;k++){ q[k]->fmt=1; q[k]->q8=malloc(64); q[k]->s=malloc(16); }
        }else{
            assert(!posix_memalign((void**)&s->slab,64,4096)); s->slab_cap=4096;
            s->fslab=malloc(16*sizeof(float)); s->fslab_cap=16;
        }
    }
    g_rss_slot_b=2.0e9; g_rss_cap_ceiling=INT_MAX;
    rss_guard_apply(&m,3.0,1.0);
    assert(m.ecap==1 && m.ecn[0]==1 && m.ecache[0][0].eid==11);
    assert(g_rss_cap_ceiling==1);
    Model repin_model={0}; RepinProbe probe={&repin_model,0,0}; pthread_t thread;
    pthread_mutex_lock(&g_pilot_mx);
    assert(!pthread_create(&thread,NULL,repin_probe,&probe));
    while(!atomic_load_explicit(&probe.started,memory_order_acquire)) usleep(100);
    usleep(20000);
    assert(!atomic_load_explicit(&probe.done,memory_order_acquire));
    pthread_mutex_unlock(&g_pilot_mx);
    assert(!pthread_join(thread,NULL));
    assert(atomic_load_explicit(&probe.done,memory_order_acquire));
    assert(!pthread_mutex_trylock(&g_pilot_mx));
    pthread_mutex_unlock(&g_pilot_mx);
    expert_lru_release(&m.ecache[0][0]);
    free(m.ecache[0]); free(m.ecache); free(m.ecn);
    printf("OK kv_alloc re-allocation\n");
    return 0;
}
