/* kv_alloc must survive re-allocation on the same KVState: every free path is
 * guarded by if(k->Lc) precisely so callers (context resize, slot re-init) can
 * call it again. A stale duplicate free block frees every Lc[i]/Rc[i] and both
 * arrays twice on the second call -> allocator abort. No model file needed:
 * the CPU path of kv_alloc only reads c->n_layers/kv_lora/qk_rope. */
#include <assert.h>
#define main coli_glm_main_unused
#include "../glm.c"
#undef main

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
    assert(cuda_ic_shadow_bytes_for(2,128,131072,1)==134217728.0);
    assert(cuda_ic_shadow_bytes_for(2,128,131072,3)==402653184.0);
    assert(cuda_ic_shadow_layer_eligible(1,1,1,1));
    assert(!cuda_ic_shadow_layer_eligible(0,1,1,1)); /* SHARED indexer */
    assert(!cuda_ic_shadow_layer_eligible(1,0,1,1)); /* dense layer */
    assert(!cuda_ic_shadow_layer_eligible(1,1,0,1)); /* resident path unavailable */
    assert(!cuda_ic_shadow_layer_eligible(1,1,1,0)); /* discrete CUDA memory */
    printf("OK kv_alloc re-allocation\n");
    return 0;
}
