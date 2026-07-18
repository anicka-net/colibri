#define main coli_olmoe_main_unused
#include "../olmoe.c"
#undef main

int main(void){
    Model model={0};
    int tokens[1]={0};
    double nll=0;
    int scored=tf_nll(&model,tokens,1,1,&nll);
    if(scored!=0 || !isnan(nll)){
        fprintf(stderr,"zero-target PPL guard failed: scored=%d nll=%g\n",scored,nll);
        return 1;
    }
    puts("zero-target PPL guard: ok");
    return 0;
}
