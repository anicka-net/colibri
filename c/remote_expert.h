#ifndef COLI_REMOTE_EXPERT_H
#define COLI_REMOTE_EXPERT_H

#include <stdio.h>

int coli_remote_expert_enabled(int layer, int S);
int coli_remote_expert_run(int layer, const float *x, int S, const int *uniq,
                           int nu, const int *idxs, const float *weights,
                           const int *keff, int K, int D, float *out);
void coli_remote_expert_stats(FILE *stream);
void coli_remote_expert_shutdown(void);

#endif
