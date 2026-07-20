#ifndef COLI_NATIVE_CHAT_H
#define COLI_NATIVE_CHAT_H
int coli_chat_run(const char *engine, const char *model, int cap,
                  int expert_bits, int dense_bits, int max_tokens, int context);
#endif
