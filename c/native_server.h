#ifndef COLI_NATIVE_SERVER_H
#define COLI_NATIVE_SERVER_H

typedef struct {
  const char *model;
  const char *engine;
  const char *host;
  const char *model_id;
  const char *api_key;
  const char *web_root;
  const char **model_aliases;
  int model_alias_count;
  const char **hidden_model_aliases;
  int hidden_model_alias_count;
  const char **cors_origins;
  int cors_origin_count;
  int port;
  int cap;
  int max_tokens;
  int max_queue;
  double queue_timeout;
  int kv_slots;
  int expert_bits;
  int dense_bits;
  int context_length;
  int default_thinking;
} coli_server_config;

int coli_server_run(const coli_server_config *config);

#endif
