#ifndef COLI_NATIVE_PLAN_H
#define COLI_NATIVE_PLAN_H

#include <stddef.h>
#include <stdio.h>

#define COLI_GB 1000000000ULL
#define COLI_MAX_GPUS 16

typedef struct {
  int index;
  char name[128];
  unsigned long long total_bytes, free_bytes;
} coli_gpu_info;

typedef struct {
  char path[4096];
  int shards, expert_count, expert_layers, configured_experts;
  unsigned long long model_bytes, dense_bytes, expert_bytes;
  unsigned long long typical_expert_bytes, per_cap_bytes;
  int num_hidden_layers, kv_lora_rank, qk_rope_head_dim;
  int qk_nope_head_dim, v_head_dim, num_attention_heads;
} coli_model_info;

typedef struct {
  const char *policy;
  double ram_gb, vram_gb;
  int context, gpu_disabled;
  const int *gpu_indices;
  int gpu_index_count;
} coli_plan_request;

typedef struct {
  coli_model_info model;
  char policy[32], bottleneck[80];
  int quality_preserving, physical_cores, cache_slots;
  unsigned long long available_memory, available_disk, ram_budget;
  unsigned long long runtime_bytes, cache_bytes, warm_bytes, cold_bytes;
  unsigned long long vram_budget, hot_bytes;
  int vram_experts, gpu_count;
  coli_gpu_info gpus[COLI_MAX_GPUS];
  char warnings[8][192];
  int warning_count;
} coli_plan;

int coli_analyze_model(const char *model, coli_model_info *info, char *error,
                       size_t error_cap);
unsigned long long coli_memory_available(void);
int coli_discover_gpus(coli_gpu_info *gpus, int cap);
int coli_build_plan(const char *model, const coli_plan_request *request,
                    coli_plan *plan, char *error, size_t error_cap);
void coli_format_plan(FILE *out, const coli_plan *plan, int json);
int coli_doctor(FILE *out, const char *model, const char *engine,
                const coli_plan_request *request, int json);

#endif
