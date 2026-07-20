#define _GNU_SOURCE
#include "native_plan.h"
#include "json.h"

#include <dirent.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

typedef struct {
  int layer, expert;
  unsigned long long bytes;
} expert_group;

static char *read_file(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return NULL;
  if (fseek(f, 0, SEEK_END)) {
    fclose(f);
    return NULL;
  }
  long n = ftell(f);
  if (n < 0 || fseek(f, 0, SEEK_SET)) {
    fclose(f);
    return NULL;
  }
  char *p = malloc((size_t)n + 1);
  if (fread(p, 1, (size_t)n, f) != (size_t)n) {
    free(p);
    fclose(f);
    return NULL;
  }
  fclose(f);
  p[n] = 0;
  if (size)
    *size = (size_t)n;
  return p;
}
static int cmp_u64(const void *a, const void *b) {
  unsigned long long x = *(const unsigned long long *)a,
                     y = *(const unsigned long long *)b;
  return x > y ? 1 : x < y ? -1 : 0;
}
static unsigned long long median(unsigned long long *v, int n) {
  if (!n)
    return 0;
  qsort(v, (size_t)n, sizeof(*v), cmp_u64);
  return v[n / 2];
}
static long long jinteger(jval *o, const char *k) {
  jval *v = json_get(o, k);
  return v && v->t == J_NUM ? (long long)v->num : 0;
}

int coli_analyze_model(const char *model, coli_model_info *info, char *error,
                       size_t ec) {
  memset(info, 0, sizeof(*info));
  char resolved[4096];
  if (!realpath(model, resolved)) {
    snprintf(error, ec, "model directory does not exist: %s", model);
    return -1;
  }
  snprintf(info->path, sizeof(info->path), "%s", resolved);
  char cfg_path[8192];
  snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", resolved);
  size_t cfg_n;
  char *cfg = read_file(cfg_path, &cfg_n);
  if (!cfg) {
    snprintf(error, ec, "missing config.json: %s", resolved);
    return -1;
  }
  jval *root = json_parse(cfg, NULL);
  free(cfg);
  if (!root || root->t != J_OBJ) {
    json_free(root);
    snprintf(error, ec, "invalid config.json: %s", resolved);
    return -1;
  }
  info->num_hidden_layers = (int)jinteger(root, "num_hidden_layers");
  info->configured_experts = (int)jinteger(root, "n_routed_experts");
  info->kv_lora_rank = (int)jinteger(root, "kv_lora_rank");
  info->qk_rope_head_dim = (int)jinteger(root, "qk_rope_head_dim");
  info->qk_nope_head_dim = (int)jinteger(root, "qk_nope_head_dim");
  info->v_head_dim = (int)jinteger(root, "v_head_dim");
  info->num_attention_heads = (int)jinteger(root, "num_attention_heads");
  json_free(root);
  DIR *d = opendir(resolved);
  if (!d) {
    snprintf(error, ec, "cannot read model: %s", strerror(errno));
    return -1;
  }
  expert_group *groups = NULL;
  int gn = 0, gcap = 0;
  struct dirent *de;
  while ((de = readdir(d))) {
    size_t ln = strlen(de->d_name);
    if (ln < 12 || strcmp(de->d_name + ln - 12, ".safetensors"))
      continue;
    char path[8192];
    snprintf(path, sizeof(path), "%s/%s", resolved, de->d_name);
    struct stat st;
    if (stat(path, &st))
      continue;
    info->shards++;
    info->model_bytes += (unsigned long long)st.st_size;
    FILE *f = fopen(path, "rb");
    if (!f)
      continue;
    uint64_t hn = 0;
    if (fread(&hn, 1, 8, f) != 8 || hn < 2 || hn > (uint64_t)st.st_size - 8) {
      fclose(f);
      closedir(d);
      free(groups);
      snprintf(error, ec, "invalid safetensors header: %s", path);
      return -1;
    }
    char *h = malloc((size_t)hn + 1);
    if (fread(h, 1, (size_t)hn, f) != (size_t)hn) {
      free(h);
      fclose(f);
      closedir(d);
      free(groups);
      snprintf(error, ec, "short safetensors header: %s", path);
      return -1;
    }
    fclose(f);
    h[hn] = 0;
    jval *hdr = json_parse(h, NULL);
    free(h);
    if (!hdr || hdr->t != J_OBJ) {
      json_free(hdr);
      closedir(d);
      free(groups);
      snprintf(error, ec, "invalid safetensors JSON: %s", path);
      return -1;
    }
    for (int i = 0; i < hdr->len; i++) {
      if (!strcmp(hdr->keys[i], "__metadata__"))
        continue;
      jval *offs = json_get(hdr->kids[i], "data_offsets");
      if (!offs || offs->t != J_ARR || offs->len != 2 ||
          offs->kids[0]->t != J_NUM || offs->kids[1]->t != J_NUM)
        continue;
      unsigned long long begin = (unsigned long long)offs->kids[0]->num,
                         end = (unsigned long long)offs->kids[1]->num;
      if (end < begin || end > (unsigned long long)st.st_size - 8 - hn) {
        json_free(hdr);
        closedir(d);
        free(groups);
        snprintf(error, ec, "invalid tensor offsets: %s", path);
        return -1;
      }
      unsigned long long bytes = end - begin;
      int layer, expert;
      const char *p = strstr(hdr->keys[i], "model.layers.");
      if (p &&
          sscanf(p, "model.layers.%d.mlp.experts.%d.", &layer, &expert) == 2) {
        int at = -1;
        for (int q = 0; q < gn; q++)
          if (groups[q].layer == layer && groups[q].expert == expert) {
            at = q;
            break;
          }
        if (at < 0) {
          if (gn == gcap) {
            gcap = gcap ? gcap * 2 : 256;
            groups = realloc(groups, (size_t)gcap * sizeof(*groups));
          }
          at = gn++;
          groups[at] = (expert_group){layer, expert, 0};
        }
        groups[at].bytes += bytes;
      } else
        info->dense_bytes += bytes;
    }
    json_free(hdr);
  }
  closedir(d);
  if (!info->shards) {
    free(groups);
    snprintf(error, ec, "no safetensors shards: %s", resolved);
    return -1;
  }
  info->expert_count = gn;
  for (int i = 0; i < gn; i++)
    info->expert_bytes += groups[i].bytes;
  unsigned long long *layer_medians =
      calloc((size_t)(info->num_hidden_layers + 2), sizeof(*layer_medians));
  int layers = 0;
  for (int l = 0; l <= info->num_hidden_layers; l++) {
    unsigned long long *v = malloc((size_t)(gn ? gn : 1) * sizeof(*v));
    int n = 0;
    for (int i = 0; i < gn; i++)
      if (groups[i].layer == l)
        v[n++] = groups[i].bytes;
    if (n) {
      layer_medians[l] = median(v, n);
      info->per_cap_bytes += layer_medians[l];
      layers++;
    }
    free(v);
  }
  info->expert_layers = layers;
  unsigned long long *lm = malloc((size_t)(layers ? layers : 1) * sizeof(*lm));
  int li = 0;
  for (int l = 0; l <= info->num_hidden_layers; l++)
    if (layer_medians[l])
      lm[li++] = layer_medians[l];
  info->typical_expert_bytes = median(lm, li);
  free(lm);
  free(layer_medians);
  free(groups);
  return 0;
}

unsigned long long coli_memory_available(void) {
  FILE *f = fopen("/proc/meminfo", "r");
  if (!f)
    return 0;
  char line[256];
  unsigned long long kb = 0;
  while (fgets(line, sizeof(line), f))
    if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1)
      break;
  fclose(f);
  return kb * 1024ULL;
}
int coli_discover_gpus(coli_gpu_info *g, int cap) {
  FILE *f = popen("nvidia-smi --query-gpu=index,name,memory.total,memory.free "
                  "--format=csv,noheader,nounits 2>/dev/null",
                  "r");
  if (!f)
    return 0;
  char line[512];
  int n = 0;
  while (n < cap && fgets(line, sizeof(line), f)) {
    char *save = NULL, *a = strtok_r(line, ",", &save),
         *b = strtok_r(NULL, ",", &save), *c = strtok_r(NULL, ",", &save),
         *d = strtok_r(NULL, ",\r\n", &save);
    if (!a || !b || !c || !d)
      continue;
    while (*b == ' ')
      b++;
    g[n].index = atoi(a);
    snprintf(g[n].name, sizeof(g[n].name), "%s", b);
    g[n].total_bytes = strtoull(c, NULL, 10) * 1024ULL * 1024ULL;
    g[n].free_bytes = strtoull(d, NULL, 10) * 1024ULL * 1024ULL;
    n++;
  }
  pclose(f);
  return n;
}
static int selected(const coli_plan_request *r, int index) {
  if (r->gpu_disabled)
    return 0;
  if (!r->gpu_indices)
    return 1;
  for (int i = 0; i < r->gpu_index_count; i++)
    if (r->gpu_indices[i] == index)
      return 1;
  return 0;
}
static void warn(coli_plan *p, const char *s) {
  if (p->warning_count < 8)
    snprintf(p->warnings[p->warning_count++], 192, "%s", s);
}
int coli_build_plan(const char *model, const coli_plan_request *r, coli_plan *p,
                    char *error, size_t ec) {
  memset(p, 0, sizeof(*p));
  const char *policy = r->policy ? r->policy : "quality";
  if (strcmp(policy, "quality") && strcmp(policy, "balanced") &&
      strcmp(policy, "experimental-fast")) {
    snprintf(error, ec, "unknown policy: %s", policy);
    return -1;
  }
  snprintf(p->policy, sizeof(p->policy), "%s", policy);
  p->quality_preserving = strcmp(policy, "experimental-fast") != 0;
  if (coli_analyze_model(model, &p->model, error, ec))
    return -1;
  p->available_memory = coli_memory_available();
  struct statvfs fs;
  if (!statvfs(p->model.path, &fs))
    p->available_disk = (unsigned long long)fs.f_bavail * fs.f_frsize;
  p->ram_budget = r->ram_gb > 0
                      ? (unsigned long long)(r->ram_gb * COLI_GB)
                      : (unsigned long long)(p->available_memory * .88);
  if (p->ram_budget < 4 * COLI_GB)
    p->ram_budget = 8 * COLI_GB;
  int layers = p->model.num_hidden_layers + 1;
  unsigned long long kv = (unsigned long long)layers * r->context *
                          (p->model.kv_lora_rank + p->model.qk_rope_head_dim) *
                          4ULL;
  unsigned long long kvbuf =
      (unsigned long long)r->context * p->model.num_attention_heads *
      (p->model.qk_nope_head_dim + p->model.v_head_dim) * 4ULL;
  p->runtime_bytes = (unsigned long long)(3.7 * COLI_GB) +
                     64 * p->model.typical_expert_bytes + kv + kvbuf;
  p->cache_bytes = p->ram_budget > p->model.dense_bytes + p->runtime_bytes
                       ? p->ram_budget - p->model.dense_bytes - p->runtime_bytes
                       : 0;
  p->cache_slots = p->model.per_cap_bytes
                       ? (int)(p->cache_bytes / p->model.per_cap_bytes)
                       : 0;
  if (p->model.configured_experts &&
      p->cache_slots > p->model.configured_experts)
    p->cache_slots = p->model.configured_experts;
  coli_gpu_info all[COLI_MAX_GPUS];
  int an = coli_discover_gpus(all, COLI_MAX_GPUS);
  unsigned long long safe = 0;
  for (int i = 0; i < an; i++)
    if (selected(r, all[i].index) && p->gpu_count < COLI_MAX_GPUS) {
      p->gpus[p->gpu_count] = all[i];
      unsigned long long usable =
          all[i].free_bytes > 2 * COLI_GB ? all[i].free_bytes - 2 * COLI_GB : 0;
      safe += usable;
      p->gpu_count++;
    }
  unsigned long long requested =
      r->vram_gb > 0 ? (unsigned long long)(r->vram_gb * COLI_GB) : safe;
  p->vram_budget = requested < safe ? requested : safe;
  if (p->vram_budget > p->model.expert_bytes)
    p->vram_budget = p->model.expert_bytes;
  p->vram_experts = p->model.typical_expert_bytes
                        ? (int)(p->vram_budget / p->model.typical_expert_bytes)
                        : 0;
  p->hot_bytes =
      (unsigned long long)p->vram_experts * p->model.typical_expert_bytes;
  if (p->hot_bytes > p->model.expert_bytes)
    p->hot_bytes = p->model.expert_bytes;
  unsigned long long remain = p->model.expert_bytes - p->hot_bytes;
  p->warm_bytes = remain < p->cache_bytes ? remain : p->cache_bytes;
  p->cold_bytes = remain - p->warm_bytes;
  p->physical_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
  if (p->physical_cores < 1)
    p->physical_cores = 1;
  if (p->cache_slots < 1)
    warn(p, "RAM budget cannot hold one expert slot per sparse layer");
  if (r->gpu_indices && p->gpu_count != r->gpu_index_count)
    warn(p, "one or more requested GPUs were not detected");
  if (p->gpu_count && p->vram_budget < requested)
    warn(p, "VRAM tier was clamped by free VRAM or model expert size");
  if (p->cold_bytes)
    warn(p, "cold expert misses may reach disk; normal decode speed depends on "
            "hit rate");
  snprintf(p->bottleneck, sizeof(p->bottleneck), "%s",
           p->cold_bytes   ? "disk expert misses"
           : p->warm_bytes ? "CPU expert compute and RAM bandwidth"
                           : "GPU compute and interconnect");
  return 0;
}
static double gb(unsigned long long n) { return (double)n / COLI_GB; }
void coli_format_plan(FILE *out, const coli_plan *p, int json) {
  if (json) {
    fprintf(
        out,
        "{\"version\":2,\"policy\":{\"name\":\"%s\",\"quality_preserving\":%s},"
        "\"model\":{\"path\":\"%s\",\"shards\":%d,\"model_bytes\":%llu,\"dense_"
        "bytes\":%llu,\"expert_bytes\":%llu,\"expert_count\":%d,\"expert_"
        "layers\":%d,\"typical_expert_bytes\":%llu,\"per_cap_bytes\":%llu},"
        "\"cpu\":{\"physical_cores\":%d,\"thread_policy\":\"physical-cores\"},"
        "\"tiers\":{\"disk\":{\"model_bytes\":%llu,\"available_bytes\":%llu,"
        "\"cold_expert_bytes\":%llu},\"ram\":{\"available_bytes\":%llu,"
        "\"budget_bytes\":%llu,\"dense_bytes\":%llu,\"runtime_bytes\":%llu,"
        "\"expert_cache_bytes\":%llu,\"warm_expert_bytes\":%llu,\"cache_slots_"
        "per_layer\":%d},\"vram\":{\"devices\":[",
        p->policy, p->quality_preserving ? "true" : "false", p->model.path,
        p->model.shards, p->model.model_bytes, p->model.dense_bytes,
        p->model.expert_bytes, p->model.expert_count, p->model.expert_layers,
        p->model.typical_expert_bytes, p->model.per_cap_bytes,
        p->physical_cores, p->model.model_bytes, p->available_disk,
        p->cold_bytes, p->available_memory, p->ram_budget, p->model.dense_bytes,
        p->runtime_bytes, p->cache_bytes, p->warm_bytes, p->cache_slots);
    for (int i = 0; i < p->gpu_count; i++) {
      if (i)
        fputc(',', out);
      fprintf(out,
              "{\"index\":%d,\"name\":\"%s\",\"total_bytes\":%llu,\"free_"
              "bytes\":%llu}",
              p->gpus[i].index, p->gpus[i].name, p->gpus[i].total_bytes,
              p->gpus[i].free_bytes);
    }
    fprintf(out,
            "],\"budget_bytes\":%llu,\"hot_expert_bytes\":%llu,\"expert_"
            "capacity\":%d,\"requires_host_backing\":false}},\"expected_"
            "bottleneck\":\"%s\",\"decisions\":[{\"target\":\"VRAM\"},{"
            "\"target\":\"RAM\"},{\"target\":\"Disk\"}],\"warnings\":[",
            p->vram_budget, p->hot_bytes, p->vram_experts, p->bottleneck);
    for (int i = 0; i < p->warning_count; i++) {
      if (i)
        fputc(',', out);
      fprintf(out, "\"%s\"", p->warnings[i]);
    }
    fprintf(out, "]}\n");
    return;
  }
  fprintf(out,
          "policy %s · quality-preserving %s\nmodel  %d shards · %.1f GB\ndisk "
          "  %.1f GB cold experts · %.1f GB free\nRAM    %.1f GB budget · %.1f "
          "GB dense · %.1f GB runtime · %.1f GB warm experts · cap %d/layer\n",
          p->policy, p->quality_preserving ? "yes" : "no", p->model.shards,
          gb(p->model.model_bytes), gb(p->cold_bytes), gb(p->available_disk),
          gb(p->ram_budget), gb(p->model.dense_bytes), gb(p->runtime_bytes),
          gb(p->warm_bytes), p->cache_slots);
  if (p->gpu_count) {
    fprintf(out, "VRAM   %.1f GB hot tier · ~%d experts · ", gb(p->vram_budget),
            p->vram_experts);
    for (int i = 0; i < p->gpu_count; i++)
      fprintf(out, "%s%d:%s", i ? ", " : "", p->gpus[i].index, p->gpus[i].name);
    fputc('\n', out);
  } else
    fprintf(out, "VRAM   no NVIDIA device detected · CPU path\n");
  fprintf(out, "limit  %s\n", p->bottleneck);
  for (int i = 0; i < p->warning_count; i++)
    fprintf(out, "warn   %s\n", p->warnings[i]);
}

int coli_doctor(FILE *out, const char *model, const char *engine,
                const coli_plan_request *r, int json) {
  struct stat ms, es;
  int model_ok =
      !stat(model, &ms) && S_ISDIR(ms.st_mode) && !access(model, R_OK);
  char cfg[8192], tok[8192];
  snprintf(cfg, sizeof(cfg), "%s/config.json", model);
  snprintf(tok, sizeof(tok), "%s/tokenizer.json", model);
  int cfg_ok = !access(cfg, R_OK), tok_ok = !access(tok, R_OK),
      engine_ok =
          !stat(engine, &es) && S_ISREG(es.st_mode) && !access(engine, X_OK);
  coli_plan p;
  char error[512] = {0};
  int plan_ok = !coli_build_plan(model, r, &p, error, sizeof(error));
  int failure = !model_ok || !cfg_ok || !tok_ok || !engine_ok || !plan_ok ||
                (plan_ok && p.ram_budget > p.available_memory);
  const char *status = failure                      ? "error"
                       : plan_ok && p.warning_count ? "warning"
                                                    : "ok";
  if (json) {
    fprintf(
        out,
        "{\"schema_version\":1,\"status\":\"%s\",\"model\":\"%s\",\"checks\":[",
        status, model);
    fprintf(out,
            "{\"id\":\"model.path\",\"status\":\"%s\",\"summary\":\"model "
            "directory is %s\"},",
            model_ok ? "pass" : "fail",
            model_ok ? "readable" : "missing or unreadable");
    fprintf(out,
            "{\"id\":\"model.config\",\"status\":\"%s\",\"summary\":\"config."
            "json is %s\"},",
            cfg_ok ? "pass" : "fail", cfg_ok ? "valid" : "missing or invalid");
    fprintf(out,
            "{\"id\":\"model.tokenizer\",\"status\":\"%s\",\"summary\":"
            "\"tokenizer.json %s\"},",
            tok_ok ? "pass" : "fail", tok_ok ? "found" : "is missing");
    fprintf(out,
            "{\"id\":\"engine.binary\",\"status\":\"%s\",\"summary\":\"engine "
            "%s\"},",
            engine_ok ? "pass" : "fail",
            engine_ok ? "executable is ready" : "is not built or executable");
    fprintf(out,
            "{\"id\":\"model.shards\",\"status\":\"%s\",\"summary\":\"%s\"}],"
            "\"plan\":",
            plan_ok ? "pass" : "fail",
            plan_ok ? "safetensors headers are valid" : error);
    if (plan_ok)
      coli_format_plan(out, &p, 1);
    else
      fprintf(out, "null");
    fprintf(out, "}\n");
  } else {
    fprintf(out,
            "colibri doctor · %s\n[%4s] model.path         model directory is "
            "%s\n[%4s] model.config       config.json is %s\n[%4s] "
            "model.tokenizer    tokenizer.json %s\n[%4s] engine.binary      "
            "engine %s\n[%4s] model.shards       %s\n\n",
            model, model_ok ? "ok" : "fail", model_ok ? "readable" : "missing",
            cfg_ok ? "ok" : "fail", cfg_ok ? "valid" : "missing",
            tok_ok ? "ok" : "fail", tok_ok ? "found" : "missing",
            engine_ok ? "ok" : "fail", engine_ok ? "ready" : "missing",
            plan_ok ? "ok" : "fail",
            plan_ok ? "safetensors headers valid" : error);
    if (plan_ok)
      coli_format_plan(out, &p, 0);
    fprintf(out, "\nresult %s\n", status);
  }
  return failure ? 1 : 0;
}
