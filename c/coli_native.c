#define _GNU_SOURCE
#include "native_chat.h"
#include "native_plan.h"
#include "native_server.h"

#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static const char *envs(const char *name, const char *fallback) {
  const char *v = getenv(name);
  return v && *v ? v : fallback;
}
static int envi(const char *name, int fallback) {
  const char *v = getenv(name);
  return v && *v ? atoi(v) : fallback;
}
static double envd(const char *name, double fallback) {
  const char *v = getenv(name);
  return v && *v ? strtod(v, NULL) : fallback;
}
static int exists(const char *p) {
  struct stat st;
  return !stat(p, &st);
}
static void usage(FILE *f) {
  fprintf(
      f,
      "colibrì — tiny engine, immense model\n\n"
      "  coli serve [options]       native OpenAI-compatible persistent API\n"
      "  coli run [options] PROMPT  one-shot generation\n"
      "  coli build                 build the native engine and CLI\n"
      "  coli info [--model DIR]    installation/model summary\n\n"
      "  coli plan [--json]         native RAM/VRAM placement plan\n"
      "  coli doctor [--json]       read-only installation diagnostics\n\n"
      "Common: --model DIR --cap N --ngen N --ctx N --temp F --topp F --topk "
      "N\n"
      "Serve:  --host ADDR --port N --model-id ID --api-key KEY --max-queue N\n"
      "        --queue-timeout SEC --kv-slots N --expert-bits N --dense-bits "
      "N\n");
}

static const char *self_dir(const char *argv0, char out[PATH_MAX]) {
  char tmp[PATH_MAX];
  if (realpath(argv0, tmp)) {
    char *s = strrchr(tmp, '/');
    if (s)
      *s = 0;
    snprintf(out, PATH_MAX, "%s", tmp);
    return out;
  }
  snprintf(out, PATH_MAX, ".");
  return out;
}
static int opt_value(int argc, char **argv, int *i, const char **out) {
  if (*i + 1 >= argc) {
    fprintf(stderr, "%s needs a value\n", argv[*i]);
    return -1;
  }
  *out = argv[++*i];
  return 0;
}
static int make_dir(const char *path) {
  if (!mkdir(path, 0700) || errno == EEXIST)
    return 0;
  return -1;
}
static int pidfile_path(int port, char out[PATH_MAX]) {
  const char *runtime = getenv("XDG_RUNTIME_DIR");
  if (runtime && *runtime)
    return snprintf(out, PATH_MAX, "%s/colibri-serve-%d.pid", runtime, port) <
                   PATH_MAX
               ? 0
               : -1;
  const char *home = getenv("HOME");
  if (!home || !*home)
    return -1;
  char cache[PATH_MAX], dir[PATH_MAX];
  if (snprintf(cache, sizeof(cache), "%s/.cache", home) >= (int)sizeof(cache) ||
      snprintf(dir, sizeof(dir), "%s/colibri", cache) >= (int)sizeof(dir))
    return -1;
  if (make_dir(cache) || make_dir(dir))
    return -1;
  return snprintf(out, PATH_MAX, "%s/serve-%d.pid", dir, port) < PATH_MAX ? 0
                                                                          : -1;
}
static void apply_plan_environment(const coli_plan *p) {
  char value[128];
  if (!getenv("COLI_POLICY"))
    setenv("COLI_POLICY", p->policy, 1);
  if (!getenv("OMP_NUM_THREADS")) {
    snprintf(value, sizeof(value), "%d", p->physical_cores);
    setenv("OMP_NUM_THREADS", value, 1);
  }
  if (!getenv("OMP_PROC_BIND"))
    setenv("OMP_PROC_BIND", "spread", 1);
  if (!getenv("OMP_PLACES"))
    setenv("OMP_PLACES", "cores", 1);
  if (!strcmp(p->policy, "balanced") && !getenv("REPIN"))
    setenv("REPIN", "64", 1);
  if (!getenv("RAM_GB")) {
    snprintf(value, sizeof(value), "%.3f", (double)p->ram_budget / COLI_GB);
    setenv("RAM_GB", value, 1);
  }
  if (p->gpu_count && p->vram_budget &&
      (!getenv("COLI_CUDA") || strcmp(getenv("COLI_CUDA"), "0"))) {
    if (!getenv("COLI_CUDA"))
      setenv("COLI_CUDA", "1", 1);
    if (!getenv("COLI_GPU") && !getenv("COLI_GPUS")) {
      char ids[128] = {0};
      for (int i = 0; i < p->gpu_count; i++) {
        char one[16];
        snprintf(one, sizeof(one), "%s%d", i ? "," : "", p->gpus[i].index);
        strncat(ids, one, sizeof(ids) - strlen(ids) - 1);
      }
      setenv(p->gpu_count == 1 ? "COLI_GPU" : "COLI_GPUS", ids, 1);
    }
    if (!getenv("CUDA_EXPERT_GB")) {
      snprintf(value, sizeof(value), "%.3f", (double)p->vram_budget / COLI_GB);
      setenv("CUDA_EXPERT_GB", value, 1);
    }
  }
}

int main(int argc, char **argv) {
  signal(SIGPIPE, SIG_IGN);
  if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h") ||
      !strcmp(argv[1], "help")) {
    usage(stdout);
    return 0;
  }
  const char *cmd = argv[1];
  char dir[PATH_MAX], engine_buf[PATH_MAX], web_buf[PATH_MAX];
  self_dir(argv[0], dir);
  if (snprintf(engine_buf, sizeof(engine_buf), "%s/glm", dir) >=
      (int)sizeof(engine_buf))
    return 2;
  if (!exists(engine_buf)) {
    char installed[PATH_MAX];
    if (snprintf(installed, sizeof(installed), "%s/../libexec/colibri/glm",
                 dir) < (int)sizeof(installed) &&
        exists(installed))
      snprintf(engine_buf, sizeof(engine_buf), "%s", installed);
  }
  if (snprintf(web_buf, sizeof(web_buf), "%s/../web/dist", dir) >=
      (int)sizeof(web_buf))
    web_buf[0] = 0;
  const char *model = envs("COLI_MODEL", "/home/vincenzo/glm52_i4"),
             *engine = envs("COLI_ENGINE", engine_buf), *host = "127.0.0.1",
             *model_id = envs("COLI_MODEL_ID", "glm-5.2"),
             *api_key = getenv("COLI_API_KEY"),
             *web_root = envs("COLI_WEB_ROOT", web_buf);
  int port = 8000, cap = 8, ngen = 1024, ctx = envi("COLI_CONTEXT_LENGTH", 0),
      max_queue = envi("COLI_MAX_QUEUE", 8),
      kv_slots = envi("COLI_KV_SLOTS", 1), ebits = envi("COLI_EXPERT_BITS", 8),
      dbits = envi("COLI_DENSE_BITS", ebits), thinking = envi("COLI_THINK", 0);
  double timeout = envd("COLI_QUEUE_TIMEOUT", 300), temp = -1, topp = 0;
  double ram_gb = 0, vram_gb = 0;
  int topk = 0, json = 0, gpu_disabled = 0, gpu_ids[32], gpu_n = 0;
  int auto_tier = 0, dry_run = 0, repin = 0;
  const char *policy = envs("COLI_POLICY", "quality");
  const char *aliases[32], *hidden_aliases[32], *origins[32];
  int alias_n = 0, hidden_n = 0, origin_n = 0;
  int prompt_at = argc;
  for (int i = 2; i < argc; i++) {
    const char *k = argv[i], *v = NULL;
    if (!strcmp(k, "--model") || !strcmp(k, "--host") ||
        !strcmp(k, "--model-id") || !strcmp(k, "--api-key") ||
        !strcmp(k, "--model-alias") || !strcmp(k, "--hidden-model-alias") ||
        !strcmp(k, "--cors-origin") || !strcmp(k, "--port") ||
        !strcmp(k, "--cap") || !strcmp(k, "--ngen") || !strcmp(k, "--ctx") ||
        !strcmp(k, "--context-length") || !strcmp(k, "--max-queue") ||
        !strcmp(k, "--queue-timeout") || !strcmp(k, "--kv-slots") ||
        !strcmp(k, "--expert-bits") || !strcmp(k, "--dense-bits") ||
        !strcmp(k, "--temp") || !strcmp(k, "--topp") || !strcmp(k, "--topk") ||
        !strcmp(k, "--ram") || !strcmp(k, "--vram") || !strcmp(k, "--gpu") ||
        !strcmp(k, "--policy") || !strcmp(k, "--repin")) {
      if (opt_value(argc, argv, &i, &v))
        return 2;
      if (!strcmp(k, "--model"))
        model = v;
      else if (!strcmp(k, "--host"))
        host = v;
      else if (!strcmp(k, "--model-id"))
        model_id = v;
      else if (!strcmp(k, "--api-key"))
        api_key = v;
      else if (!strcmp(k, "--model-alias") && alias_n < 32)
        aliases[alias_n++] = v;
      else if (!strcmp(k, "--hidden-model-alias") && hidden_n < 32)
        hidden_aliases[hidden_n++] = v;
      else if (!strcmp(k, "--cors-origin") && origin_n < 32)
        origins[origin_n++] = v;
      else if (!strcmp(k, "--port"))
        port = atoi(v);
      else if (!strcmp(k, "--cap"))
        cap = atoi(v);
      else if (!strcmp(k, "--ngen"))
        ngen = atoi(v);
      else if (!strcmp(k, "--ctx") || !strcmp(k, "--context-length"))
        ctx = atoi(v);
      else if (!strcmp(k, "--max-queue"))
        max_queue = atoi(v);
      else if (!strcmp(k, "--queue-timeout"))
        timeout = strtod(v, NULL);
      else if (!strcmp(k, "--kv-slots"))
        kv_slots = atoi(v);
      else if (!strcmp(k, "--expert-bits"))
        ebits = atoi(v);
      else if (!strcmp(k, "--dense-bits"))
        dbits = atoi(v);
      else if (!strcmp(k, "--temp"))
        temp = strtod(v, NULL);
      else if (!strcmp(k, "--topp"))
        topp = strtod(v, NULL);
      else if (!strcmp(k, "--topk"))
        topk = atoi(v);
      else if (!strcmp(k, "--ram"))
        ram_gb = strtod(v, NULL);
      else if (!strcmp(k, "--vram"))
        vram_gb = strtod(v, NULL);
      else if (!strcmp(k, "--policy"))
        policy = v;
      else if (!strcmp(k, "--repin"))
        repin = atoi(v);
      else if (!strcmp(k, "--gpu")) {
        if (!strcmp(v, "none"))
          gpu_disabled = 1;
        else if (strcmp(v, "auto")) {
          char *copy = strdup(v), *save = NULL, *part;
          for (part = strtok_r(copy, ",", &save); part && gpu_n < 32;
               part = strtok_r(NULL, ",", &save))
            gpu_ids[gpu_n++] = atoi(part);
          free(copy);
        }
      }
    } else if (!strcmp(k, "--default-thinking"))
      thinking = 1;
    else if (!strcmp(k, "--auto-tier"))
      auto_tier = 1;
    else if (!strcmp(k, "--dry-run"))
      dry_run = 1;
    else if (!strcmp(k, "--json"))
      json = 1;
    else if (k[0] == '-') {
      fprintf(stderr, "unknown option: %s\n", k);
      return 2;
    } else {
      prompt_at = i;
      break;
    }
  }
  if (!strcmp(cmd, "build")) {
    char make_dir[PATH_MAX];
    snprintf(make_dir, sizeof(make_dir), "%s", dir);
    execlp("make", "make", "-C", make_dir, "glm", "coli-native", (char *)NULL);
    perror("make");
    return 2;
  }
  if (!strcmp(cmd, "info")) {
    printf("model      %s\nengine     %s (%s)\nruntime    native C (libc + "
           "pthreads)\n",
           model, engine, exists(engine) ? "ready" : "not built");
    return exists(engine) ? 0 : 1;
  }
  coli_plan_request plan_request = {.policy = policy,
                                    .ram_gb = ram_gb,
                                    .vram_gb = vram_gb,
                                    .context = ctx > 0 ? ctx : 4096,
                                    .gpu_disabled = gpu_disabled,
                                    .gpu_indices = gpu_n ? gpu_ids : NULL,
                                    .gpu_index_count = gpu_n};
  if (!strcmp(cmd, "plan")) {
    coli_plan plan;
    char error[512];
    if (coli_build_plan(model, &plan_request, &plan, error, sizeof(error))) {
      fprintf(stderr, "cannot create resource plan: %s\n", error);
      return 2;
    }
    coli_format_plan(stdout, &plan, json);
    return 0;
  }
  if (!strcmp(cmd, "doctor"))
    return coli_doctor(stdout, model, engine, &plan_request, json);
  if (!strcmp(cmd, "stop")) {
    char path[PATH_MAX];
    if (pidfile_path(port, path)) {
      fprintf(stderr, "cannot locate serve pidfile\n");
      return 2;
    }
    FILE *f = fopen(path, "r");
    long pid = 0;
    if (f) {
      fscanf(f, "%ld", &pid);
      fclose(f);
    }
    if (pid <= 1 || kill((pid_t)pid, 0)) {
      printf("nothing running on port %d\n", port);
      return 0;
    }
    printf("%s %ld: coli serve on port %d\n",
           dry_run ? "would stop" : "stopping", pid, port);
    if (dry_run)
      return 0;
    kill((pid_t)pid, SIGTERM);
    for (int i = 0; i < 50; i++) {
      if (kill((pid_t)pid, 0)) {
        unlink(path);
        printf("stopped\n");
        return 0;
      }
      usleep(100000);
    }
    fprintf(stderr, "service did not stop within 5 seconds\n");
    return 1;
  }
  if (auto_tier) {
    coli_plan plan;
    char error[512];
    if (coli_build_plan(model, &plan_request, &plan, error, sizeof(error))) {
      fprintf(stderr, "invalid resource plan: %s\n", error);
      return 2;
    }
    apply_plan_environment(&plan);
  }
  if (repin > 0) {
    char value[32];
    snprintf(value, sizeof(value), "%d", repin);
    setenv("REPIN", value, 1);
  }
  if (!exists(engine)) {
    fprintf(stderr, "engine is not built: %s\n", engine);
    return 2;
  }
  if (!strcmp(cmd, "run")) {
    if (prompt_at >= argc) {
      fprintf(stderr, "usage: coli run \"your prompt\"\n");
      return 2;
    }
    size_t n = 64;
    for (int i = prompt_at; i < argc; i++)
      n += strlen(argv[i]) + 1;
    char *p = malloc(n);
    snprintf(p, n, "[gMASK]<sop><|user|>");
    for (int i = prompt_at; i < argc; i++) {
      if (i > prompt_at)
        strcat(p, " ");
      strcat(p, argv[i]);
    }
    strcat(p, "<|assistant|><think></think>");
    setenv("SNAP", model, 1);
    setenv("PROMPT", p, 1);
    char val[64];
    snprintf(val, sizeof(val), "%d", ngen);
    setenv("NGEN", val, 1);
    if (ctx) {
      snprintf(val, sizeof(val), "%d", ctx);
      setenv("CTX", val, 1);
    }
    if (temp >= 0) {
      snprintf(val, sizeof(val), "%.8g", temp);
      setenv("TEMP", val, 1);
    }
    if (topp > 0) {
      snprintf(val, sizeof(val), "%.8g", topp);
      setenv("TOPP", val, 1);
    }
    if (topk > 0) {
      snprintf(val, sizeof(val), "%d", topk);
      setenv("TOPK", val, 1);
    }
    char ccap[20], ceb[20], cdb[20];
    snprintf(ccap, sizeof(ccap), "%d", cap);
    snprintf(ceb, sizeof(ceb), "%d", ebits);
    snprintf(cdb, sizeof(cdb), "%d", dbits);
    execl(engine, engine, ccap, ceb, cdb, (char *)NULL);
    perror(engine);
    return 2;
  }
  if (!strcmp(cmd, "chat"))
    return coli_chat_run(engine, model, cap, ebits, dbits, ngen, ctx);
  if (!strcmp(cmd, "serve") || !strcmp(cmd, "web")) {
    if (port < 1 || port > 65535 || cap < 1 || ngen < 1 || max_queue < 0 ||
        timeout <= 0 || kv_slots < 1 || kv_slots > 16) {
      fprintf(stderr, "invalid serve configuration\n");
      return 2;
    }
    if (strcmp(host, "127.0.0.1") && strcmp(host, "localhost") &&
        strcmp(host, "::1") && (!api_key || !*api_key))
      fprintf(
          stderr,
          "WARNING: API is listening beyond localhost without COLI_API_KEY\n");
    char pidpath[PATH_MAX];
    int have_pid = !pidfile_path(port, pidpath);
    if (have_pid) {
      FILE *f = fopen(pidpath, "w");
      if (f) {
        fchmod(fileno(f), 0600);
        fprintf(f, "%ld %s\n", (long)getpid(), model);
        fclose(f);
      } else
        have_pid = 0;
    }
    coli_server_config c = {.model = model,
                            .engine = engine,
                            .host = host,
                            .model_id = model_id,
                            .api_key = api_key,
                            .web_root = web_root,
                            .model_aliases = aliases,
                            .model_alias_count = alias_n,
                            .hidden_model_aliases = hidden_aliases,
                            .hidden_model_alias_count = hidden_n,
                            .cors_origins = origins,
                            .cors_origin_count = origin_n,
                            .port = port,
                            .cap = cap,
                            .max_tokens = ngen,
                            .max_queue = max_queue,
                            .queue_timeout = timeout,
                            .kv_slots = kv_slots,
                            .expert_bits = ebits,
                            .dense_bits = dbits,
                            .context_length = ctx,
                            .default_thinking = thinking};
    int rc = coli_server_run(&c);
    if (have_pid)
      unlink(pidpath);
    return rc;
  }
  fprintf(stderr, "unsupported native command: %s\n", cmd);
  usage(stderr);
  return 2;
}
