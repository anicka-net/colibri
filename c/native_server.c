#define _GNU_SOURCE
#include "native_server.h"
#include "json.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <netdb.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define HTTP_MAX_BODY (4u << 20)
#define HTTP_MAX_HEADERS (64u << 10)
#define ENGINE_DATA_MAX 65536
#define PROFILE_TURNS 120

typedef struct {
  char *p;
  size_t n, cap;
} buf;

static void b_reserve(buf *b, size_t add) {
  if (b->n + add + 1 <= b->cap)
    return;
  size_t cap = b->cap ? b->cap : 256;
  while (cap < b->n + add + 1)
    cap *= 2;
  b->p = (char *)realloc(b->p, cap);
  b->cap = cap;
}
static void b_addn(buf *b, const void *p, size_t n) {
  b_reserve(b, n);
  memcpy(b->p + b->n, p, n);
  b->n += n;
  b->p[b->n] = 0;
}
static void b_add(buf *b, const char *s) { b_addn(b, s, strlen(s)); }
static void b_printf(buf *b, const char *fmt, ...) {
  va_list ap, cp;
  va_start(ap, fmt);
  va_copy(cp, ap);
  int n = vsnprintf(NULL, 0, fmt, cp);
  va_end(cp);
  if (n > 0) {
    b_reserve(b, (size_t)n);
    vsnprintf(b->p + b->n, b->cap - b->n, fmt, ap);
    b->n += (size_t)n;
  }
  va_end(ap);
}
static void b_json(buf *b, const char *s) {
  b_addn(b, "\"", 1);
  for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
    switch (*p) {
    case '"':
      b_add(b, "\\\"");
      break;
    case '\\':
      b_add(b, "\\\\");
      break;
    case '\b':
      b_add(b, "\\b");
      break;
    case '\f':
      b_add(b, "\\f");
      break;
    case '\n':
      b_add(b, "\\n");
      break;
    case '\r':
      b_add(b, "\\r");
      break;
    case '\t':
      b_add(b, "\\t");
      break;
    default:
      if (*p < 0x20)
        b_printf(b, "\\u%04x", *p);
      else
        b_addn(b, p, 1);
    }
  }
  b_addn(b, "\"", 1);
}
static void b_jval(buf *b, jval *v) {
  if (!v) {
    b_add(b, "null");
    return;
  }
  switch (v->t) {
  case J_NULL:
    b_add(b, "null");
    break;
  case J_BOOL:
    b_add(b, v->boolean ? "true" : "false");
    break;
  case J_NUM:
    b_printf(b, "%.17g", v->num);
    break;
  case J_STR:
    b_json(b, v->str);
    break;
  case J_ARR:
    b_add(b, "[");
    for (int i = 0; i < v->len; i++) {
      if (i)
        b_add(b, ",");
      b_jval(b, v->kids[i]);
    }
    b_add(b, "]");
    break;
  case J_OBJ:
    b_add(b, "{");
    for (int i = 0; i < v->len; i++) {
      if (i)
        b_add(b, ",");
      b_json(b, v->keys[i]);
      b_add(b, ":");
      b_jval(b, v->kids[i]);
    }
    b_add(b, "}");
    break;
  }
}
static void b_free(buf *b) {
  free(b->p);
  memset(b, 0, sizeof(*b));
}

static jval *jget(jval *o, const char *k) { return json_get(o, k); }
static const char *jstr(jval *o, const char *k, const char *d) {
  jval *v = jget(o, k);
  return v && v->t == J_STR ? v->str : d;
}
static int jbool(jval *o, const char *k, int d) {
  jval *v = jget(o, k);
  return v && v->t == J_BOOL ? v->boolean : d;
}
static double jnum(jval *o, const char *k, double d) {
  jval *v = jget(o, k);
  return v && v->t == J_NUM ? v->num : d;
}
static int valid_optional_type(jval *o, const char *k, jtype type) {
  jval *v = jget(o, k);
  return !v || v->t == type;
}
static int valid_positive_integer(jval *v) {
  return v && v->t == J_NUM && isfinite(v->num) && v->num >= 1 &&
         v->num == floor(v->num) && v->num <= INT_MAX;
}

typedef struct request request;
struct request {
  unsigned long long id;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  buf data;
  int done, error, cancelled;
  int completion_tokens, prompt_tokens, length_limited;
  double tps, hit, rss;
  request *next;
};
typedef struct {
  double wall, disk, wait, matmul, attention, lm;
  int prompt, completion, forwards;
} profile_turn;
typedef struct {
  pid_t pid;
  int in_fd, out_fd;
  FILE *out;
  pthread_t dispatcher;
  pthread_mutex_t write_mu, pending_mu;
  request *pending;
  unsigned long long next_id;
  int stopped;
  int tiers_vram, tiers_ram, tiers_disk;
  double tiers_vram_gb, tiers_ram_gb;
  int cores, gpus;
  double ram_total, ram_avail, vram_total;
  char cpu[192], gpu[192];
  profile_turn profile[PROFILE_TURNS];
  int profile_len, profile_at;
  unsigned long profile_seq;
} engine;
typedef struct response_entry response_entry;
struct response_entry {
  char id[80];
  char *prompt;
  size_t bytes;
  response_entry *next;
};

typedef struct {
  coli_server_config c;
  int listener;
  engine e;
  pthread_mutex_t sched_mu;
  pthread_cond_t sched_cv;
  int *slot_busy;
  int active, queued, watchdog_active;
  unsigned long admitted, completed, rejected, timed_out, cancelled;
  time_t created;
  unsigned long long model_size;
  time_t model_modified;
  pthread_mutex_t history_mu;
  response_entry *history_head, *history_tail;
  int history_count;
  size_t history_bytes;
  pthread_mutex_t clients_mu;
  pthread_cond_t clients_cv;
  int clients;
} server;
static volatile sig_atomic_t stopping;
static volatile sig_atomic_t signal_listener = -1;
static _Thread_local char response_origin[512];
static _Thread_local char response_path[1024];
static _Thread_local char response_request_id[96];

static void model_metadata(const char *path, unsigned long long *size,
                           time_t *modified) {
  struct stat st;
  if (lstat(path, &st))
    return;
  if (S_ISREG(st.st_mode)) {
    if (st.st_size > 0 && *size <= ULLONG_MAX - (unsigned long long)st.st_size)
      *size += (unsigned long long)st.st_size;
    if (st.st_mtime > *modified)
      *modified = st.st_mtime;
    return;
  }
  if (!S_ISDIR(st.st_mode))
    return;
  DIR *dir = opendir(path);
  if (!dir)
    return;
  struct dirent *entry;
  while ((entry = readdir(dir))) {
    if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..") ||
        !strncmp(entry->d_name, ".coli", 5))
      continue;
    char child[PATH_MAX];
    if (snprintf(child, sizeof(child), "%s/%s", path, entry->d_name) <
        (int)sizeof(child))
      model_metadata(child, size, modified);
  }
  closedir(dir);
}

static int http_debug_enabled(void) {
  const char *value = getenv("COLI_DEBUG_HTTP");
  return value && *value && strcmp(value, "0");
}

static ssize_t write_all(int fd, const void *p, size_t n) {
  const char *s = p;
  while (n) {
    ssize_t w = write(fd, s, n);
    if (w < 0 && errno == EINTR)
      continue;
    if (w <= 0)
      return -1;
    s += w;
    n -= (size_t)w;
  }
  return 0;
}
static int read_exact(FILE *f, void *p, size_t n) {
  return fread(p, 1, n, f) == n ? 0 : -1;
}
static request *pending_find(engine *e, unsigned long long id) {
  for (request *r = e->pending; r; r = r->next)
    if (r->id == id)
      return r;
  return NULL;
}
static request *pending_take(engine *e, unsigned long long id) {
  request **p = &e->pending;
  while (*p && (*p)->id != id)
    p = &(*p)->next;
  if (!*p)
    return NULL;
  request *r = *p;
  *p = r->next;
  r->next = NULL;
  return r;
}

static void request_signal(request *r, int error) {
  pthread_mutex_lock(&r->mu);
  r->error = error;
  r->done = 1;
  pthread_cond_broadcast(&r->cv);
  pthread_mutex_unlock(&r->mu);
}

static void *engine_dispatch(void *arg) {
  engine *e = arg;
  char *line = NULL;
  size_t cap = 0;
  while (getline(&line, &cap, e->out) >= 0) {
    unsigned long long id;
    int n;
    if (sscanf(line, "DATA %llu %d", &id, &n) == 2) {
      if (n < 0 || n > ENGINE_DATA_MAX)
        break;
      char data[ENGINE_DATA_MAX + 1];
      if (read_exact(e->out, data, (size_t)n) || fgetc(e->out) != '\n')
        break;
      pthread_mutex_lock(&e->pending_mu);
      request *r = pending_find(e, id);
      if (r) {
        pthread_mutex_lock(&r->mu);
        b_addn(&r->data, data, (size_t)n);
        pthread_cond_broadcast(&r->cv);
        pthread_mutex_unlock(&r->mu);
      }
      pthread_mutex_unlock(&e->pending_mu);
    } else if (!strncmp(line, "DONE ", 5)) {
      int ct = 0, pt = 0, ll = 0;
      double tps = 0, hit = 0, rss = 0;
      if (sscanf(line, "DONE %llu STAT %d %lf %lf %lf %d %d", &id, &ct, &tps,
                 &hit, &rss, &pt, &ll) < 5)
        break;
      pthread_mutex_lock(&e->pending_mu);
      request *r = pending_take(e, id);
      pthread_mutex_unlock(&e->pending_mu);
      if (r) {
        r->completion_tokens = ct;
        r->prompt_tokens = pt;
        r->length_limited = ll;
        r->tps = tps;
        r->hit = hit;
        r->rss = rss;
        request_signal(r, 0);
      }
    } else if (sscanf(line, "TIERS %d %d %d %lf %lf", &e->tiers_vram,
                      &e->tiers_ram, &e->tiers_disk, &e->tiers_vram_gb,
                      &e->tiers_ram_gb) == 5) {
    } else if (!strncmp(line, "HWINFO ", 7)) {
      char rest[400] = {0};
      sscanf(line, "HWINFO %d %lf %lf %d %lf %399[^\n]", &e->cores,
             &e->ram_total, &e->ram_avail, &e->gpus, &e->vram_total, rest);
      char *bar = strchr(rest, '|');
      if (bar) {
        *bar = 0;
        snprintf(e->cpu, sizeof(e->cpu), "%.191s", rest);
        snprintf(e->gpu, sizeof(e->gpu), "%.191s", bar + 1);
      }
    } else if (!strncmp(line, "PROF ", 5)) {
      profile_turn t = {0};
      if (sscanf(line, "PROF %lf %d %d %lf %lf %lf %lf %lf %d", &t.wall,
                 &t.prompt, &t.completion, &t.disk, &t.wait, &t.matmul,
                 &t.attention, &t.lm, &t.forwards) == 9) {
        e->profile[e->profile_at] = t;
        e->profile_at = (e->profile_at + 1) % PROFILE_TURNS;
        if (e->profile_len < PROFILE_TURNS)
          e->profile_len++;
        e->profile_seq++;
      }
    } else if (sscanf(line, "ERROR %llu", &id) == 1) {
      pthread_mutex_lock(&e->pending_mu);
      request *r = pending_take(e, id);
      pthread_mutex_unlock(&e->pending_mu);
      if (r)
        request_signal(r, 1);
    } /* EMAP/HITS are dashboard telemetry and may be ignored by API clients. */
  }
  free(line);
  e->stopped = 1;
  pthread_mutex_lock(&e->pending_mu);
  request *pending = e->pending;
  e->pending = NULL;
  for (request *r = pending; r;) {
    request *next = r->next;
    r->next = NULL;
    request_signal(r, 1);
    r = next;
  }
  pthread_mutex_unlock(&e->pending_mu);
  return NULL;
}

static int wait_ready(FILE *f) {
  static const char marker[] = "\1\1READY\1\1\n";
  size_t n = 0;
  int ch;
  while ((ch = fgetc(f)) != EOF) {
    n = (ch == marker[n]) ? n + 1 : (ch == marker[0] ? 1 : 0);
    if (n == sizeof(marker) - 1) {
      char line[256];
      return fgets(line, sizeof(line), f) ? 0 : -1;
    }
  }
  return -1;
}
static int engine_start(engine *e, const coli_server_config *c) {
  int in[2], out[2];
  if (pipe(in) || pipe(out))
    return -1;
  pid_t pid = fork();
  if (pid < 0)
    return -1;
  if (!pid) {
    dup2(in[0], 0);
    dup2(out[1], 1);
    close(in[0]);
    close(in[1]);
    close(out[0]);
    close(out[1]);
    setenv("SNAP", c->model, 1);
    setenv("SERVE", "1", 1);
    setenv("SERVE_BATCH", "1", 1);
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%d", c->max_tokens);
    setenv("NGEN", tmp, 1);
    snprintf(tmp, sizeof(tmp), "%d", c->kv_slots);
    setenv("KV_SLOTS", tmp, 1);
    snprintf(tmp, sizeof(tmp), "%d", c->context_length);
    setenv("CTX", tmp, 1);
    setenv("COLI_MODE", "serve", 1);
    /* A mux linked with libgomp may bind its initial thread before main when
       OMP_PROC_BIND is set. Services should start the mux with binding off
       and pass the desired engine-only policy through this variable. */
    const char *engine_bind = getenv("COLI_ENGINE_OMP_PROC_BIND");
    if (engine_bind && *engine_bind)
      setenv("OMP_PROC_BIND", engine_bind, 1);
    char cap[20], eb[20], db[20];
    snprintf(cap, sizeof(cap), "%d", c->cap);
    snprintf(eb, sizeof(eb), "%d", c->expert_bits);
    snprintf(db, sizeof(db), "%d", c->dense_bits);
    execl(c->engine, c->engine, cap, eb, db, (char *)NULL);
    _exit(127);
  }
  close(in[0]);
  close(out[1]);
  memset(e, 0, sizeof(*e));
  e->pid = pid;
  e->in_fd = in[1];
  e->out_fd = out[0];
  e->out = fdopen(out[0], "rb");
  e->next_id = 1;
  pthread_mutex_init(&e->write_mu, NULL);
  pthread_mutex_init(&e->pending_mu, NULL);
  if (!e->out || wait_ready(e->out)) {
    kill(pid, SIGTERM);
    return -1;
  }
  return pthread_create(&e->dispatcher, NULL, engine_dispatch, e);
}
static void engine_stop(engine *e) {
  if (!e->pid)
    return;
  e->stopped = 1;
  kill(e->pid, SIGTERM);
  int status;
  for (int i = 0; i < 50; i++) {
    if (waitpid(e->pid, &status, WNOHANG) == e->pid)
      goto done;
    usleep(100000);
  }
  kill(e->pid, SIGKILL);
  waitpid(e->pid, &status, 0);
done:
  close(e->in_fd);
  if (e->out)
    fclose(e->out);
  pthread_join(e->dispatcher, NULL);
}

typedef int (*engine_data_cb)(const char *data, size_t n, void *opaque);
static int engine_generate(engine *e, const char *prompt, int max, double temp,
                           double top_p, int slot, request *r,
                           engine_data_cb on_data, void *opaque) {
  memset(r, 0, sizeof(*r));
  pthread_mutex_init(&r->mu, NULL);
  pthread_cond_init(&r->cv, NULL);
  pthread_mutex_lock(&e->pending_mu);
  r->id = e->next_id++;
  r->next = e->pending;
  e->pending = r;
  pthread_mutex_unlock(&e->pending_mu);
  size_t n = strlen(prompt);
  char hdr[160];
  int hn = snprintf(hdr, sizeof(hdr), "SUBMIT %llu %d %zu %d %.8g %.8g\n",
                    r->id, slot, n, max, temp, top_p);
  pthread_mutex_lock(&e->write_mu);
  int bad = write_all(e->in_fd, hdr, (size_t)hn) ||
            write_all(e->in_fd, prompt, n) || write_all(e->in_fd, "\n", 1);
  pthread_mutex_unlock(&e->write_mu);
  if (bad) {
    pthread_mutex_lock(&e->pending_mu);
    pending_take(e, r->id);
    pthread_mutex_unlock(&e->pending_mu);
    return -1;
  }
  size_t emitted = 0;
  int cancel_sent = 0;
  pthread_mutex_lock(&r->mu);
  while (!r->done || emitted < r->data.n) {
    while (!r->done && emitted == r->data.n)
      pthread_cond_wait(&r->cv, &r->mu);
    if (emitted < r->data.n) {
      size_t n = r->data.n - emitted;
      char *chunk = malloc(n);
      memcpy(chunk, r->data.p + emitted, n);
      emitted += n;
      pthread_mutex_unlock(&r->mu);
      int cancel = on_data ? on_data(chunk, n, opaque) : 0;
      free(chunk);
      if (cancel && !cancel_sent) {
        char line[80];
        int ln = snprintf(line, sizeof(line), "CANCEL %llu\n", r->id);
        pthread_mutex_lock(&e->write_mu);
        write_all(e->in_fd, line, (size_t)ln);
        pthread_mutex_unlock(&e->write_mu);
        cancel_sent = 1;
      }
      pthread_mutex_lock(&r->mu);
    }
  }
  pthread_mutex_unlock(&r->mu);
  return r->error ? -1 : 0;
}

typedef struct {
  char method[12], path[1024], authorization[1024], api_key[512], origin[512];
  size_t content_length;
  char *body;
  int watchdog;
} http_req;
static int sock_read_headers(int fd, buf *b) {
  char tmp[4096];
  while (b->n < HTTP_MAX_HEADERS) {
    ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
    if (n <= 0)
      return -1;
    b_addn(b, tmp, (size_t)n);
    char *p = strstr(b->p, "\r\n\r\n");
    if (p)
      return (int)(p + 4 - b->p);
  }
  return -1;
}
static int parse_request(int fd, http_req *r) {
  buf raw = {0};
  int off = sock_read_headers(fd, &raw);
  if (off < 0) {
    b_free(&raw);
    return -1;
  }
  /* strtok skips runs of CR/LF and would otherwise continue into a body that
   * arrived in the same recv(), replacing JSON colons with NULs.  Isolate
   * the header block before tokenizing it. */
  size_t buffered_body = raw.n > (size_t)off ? raw.n - (size_t)off : 0;
  char *early = buffered_body ? malloc(buffered_body) : NULL;
  if (early)
    memcpy(early, raw.p + off, buffered_body);
  raw.p[off - 2] = 0;
  char *save = NULL;
  char *line = strtok_r(raw.p, "\r\n", &save);
  if (!line || sscanf(line, "%11s %1023s", r->method, r->path) != 2) {
    free(early);
    b_free(&raw);
    return -1;
  }
  while ((line = strtok_r(NULL, "\r\n", &save))) {
    char *colon = strchr(line, ':');
    if (!colon)
      continue;
    *colon++ = 0;
    while (*colon == ' ' || *colon == '\t')
      colon++;
    if (!strcasecmp(line, "Content-Length"))
      r->content_length = (size_t)strtoull(colon, NULL, 10);
    else if (!strcasecmp(line, "Authorization"))
      snprintf(r->authorization, sizeof(r->authorization), "%s", colon);
    else if (!strcasecmp(line, "x-api-key"))
      snprintf(r->api_key, sizeof(r->api_key), "%s", colon);
    else if (!strcasecmp(line, "Origin"))
      snprintf(r->origin, sizeof(r->origin), "%s", colon);
    else if (!strcasecmp(line, "X-Colibri-Watchdog"))
      r->watchdog = atoi(colon) == 1;
  }
  if (r->content_length > HTTP_MAX_BODY) {
    free(early);
    b_free(&raw);
    return -2;
  }
  r->body = calloc(r->content_length + 1, 1);
  size_t have = buffered_body;
  if (have > r->content_length)
    have = r->content_length;
  if (have)
    memcpy(r->body, early, have);
  free(early);
  b_free(&raw);
  while (have < r->content_length) {
    ssize_t n = recv(fd, r->body + have, r->content_length - have, 0);
    if (n <= 0)
      return -1;
    have += (size_t)n;
  }
  return 0;
}
static const char *status_text(int s) {
  switch (s) {
  case 200:
    return "OK";
  case 204:
    return "No Content";
  case 400:
    return "Bad Request";
  case 401:
    return "Unauthorized";
  case 404:
    return "Not Found";
  case 405:
    return "Method Not Allowed";
  case 429:
    return "Too Many Requests";
  default:
    return "Internal Server Error";
  }
}
static int send_head(int fd, int status, const char *type, size_t n,
                     const char *extra) {
  buf h = {0};
  b_printf(&h,
           "HTTP/1.1 %d %s\r\nServer: colibri\r\nContent-Type: "
           "%s\r\nContent-Length: %zu\r\nConnection: close\r\n",
           status, status_text(status), type, n);
  if (response_origin[0])
    b_printf(&h,
             "Access-Control-Allow-Origin: %s\r\nVary: "
             "Origin\r\nAccess-Control-Expose-Headers: x-request-id, "
             "x-colibri-queue-wait-ms, Retry-After\r\n",
             response_origin);
  if (response_request_id[0])
    b_printf(&h, "x-request-id: %s\r\nrequest-id: %s\r\n", response_request_id,
             response_request_id);
  if (extra)
    b_add(&h, extra);
  b_add(&h, "\r\n");
  int rc = write_all(fd, h.p, h.n);
  b_free(&h);
  return rc;
}
static int send_body(int fd, int status, const char *type, const char *p,
                     size_t n, const char *extra) {
  return send_head(fd, status, type, n, extra) || write_all(fd, p, n);
}
static int send_stream_head(int fd, const char *type) {
  buf h = {0};
  b_printf(&h,
           "HTTP/1.1 200 OK\r\nServer: colibri\r\nContent-Type: %s\r\n"
           "Cache-Control: no-cache\r\nConnection: close\r\n",
           type);
  if (response_origin[0])
    b_printf(&h, "Access-Control-Allow-Origin: %s\r\nVary: Origin\r\n",
             response_origin);
  if (response_request_id[0])
    b_printf(&h, "x-request-id: %s\r\nrequest-id: %s\r\n", response_request_id,
             response_request_id);
  b_add(&h, "\r\n");
  int rc = write_all(fd, h.p, h.n);
  b_free(&h);
  return rc;
}
static int send_json(int fd, int status, buf *b) {
  return send_body(fd, status, "application/json", b->p ? b->p : "{}",
                   b->p ? b->n : 2, NULL);
}
static void api_error(int fd, int status, const char *message) {
  if (http_debug_enabled())
    fprintf(stderr, "[http] id=%s path=%s status=%d error=%s\n",
            response_request_id, response_path, status, message);
  buf b = {0};
  if (!strncmp(response_path, "/v1/messages", 12)) {
    b_add(&b, "{\"type\":\"error\",\"error\":{\"type\":\"");
    b_add(&b, status == 401   ? "authentication_error"
              : status >= 500 ? "api_error"
                              : "invalid_request_error");
    b_add(&b, "\",\"message\":");
    b_json(&b, message);
    b_add(&b, "},\"request_id\":");
    b_json(&b, response_request_id);
    b_add(&b, "}");
  } else if (!strncmp(response_path, "/api/", 5)) {
    b_add(&b, "{\"error\":");
    b_json(&b, message);
    b_add(&b, "}");
  } else {
    b_add(&b, "{\"error\":{\"message\":");
    b_json(&b, message);
    b_add(&b,
          ",\"type\":\"invalid_request_error\",\"param\":null,\"code\":null}}");
  }
  send_json(fd, status, &b);
  b_free(&b);
}

static int auth_ok(server *s, http_req *r) {
  if (!s->c.api_key || !*s->c.api_key)
    return 1;
  if (!strcmp(r->api_key, s->c.api_key))
    return 1;
  char want[600];
  snprintf(want, sizeof(want), "Bearer %s", s->c.api_key);
  return !strcmp(r->authorization, want);
}
static int schedule_enter(server *s, int desired, int *slot) {
  struct timespec until;
  clock_gettime(CLOCK_REALTIME, &until);
  long ns = (long)((s->c.queue_timeout - floor(s->c.queue_timeout)) * 1e9);
  until.tv_sec +=
      (time_t)s->c.queue_timeout + (until.tv_nsec + ns) / 1000000000;
  until.tv_nsec = (until.tv_nsec + ns) % 1000000000;
  pthread_mutex_lock(&s->sched_mu);
  int available = -1;
  if (desired >= 0) {
    if (!s->slot_busy[desired])
      available = desired;
  } else
    for (int i = 0; i < s->c.kv_slots; i++)
      if (!s->slot_busy[i]) {
        available = i;
        break;
      }
  if (available < 0 && s->queued >= s->c.max_queue) {
    s->rejected++;
    pthread_mutex_unlock(&s->sched_mu);
    return -1;
  }
  s->queued++;
  while (available < 0 && !stopping) {
    if (pthread_cond_timedwait(&s->sched_cv, &s->sched_mu, &until) ==
        ETIMEDOUT) {
      s->queued--;
      s->timed_out++;
      pthread_mutex_unlock(&s->sched_mu);
      return -2;
    }
    if (desired >= 0) {
      if (!s->slot_busy[desired])
        available = desired;
    } else
      for (int i = 0; i < s->c.kv_slots; i++)
        if (!s->slot_busy[i]) {
          available = i;
          break;
        }
  }
  s->queued--;
  if (available < 0) {
    pthread_mutex_unlock(&s->sched_mu);
    return -3;
  }
  s->slot_busy[available] = 1;
  s->active++;
  s->admitted++;
  *slot = available;
  pthread_mutex_unlock(&s->sched_mu);
  return 0;
}
static void schedule_leave(server *s, int slot) {
  pthread_mutex_lock(&s->sched_mu);
  s->slot_busy[slot] = 0;
  s->active--;
  s->completed++;
  pthread_cond_broadcast(&s->sched_cv);
  pthread_mutex_unlock(&s->sched_mu);
}

static int render_content(buf *out, jval *v) {
  if (!v || v->t == J_NULL)
    return 0;
  if (v->t == J_STR) {
    b_add(out, v->str);
    return 0;
  }
  if (v->t != J_ARR)
    return -1;
  for (int i = 0; i < v->len; i++) {
    jval *p = v->kids[i];
    jval *t = jget(p, "type"), *x = jget(p, "text");
    if (!p || p->t != J_OBJ || !t || t->t != J_STR ||
        (strcmp(t->str, "text") && strcmp(t->str, "input_text")) || !x ||
        x->t != J_STR)
      return -1;
    b_add(out, x->str);
  }
  return 0;
}

/* Claude Code currently prepends transport/accounting metadata to the first
   system block. It is useful to the API boundary but meaningless to the model,
   and its per-request cch value destroys exact-prefix KV reuse. Keep the JSON
   body untouched; strip only recognized leading transport headers while
   rendering the model prompt. */
static const char *strip_anthropic_headers(const char *s) {
  if (!s)
    return "";
  if (strncasecmp(s, "x-anthropic-billing-header:", 27))
    return s;
  const char *cch = strstr(s + 27, "cch=");
  const char *end = cch ? strchr(cch, ';') : NULL;
  if (!end)
    return s;
  s = end + 1;
  for (;;) {
    while (*s == '\r' || *s == '\n')
      s++;
    const char *colon = strchr(s, ':'), *newline = strchr(s, '\n');
    if (!colon || !newline || colon > newline)
      break;
    size_t name = (size_t)(colon - s);
    int transport = (name >= 2 && !strncasecmp(s, "x-", 2)) ||
                    (name >= 10 && !strncasecmp(s, "anthropic-", 10)) ||
                    (name == 13 && !strncasecmp(s, "authorization", 13)) ||
                    (name == 12 && !strncasecmp(s, "content-type", 12)) ||
                    (name == 10 && !strncasecmp(s, "user-agent", 10));
    if (!transport)
      break;
    s = newline + 1;
  }
  return s;
}

static int render_anthropic_system(buf *out, jval *v) {
  if (!v || v->t == J_NULL)
    return 0;
  if (v->t == J_STR) {
    b_add(out, strip_anthropic_headers(v->str));
    return 0;
  }
  if (v->t != J_ARR)
    return -1;
  for (int i = 0; i < v->len; i++) {
    jval *p = v->kids[i];
    jval *t = jget(p, "type"), *x = jget(p, "text");
    if (!p || p->t != J_OBJ || !t || t->t != J_STR ||
        (strcmp(t->str, "text") && strcmp(t->str, "input_text")) || !x ||
        x->t != J_STR)
      return -1;
    b_add(out, i == 0 ? strip_anthropic_headers(x->str) : x->str);
  }
  return 0;
}
static jval *tool_function(jval *t) {
  jval *f = jget(t, "function");
  return f && f->t == J_OBJ ? f : t;
}
static void render_tool_definition(buf *out, jval *tool) {
  jval *f = tool_function(tool);
  jval *schema = jget(f, "input_schema");
  if (schema) {
    /* Anthropic calls this field input_schema; the model's native tool
     * prompt and the OpenAI/Responses dialects call it parameters. */
    b_add(out, "{\"name\":");
    b_json(out, jstr(f, "name", ""));
    const char *description = jstr(f, "description", NULL);
    if (description) {
      b_add(out, ",\"description\":");
      b_json(out, description);
    }
    b_add(out, ",\"parameters\":");
    b_jval(out, schema);
    b_add(out, "}");
  } else
    b_jval(out, f);
}
static jval *find_tool(jval *tools, const char *name) {
  if (!tools || tools->t != J_ARR || !name)
    return NULL;
  for (int i = 0; i < tools->len; i++)
    if (!strcmp(jstr(tool_function(tools->kids[i]), "name", ""), name))
      return tools->kids[i];
  return NULL;
}
static void render_tools(buf *out, jval *tools) {
  if (!tools || tools->t != J_ARR || !tools->len)
    return;
  int included = 0;
  for (int i = 0; i < tools->len; i++) {
    jval *defer = jget(tools->kids[i], "defer_loading");
    if (!defer || defer->t != J_BOOL || !defer->boolean)
      included++;
  }
  if (!included)
    return;
  b_add(out, "<|system|>\n# Tools\n\nYou may call one or more functions to "
             "assist with the user query.\n\nYou are provided with function "
             "signatures within <tools></tools> XML tags:\n<tools>\n");
  for (int i = 0; i < tools->len; i++) {
    jval *defer = jget(tools->kids[i], "defer_loading");
    if (defer && defer->t == J_BOOL && defer->boolean)
      continue;
    render_tool_definition(out, tools->kids[i]);
    b_add(out, "\n");
  }
  b_add(out,
        "</tools>\n\nFor each function call, output the function name and "
        "arguments within the following XML "
        "format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</"
        "arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</"
        "arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>");
}
static int render_chat(buf *out, jval *body, int thinking) {
  jval *msgs = jget(body, "messages");
  if (!msgs || msgs->t != J_ARR || !msgs->len)
    return -1;
  b_add(out, "[gMASK]<sop>");
  jval *tools = jget(body, "tools");
  if (!tools)
    tools = jget(body, "functions");
  const char *choice = jstr(body, "tool_choice", "auto");
  if (strcmp(choice, "none"))
    render_tools(out, tools);
  int prev_tool = 0;
  for (int i = 0; i < msgs->len; i++) {
    jval *m = msgs->kids[i], *role = jget(m, "role"),
         *content = jget(m, "content");
    if (!m || m->t != J_OBJ || !role || role->t != J_STR)
      return -1;
    if (!strcmp(role->str, "developer"))
      b_add(out, "<|system|>");
    else if (!strcmp(role->str, "system"))
      b_add(out, "<|system|>");
    else if (!strcmp(role->str, "user"))
      b_add(out, "<|user|>");
    else if (!strcmp(role->str, "assistant")) {
      b_add(out, "<|assistant|><think></think>");
    } else if (!strcmp(role->str, "tool")) {
      if (!prev_tool)
        b_add(out, "<|observation|>");
      b_add(out, "<tool_response>");
    } else
      return -1;
    if (render_content(out, content))
      return -1;
    if (!strcmp(role->str, "assistant")) {
      jval *calls = jget(m, "tool_calls");
      if (calls && calls->t == J_ARR)
        for (int k = 0; k < calls->len; k++) {
          jval *f = tool_function(calls->kids[k]);
          b_add(out, "<tool_call>");
          b_add(out, jstr(f, "name", ""));
          jval *args = jget(f, "arguments");
          jval *parsed = NULL;
          if (args && args->t == J_STR)
            parsed = json_parse(args->str, NULL);
          else
            parsed = args;
          if (parsed && parsed->t == J_OBJ)
            for (int q = 0; q < parsed->len; q++) {
              b_add(out, "<arg_key>");
              b_add(out, parsed->keys[q]);
              b_add(out, "</arg_key><arg_value>");
              if (parsed->kids[q]->t == J_STR)
                b_add(out, parsed->kids[q]->str);
              else
                b_jval(out, parsed->kids[q]);
              b_add(out, "</arg_value>");
            }
          if (args && args->t == J_STR)
            json_free(parsed);
          b_add(out, "</tool_call>");
        }
    }
    if (!strcmp(role->str, "tool"))
      b_add(out, "</tool_response>");
    prev_tool = !strcmp(role->str, "tool");
  }
  b_add(out,
        thinking ? "<|assistant|><think>" : "<|assistant|><think></think>");
  return 0;
}

static void health(server *s, int fd) {
  engine *e = &s->e;
  buf b = {0};
  pthread_mutex_lock(&s->sched_mu);
  b_printf(&b,
           "{\"status\":\"ok\",\"scheduler\":{\"active\":%d,\"queued\":%d,"
           "\"capacity\":%d,\"max_queue\":%d,\"queue_timeout_seconds\":%.3g,"
           "\"admitted\":%lu,\"completed\":%lu,\"rejected\":%lu,\"timed_out\":%"
           "lu,\"cancelled\":%lu},\"kv_slots\":%d,\"watchdog_active\":%d",
           s->active, s->queued, s->c.kv_slots, s->c.max_queue,
           s->c.queue_timeout, s->admitted, s->completed, s->rejected,
           s->timed_out, s->cancelled, s->c.kv_slots, s->watchdog_active);
  pthread_mutex_unlock(&s->sched_mu);
  b_printf(&b,
           ",\"tiers\":{\"vram\":%d,\"ram\":%d,\"disk\":%d,\"vram_gb\":%.2f,"
           "\"ram_gb\":%.2f},\"hwinfo\":{\"cores\":%d,\"ram_total_gb\":%.1f,"
           "\"ram_avail_gb\":%.1f,\"gpus\":%d,\"vram_total_gb\":%.1f,\"cpu\":",
           e->tiers_vram, e->tiers_ram, e->tiers_disk, e->tiers_vram_gb,
           e->tiers_ram_gb, e->cores, e->ram_total, e->ram_avail, e->gpus,
           e->vram_total);
  b_json(&b, e->cpu);
  b_add(&b, ",\"gpu\":");
  b_json(&b, e->gpu);
  b_add(&b, "}}");
  send_json(fd, 200, &b);
  b_free(&b);
}
static int model_allowed(server *s, const char *model);
static void model_item(server *s, buf *b, const char *id) {
  b_add(b, "{\"id\":");
  b_json(b, id);
  b_printf(b,
           ",\"object\":\"model\",\"type\":\"model\",\"created\":%lld,\"owned_"
           "by\":\"colibri\",\"display_name\":\"GLM-5.2 (Colibri)\"",
           (long long)s->created);
  if (s->c.context_length)
    b_printf(b,
             ",\"context_length\":%d,\"top_provider\":{\"context_length\":%d,"
             "\"max_completion_tokens\":%d,\"is_moderated\":false},\"supported_"
             "parameters\":[\"tools\",\"tool_choice\",\"max_tokens\","
             "\"temperature\",\"top_p\",\"stream\",\"reasoning_effort\"]",
             s->c.context_length, s->c.context_length, s->c.max_tokens);
  b_add(b, "}");
}
static void models(server *s, int fd) {
  buf b = {0};
  b_add(&b, "{\"object\":\"list\",\"data\":[");
  model_item(s, &b, s->c.model_id);
  for (int i = 0; i < s->c.model_alias_count; i++) {
    b_add(&b, ",");
    model_item(s, &b, s->c.model_aliases[i]);
  }
  b_add(&b, "],\"has_more\":false}");
  send_json(fd, 200, &b);
  b_free(&b);
}
static void model_get(server *s, int fd, const char *id) {
  if (!model_allowed(s, id)) {
    api_error(fd, 404, "The requested model is not available.");
    return;
  }
  buf b = {0};
  model_item(s, &b, id);
  send_json(fd, 200, &b);
  b_free(&b);
}
static void profile(server *s, int fd) {
  engine *e = &s->e;
  buf b = {0};
  b_printf(&b, "{\"seq\":%lu,\"turns\":[", e->profile_seq);
  for (int i = 0; i < e->profile_len; i++) {
    int at =
        (e->profile_at - e->profile_len + i + PROFILE_TURNS) % PROFILE_TURNS;
    profile_turn *t = &e->profile[at];
    if (i)
      b_add(&b, ",");
    b_printf(
        &b,
        "{\"wall_s\":%.6g,\"prompt_tokens\":%d,\"completion_tokens\":%d,"
        "\"expert_disk_s\":%.6g,\"expert_wait_s\":%.6g,\"expert_matmul_s\":%."
        "6g,\"attention_s\":%.6g,\"lm_head_s\":%.6g,\"forwards\":%d}",
        t->wall, t->prompt, t->completion, t->disk, t->wait, t->matmul,
        t->attention, t->lm, t->forwards);
  }
  b_add(&b, "]}");
  send_json(fd, 200, &b);
  b_free(&b);
}

static int model_allowed(server *s, const char *model) {
  if (!strcmp(model, s->c.model_id))
    return 1;
  for (int i = 0; i < s->c.model_alias_count; i++)
    if (!strcmp(model, s->c.model_aliases[i]))
      return 1;
  for (int i = 0; i < s->c.hidden_model_alias_count; i++)
    if (!strcmp(model, s->c.hidden_model_aliases[i]))
      return 1;
  return 0;
}
static int generate_prompt_cb(server *s, const char *prompt, int maximum,
                              double temp, double top_p, int desired,
                              request *r, engine_data_cb on_data,
                              void *opaque) {
  int slot, ad = schedule_enter(s, desired, &slot);
  if (ad)
    return ad;
  int rc = engine_generate(&s->e, prompt, maximum, temp, top_p, slot, r,
                           on_data, opaque);
  schedule_leave(s, slot);
  return rc ? -4 : 0;
}
static int generate_prompt(server *s, const char *prompt, int maximum,
                           double temp, double top_p, int desired, request *r) {
  return generate_prompt_cb(s, prompt, maximum, temp, top_p, desired, r, NULL,
                            NULL);
}
static void request_destroy(request *r) {
  b_free(&r->data);
  pthread_cond_destroy(&r->cv);
  pthread_mutex_destroy(&r->mu);
}
static char *history_get(server *s, const char *id) {
  if (!id)
    return NULL;
  pthread_mutex_lock(&s->history_mu);
  response_entry *e = s->history_head;
  while (e && strcmp(e->id, id))
    e = e->next;
  char *out = e ? strdup(e->prompt) : NULL;
  pthread_mutex_unlock(&s->history_mu);
  return out;
}
static void history_put(server *s, const char *id, const char *prompt) {
  response_entry *e = calloc(1, sizeof(*e));
  snprintf(e->id, sizeof(e->id), "%s", id);
  e->prompt = strdup(prompt);
  e->bytes = strlen(prompt);
  pthread_mutex_lock(&s->history_mu);
  if (s->history_tail)
    s->history_tail->next = e;
  else
    s->history_head = e;
  s->history_tail = e;
  s->history_count++;
  s->history_bytes += e->bytes;
  while (s->history_count > 32 || s->history_bytes > (16u << 20)) {
    response_entry *old = s->history_head;
    s->history_head = old->next;
    if (!s->history_head)
      s->history_tail = NULL;
    s->history_count--;
    s->history_bytes -= old->bytes;
    free(old->prompt);
    free(old);
  }
  pthread_mutex_unlock(&s->history_mu);
}
static void strip_tool_preamble(buf *p) {
  char *start = strstr(p->p ? p->p : "", "<|system|>\n# Tools");
  if (!start)
    return;
  char *user = strstr(start, "<|user|>");
  if (!user)
    return;
  size_t tail = strlen(user);
  memmove(start, user, tail + 1);
  p->n = (size_t)(start - p->p) + tail;
}
static void split_reasoning(const char *raw, int enabled, buf *reasoning,
                            buf *content) {
  if (!enabled) {
    b_add(content, raw);
    return;
  }
  const char *end = strstr(raw, "</think>");
  if (!end) {
    b_add(reasoning, raw);
    return;
  }
  b_addn(reasoning, raw, (size_t)(end - raw));
  end += 8;
  for (;;) {
    const char *extra = strstr(end, "</think>");
    if (!extra) {
      b_add(content, end);
      break;
    }
    b_addn(content, end, (size_t)(extra - end));
    end = extra + 8;
  }
}

typedef struct {
  char name[128], id[80];
  buf args;
} tool_call;
static const char *tool_param_type(jval *tools, const char *name,
                                   const char *key) {
  if (!tools || tools->t != J_ARR)
    return NULL;
  for (int i = 0; i < tools->len; i++) {
    jval *f = tool_function(tools->kids[i]);
    if (strcmp(jstr(f, "name", ""), name))
      continue;
    jval *schema = jget(f, "parameters");
    if (!schema)
      schema = jget(f, "input_schema");
    jval *p = jget(schema, "properties"), *spec = jget(p, key),
         *type = jget(spec, "type");
    if (type && type->t == J_STR)
      return type->str;
  }
  return NULL;
}
static void emit_tool_arg(buf *out, const char *raw, size_t n,
                          const char *declared) {
  char *value = strndup(raw, n);
  if (declared && !strcmp(declared, "string")) {
    b_json(out, value);
    free(value);
    return;
  }
  const char *p = value;
  while (isspace((unsigned char)*p))
    p++;
  int looks = *p == '{' || *p == '[' || *p == '"' || *p == '-' ||
              isdigit((unsigned char)*p) || !strncmp(p, "true", 4) ||
              !strncmp(p, "false", 5) || !strncmp(p, "null", 4);
  if (looks) {
    jval *v = json_parse(p, NULL);
    if (v) {
      b_jval(out, v);
      json_free(v);
      free(value);
      return;
    }
  }
  b_json(out, value);
  free(value);
}
static int tool_has_param(jval *tools, const char *name, const char *key) {
  if (!tools || tools->t != J_ARR)
    return 0;
  for (int i = 0; i < tools->len; i++) {
    jval *f = tool_function(tools->kids[i]);
    if (strcmp(jstr(f, "name", ""), name))
      continue;
    jval *schema = jget(f, "parameters");
    if (!schema)
      schema = jget(f, "input_schema");
    jval *p = jget(schema, "properties");
    return p && p->t == J_OBJ && jget(p, key);
  }
  return 0;
}
static const char *tool_single_required_param(jval *tools, const char *name) {
  jval *tool = find_tool(tools, name);
  if (!tool)
    return NULL;
  jval *f = tool_function(tool);
  jval *schema = jget(f, "parameters");
  if (!schema)
    schema = jget(f, "input_schema");
  jval *required = jget(schema, "required"), *properties = jget(schema, "properties");
  if (!required || required->t != J_ARR || required->len != 1 ||
      !required->kids[0] || required->kids[0]->t != J_STR ||
      !properties || properties->t != J_OBJ ||
      !jget(properties, required->kids[0]->str))
    return NULL;
  return required->kids[0]->str;
}
/* GLM occasionally uses its learned Python-like tool syntax even when the
 * prompt requests the XML argument dialect, for example:
 *
 *   <tool_call>WebSearch(**query**: "NE555 datasheet")
 *
 * It also commonly omits </tool_call>.  Accept one such call conservatively:
 * the name must be a tool offered by the client and every emitted argument
 * must exist in that tool's schema.  The caller stops at the first
 * unterminated call because text generated after the missing terminator is
 * not a trustworthy parallel-call request. */
static int parse_native_tool_call(const char *inner, const char *limit,
                                  jval *tools, tool_call *c) {
  memset(c, 0, sizeof(*c));
  const char *p = inner;
  while (p < limit && isspace((unsigned char)*p))
    p++;
  const char *ns = p;
  while (p < limit && (isalnum((unsigned char)*p) || *p == '_' || *p == '.' ||
                       *p == '-'))
    p++;
  size_t nn = (size_t)(p - ns);
  if (!nn || nn >= sizeof(c->name))
    return 0;
  memcpy(c->name, ns, nn);
  if (!find_tool(tools, c->name))
    return 0;
  while (p < limit && isspace((unsigned char)*p))
    p++;
  if (p == limit || *p != '(')
    return 0;
  p++;
  b_add(&c->args, "{");
  int argc = 0;
  while (p < limit) {
    while (p < limit && (isspace((unsigned char)*p) || *p == ','))
      p++;
    if (p == limit || *p == ')')
      break;
    if (p + 1 < limit && p[0] == '*' && p[1] == '*')
      p += 2;
    const char *ks = p;
    while (p < limit && (isalnum((unsigned char)*p) || *p == '_' || *p == '.' ||
                         *p == '-'))
      p++;
    if (p == ks)
      break;
    char *key = strndup(ks, (size_t)(p - ks));
    if (p + 1 < limit && p[0] == '*' && p[1] == '*')
      p += 2;
    while (p < limit && isspace((unsigned char)*p))
      p++;
    if (p == limit || (*p != ':' && *p != '=')) {
      free(key);
      break;
    }
    p++;
    while (p < limit && isspace((unsigned char)*p))
      p++;
    const char *vs = p;
    if (p < limit && *p == '"') {
      p++;
      while (p < limit) {
        if (*p == '\\' && p + 1 < limit) {
          p += 2;
          continue;
        }
        if (*p++ == '"')
          break;
      }
    } else if (p < limit && (*p == '[' || *p == '{')) {
      char open = *p++, close = open == '[' ? ']' : '}';
      int depth = 1, quoted = 0;
      while (p < limit && depth) {
        if (quoted) {
          if (*p == '\\' && p + 1 < limit)
            p += 2;
          else {
            if (*p == '"')
              quoted = 0;
            p++;
          }
        } else if (*p == '"')
          quoted = 1, p++;
        else {
          if (*p == open)
            depth++;
          else if (*p == close)
            depth--;
          p++;
        }
      }
    } else {
      while (p < limit && *p != ',' && *p != ')')
        p++;
      while (p > vs && isspace((unsigned char)p[-1]))
        p--;
    }
    if (p == vs) {
      free(key);
      break;
    }
    if (tool_has_param(tools, c->name, key)) {
      if (argc++)
        b_add(&c->args, ",");
      b_json(&c->args, key);
      b_add(&c->args, ":");
      if (*vs == '"') {
        char *quoted = strndup(vs, (size_t)(p - vs));
        jval *v = json_parse(quoted, NULL);
        if (v) {
          b_jval(&c->args, v);
          json_free(v);
        } else
          emit_tool_arg(&c->args, vs + 1, (size_t)(p - vs - 2), "string");
        free(quoted);
      } else
        emit_tool_arg(&c->args, vs, (size_t)(p - vs),
                      tool_param_type(tools, c->name, key));
    }
    free(key);
  }
  b_add(&c->args, "}");
  if (!argc) {
    b_free(&c->args);
    return 0;
  }
  snprintf(c->id, sizeof(c->id), "call_%ld_0", (long)time(NULL));
  return 1;
}
static int parse_tool_calls(const char *raw, jval *tools, buf *content,
                            tool_call *calls, int max_calls) {
  const char *p = raw, *box;
  int n = 0;
  while ((box = strstr(p, "<tool_call>"))) {
    b_addn(content, p, (size_t)(box - p));
    const char *inner = box + 11, *end = strstr(inner, "</tool_call>");
    if (!end) {
      const char *limit = strchr(inner, '\n');
      const char *next = strstr(inner, "<tool_call>");
      if (!limit || (next && next < limit))
        limit = next;
      if (!limit)
        limit = inner + strlen(inner);
      if (n < max_calls && parse_native_tool_call(inner, limit, tools, &calls[n]))
        n++;
      else
        b_add(content, box);
      return n;
    }
    if (n < max_calls) {
      tool_call *c = &calls[n];
      memset(c, 0, sizeof(*c));
      const char *q = inner;
      while (q < end && isspace((unsigned char)*q))
        q++;
      const char *name = q;
      while (q < end && (isalnum((unsigned char)*q) || *q == '_' || *q == '.' ||
                         *q == '-'))
        q++;
      size_t nn = (size_t)(q - name);
      if (nn >= sizeof(c->name))
        nn = sizeof(c->name) - 1;
      memcpy(c->name, name, nn);
      snprintf(c->id, sizeof(c->id), "call_%ld_%d", (long)time(NULL), n);
      b_add(&c->args, "{");
      int argc = 0;
      while (q < end) {
        const char *ko = strstr(q, "<arg_key>"),
                   *kc = ko ? strstr(ko + 9, "</arg_key>") : NULL,
                   *vo = kc ? strstr(kc + 10, "<arg_value>") : NULL,
                   *vc = vo ? strstr(vo + 11, "</arg_value>") : NULL;
        if (!ko || !kc || !vo || !vc || vc > end)
          break;
        if (argc++)
          b_add(&c->args, ",");
        char *key = strndup(ko + 9, (size_t)(kc - (ko + 9)));
        b_json(&c->args, key);
        b_add(&c->args, ":");
        emit_tool_arg(&c->args, vo + 11, (size_t)(vc - (vo + 11)),
                      tool_param_type(tools, c->name, key));
        free(key);
        q = vc + 12;
      }
      /* A quantized model can retain the value while corrupting the opening
       * argument tags, e.g. bash</arg_value>command</arg_value>. Recover only
       * when the schema makes the mapping unambiguous: exactly one required
       * parameter and exactly one orphaned value. */
      if (!argc) {
        while (q < end && isspace((unsigned char)*q))
          q++;
        static const char orphan[] = "</arg_value>";
        if ((size_t)(end - q) > sizeof(orphan) - 1 &&
            !memcmp(q, orphan, sizeof(orphan) - 1)) {
          const char *vs = q + sizeof(orphan) - 1;
          const char *vc = strstr(vs, orphan);
          const char *key = tool_single_required_param(tools, c->name);
          const char *extra = vc ? strstr(vc + sizeof(orphan) - 1, orphan) : NULL;
          if (vc && vc < end && key && (!extra || extra >= end)) {
            b_json(&c->args, key);
            b_add(&c->args, ":");
            emit_tool_arg(&c->args, vs, (size_t)(vc - vs),
                          tool_param_type(tools, c->name, key));
            argc = 1;
          }
        }
      }
      b_add(&c->args, "}");
      n++;
    }
    p = end + 12;
  }
  b_add(content, p);
  char *t;
  while ((t = strstr(content->p ? content->p : "", "<think>")))
    memmove(t, t + 7, strlen(t + 7) + 1), content->n -= 7;
  while ((t = strstr(content->p ? content->p : "", "</think>")))
    memmove(t, t + 8, strlen(t + 8) + 1), content->n -= 8;
  return n;
}
static void free_tool_calls(tool_call *calls, int n) {
  for (int i = 0; i < n; i++)
    b_free(&calls[i].args);
}

typedef enum { SEM_REASONING, SEM_TEXT, SEM_TOOL } semantic_kind;
typedef int (*semantic_emit_fn)(void *opaque, semantic_kind kind,
                                const char *data, size_t n,
                                const tool_call *call);
typedef struct {
  int thinking, failed, tool_count;
  jval *tools;
  buf pending, raw_debug;
  semantic_emit_fn emit;
  void *opaque;
} semantic_stream;

static void semantic_consume(semantic_stream *s, size_t n) {
  if (n >= s->pending.n) {
    b_free(&s->pending);
    return;
  }
  memmove(s->pending.p, s->pending.p + n, s->pending.n - n);
  s->pending.n -= n;
  s->pending.p[s->pending.n] = 0;
}
static size_t semantic_utf8_prefix(const char *p, size_t n) {
  size_t i = 0;
  while (i < n) {
    unsigned char c = (unsigned char)p[i];
    size_t need = c < 0x80 ? 1 : (c & 0xe0) == 0xc0 ? 2
                              : (c & 0xf0) == 0xe0   ? 3
                              : (c & 0xf8) == 0xf0   ? 4
                                                     : 1;
    if (i + need > n)
      break;
    int valid = 1;
    for (size_t j = 1; j < need; j++)
      if (((unsigned char)p[i + j] & 0xc0) != 0x80)
        valid = 0;
    if (!valid)
      need = 1;
    i += need;
  }
  return i;
}
static int semantic_emit_bytes(semantic_stream *s, semantic_kind kind,
                               size_t n) {
  if (!n)
    return 0;
  n = semantic_utf8_prefix(s->pending.p, n);
  if (!n)
    return 0;
  if (s->emit(s->opaque, kind, s->pending.p, n, NULL)) {
    s->failed = 1;
    return 1;
  }
  semantic_consume(s, n);
  return 0;
}
static size_t semantic_marker_keep(const buf *pending, const char *marker) {
  size_t ml = strlen(marker), max = pending->n < ml - 1 ? pending->n : ml - 1;
  for (size_t k = max; k; k--)
    if (!memcmp(pending->p + pending->n - k, marker, k))
      return k;
  return 0;
}
/* Incremental GLM output tokenizer. It keeps only a marker-length suffix of
 * ordinary text, so XML delimiters may be split across engine DATA frames.
 * Completed tool calls are parsed by the same schema-aware parser used for
 * non-streaming responses; unterminated native calls are handled at finish. */
static int semantic_feed(semantic_stream *s, const char *data, size_t n) {
  static const char think_end[] = "</think>";
  static const char tool_open[] = "<tool_call>";
  static const char tool_end[] = "</tool_call>";
  if (s->failed)
    return 1;
  if (http_debug_enabled())
    b_addn(&s->raw_debug, data, n);
  b_addn(&s->pending, data, n);
  for (;;) {
    if (s->thinking) {
      char *end = strstr(s->pending.p ? s->pending.p : "", think_end);
      if (!end) {
        size_t keep = semantic_marker_keep(&s->pending, think_end);
        return s->pending.n > keep
                   ? semantic_emit_bytes(s, SEM_REASONING, s->pending.n - keep)
                   : 0;
      }
      size_t before = (size_t)(end - s->pending.p);
      if (semantic_emit_bytes(s, SEM_REASONING, before))
        return 1;
      semantic_consume(s, sizeof(think_end) - 1);
      s->thinking = 0;
      continue;
    }
    if (s->pending.n >= 7 && !memcmp(s->pending.p, "<think>", 7)) {
      semantic_consume(s, 7);
      continue;
    }
    char *open = s->tools ? strstr(s->pending.p ? s->pending.p : "", tool_open)
                          : NULL;
    if (!open) {
      size_t keep = s->tools ? semantic_marker_keep(&s->pending, tool_open) : 0;
      return s->pending.n > keep
                 ? semantic_emit_bytes(s, SEM_TEXT, s->pending.n - keep)
                 : 0;
    }
    size_t before = (size_t)(open - s->pending.p);
    if (semantic_emit_bytes(s, SEM_TEXT, before))
      return 1;
    char *end = strstr(s->pending.p + sizeof(tool_open) - 1, tool_end);
    if (!end)
      return 0;
    size_t call_bytes = (size_t)(end - s->pending.p) + sizeof(tool_end) - 1;
    char *raw = strndup(s->pending.p, call_bytes);
    tool_call calls[16];
    buf ignored = {0};
    int count = parse_tool_calls(raw, s->tools, &ignored, calls, 16);
    b_free(&ignored);
    free(raw);
    if (!count) {
      if (semantic_emit_bytes(s, SEM_TEXT, call_bytes))
        return 1;
    } else {
      for (int i = 0; i < count; i++) {
        if (s->emit(s->opaque, SEM_TOOL, NULL, 0, &calls[i]))
          s->failed = 1;
        else
          s->tool_count++;
      }
      free_tool_calls(calls, count);
      semantic_consume(s, call_bytes);
      if (s->failed)
        return 1;
    }
  }
}
static int semantic_finish(semantic_stream *s) {
  if (http_debug_enabled() && s->raw_debug.p) {
    buf quoted = {0};
    b_json(&quoted, s->raw_debug.p);
    fprintf(stderr, "[http] raw_generation=%s\n", quoted.p);
    fflush(stderr);
    b_free(&quoted);
  }
  if (s->failed)
    return 1;
  if (s->thinking)
    return semantic_emit_bytes(s, SEM_REASONING, s->pending.n);
  if (s->tools && s->pending.n &&
      strstr(s->pending.p, "<tool_call>") == s->pending.p) {
    tool_call calls[16];
    buf ignored = {0};
    int count = parse_tool_calls(s->pending.p, s->tools, &ignored, calls, 16);
    b_free(&ignored);
    if (count) {
      for (int i = 0; i < count; i++) {
        if (s->emit(s->opaque, SEM_TOOL, NULL, 0, &calls[i]))
          s->failed = 1;
        else
          s->tool_count++;
      }
      free_tool_calls(calls, count);
      b_free(&s->pending);
      return s->failed;
    }
  }
  return semantic_emit_bytes(s, SEM_TEXT, s->pending.n);
}
static void semantic_free(semantic_stream *s) {
  b_free(&s->pending);
  b_free(&s->raw_debug);
}

static int render_anthropic(buf *out, jval *body, int thinking) {
  b_add(out, "[gMASK]<sop>");
  jval *system = jget(body, "system");
  if (system) {
    b_add(out, "<|system|>");
    if (render_anthropic_system(out, system))
      return -1;
  }
  render_tools(out, jget(body, "tools"));
  jval *msgs = jget(body, "messages");
  if (!msgs || msgs->t != J_ARR || !msgs->len)
    return -1;
  for (int i = 0; i < msgs->len; i++) {
    jval *m = msgs->kids[i], *role = jget(m, "role"),
         *content = jget(m, "content");
    if (!role || role->t != J_STR)
      return -1;
    if (!strcmp(role->str, "user"))
      b_add(out, "<|user|>");
    else if (!strcmp(role->str, "assistant")) {
      b_add(out, "<|assistant|><think></think>");
    } else
      return -1;
    if (content && content->t == J_ARR) {
      for (int k = 0; k < content->len; k++) {
        jval *p = content->kids[k];
        const char *t = jstr(p, "type", "");
        if (!strcmp(t, "text")) {
          jval *x = jget(p, "text");
          if (!x || x->t != J_STR)
            return -1;
          b_add(out, x->str);
        } else if (!strcmp(t, "tool_result")) {
          b_add(out, "<|observation|>");
          jval *x = jget(p, "content");
          if (x && x->t == J_ARR) {
            for (int q = 0; q < x->len; q++) {
              jval *block = x->kids[q];
              const char *block_type = jstr(block, "type", "");
              if (!strcmp(block_type, "tool_reference")) {
                jval *tool = find_tool(jget(body, "tools"),
                                       jstr(block, "tool_name", NULL));
                if (!tool)
                  return -1;
                b_add(out, "<tool_reference>");
                render_tool_definition(out, tool);
                b_add(out, "</tool_reference>");
              } else if (!strcmp(block_type, "text")) {
                jval *text = jget(block, "text");
                if (!text || text->t != J_STR)
                  return -1;
                b_add(out, text->str);
              } else
                return -1;
            }
          } else if (render_content(out, x))
            return -1;
        } else if (!strcmp(t, "thinking") || !strcmp(t, "redacted_thinking") ||
                   !strcmp(t, "tool_use"))
          continue;
        else
          return -1;
      }
    } else if (render_content(out, content))
      return -1;
  }
  b_add(out,
        thinking ? "<|assistant|><think>" : "<|assistant|><think></think>");
  return 0;
}

static int render_responses(buf *out, jval *body, int thinking) {
  b_add(out, "[gMASK]<sop>");
  jval *inst = jget(body, "instructions");
  if (inst) {
    b_add(out, "<|system|>");
    if (render_content(out, inst))
      return -1;
  }
  render_tools(out, jget(body, "tools"));
  jval *input = jget(body, "input");
  if (!input)
    return -1;
  if (input->t == J_STR) {
    b_add(out, "<|user|>");
    b_add(out, input->str);
  } else if (input->t == J_ARR) {
    for (int i = 0; i < input->len; i++) {
      jval *m = input->kids[i];
      const char *role = jstr(m, "role", NULL);
      const char *type = jstr(m, "type", "");
      if (!strcmp(type, "function_call_output")) {
        b_add(out, "<|observation|>");
        jval *x = jget(m, "output");
        if (render_content(out, x))
          return -1;
        continue;
      }
      if (!role)
        return -1;
      if (!strcmp(role, "system") || !strcmp(role, "developer"))
        b_add(out, "<|system|>");
      else if (!strcmp(role, "user"))
        b_add(out, "<|user|>");
      else if (!strcmp(role, "assistant"))
        b_add(out, "<|assistant|><think></think>");
      else
        return -1;
      if (render_content(out, jget(m, "content")))
        return -1;
    }
  } else
    return -1;
  b_add(out,
        thinking ? "<|assistant|><think>" : "<|assistant|><think></think>");
  return 0;
}

typedef struct {
  int fd, chat, failed;
  const char *id, *model;
  semantic_stream semantic;
} openai_stream;
static int openai_delta(openai_stream *c, const char *field, const char *text,
                        size_t n) {
  char *copy = strndup(text, n);
  buf event = {0};
  b_add(&event, "data: {\"id\":");
  b_json(&event, c->id);
  b_printf(&event, ",\"object\":\"%s\",\"created\":%lld,\"model\":",
           c->chat ? "chat.completion.chunk" : "text_completion",
           (long long)time(NULL));
  b_json(&event, c->model);
  if (c->chat) {
    b_add(&event, ",\"choices\":[{\"index\":0,\"delta\":{");
    b_json(&event, field);
    b_add(&event, ":");
    b_json(&event, copy);
    b_add(&event, "},\"logprobs\":null,\"finish_reason\":null}]}\n\n");
  } else {
    b_add(&event, ",\"choices\":[{\"index\":0,\"text\":");
    b_json(&event, copy);
    b_add(&event, ",\"logprobs\":null,\"finish_reason\":null}]}\n\n");
  }
  int rc = write_all(c->fd, event.p, event.n);
  b_free(&event);
  free(copy);
  if (rc)
    c->failed = 1;
  return rc;
}
static int openai_tool_delta(openai_stream *c, const tool_call *call,
                             int index) {
  buf event = {0};
  b_add(&event, "data: {\"id\":");
  b_json(&event, c->id);
  b_printf(&event,
           ",\"object\":\"chat.completion.chunk\",\"created\":%lld,\"model\":",
           (long long)time(NULL));
  b_json(&event, c->model);
  b_printf(&event,
           ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{"
           "\"index\":%d,\"id\":",
           index);
  b_json(&event, call->id);
  b_add(&event, ",\"type\":\"function\",\"function\":{\"name\":");
  b_json(&event, call->name);
  b_add(&event, ",\"arguments\":");
  b_json(&event, call->args.p ? call->args.p : "{}");
  b_add(&event,
        "}}]},\"logprobs\":null,\"finish_reason\":null}]}\n\n");
  int rc = write_all(c->fd, event.p, event.n);
  b_free(&event);
  if (rc)
    c->failed = 1;
  return rc;
}
static int openai_semantic_emit(void *opaque, semantic_kind kind,
                                const char *data, size_t n,
                                const tool_call *call) {
  openai_stream *c = opaque;
  if (c->failed)
    return 1;
  if (kind == SEM_TOOL)
    return c->chat ? openai_tool_delta(c, call, c->semantic.tool_count) : 0;
  return openai_delta(c,
                      kind == SEM_REASONING ? "reasoning_content"
                                            : (c->chat ? "content" : "text"),
                      data, n);
}
static int openai_on_data(const char *data, size_t n, void *opaque) {
  openai_stream *c = opaque;
  return semantic_feed(&c->semantic, data, n);
}

static void completion(server *s, int fd, http_req *hr, jval *body, int chat) {
  if (!valid_optional_type(body, "model", J_STR) ||
      !valid_optional_type(body, "stream", J_BOOL) ||
      !valid_optional_type(body, "temperature", J_NUM) ||
      !valid_optional_type(body, "top_p", J_NUM)) {
    api_error(fd, 400, "Invalid request option type.");
    return;
  }
  const char *model = jstr(body, "model", s->c.model_id);
  int model_ok = !strcmp(model, s->c.model_id);
  for (int i = 0; !model_ok && i < s->c.model_alias_count; i++)
    model_ok = !strcmp(model, s->c.model_aliases[i]);
  for (int i = 0; !model_ok && i < s->c.hidden_model_alias_count; i++)
    model_ok = !strcmp(model, s->c.hidden_model_aliases[i]);
  if (!model_ok) {
    api_error(fd, 404, "The requested model is not available.");
    return;
  }
  jval *nval = jget(body, "n");
  if (nval && (!valid_positive_integer(nval) || nval->num != 1)) {
    api_error(fd, 400, "Colibri currently supports `n=1` only.");
    return;
  }
  jval *stop = jget(body, "stop");
  if (stop && stop->t != J_NULL) {
    api_error(fd, 400, "Custom stop sequences are not supported yet.");
    return;
  }
  jval *stream_options = jget(body, "stream_options");
  if (stream_options && stream_options->t != J_OBJ) {
    api_error(fd, 400, "`stream_options` must be an object.");
    return;
  }
  jval *maxval = jget(body, "max_completion_tokens");
  if (!maxval)
    maxval = jget(body, "max_tokens");
  if (maxval && !valid_positive_integer(maxval)) {
    api_error(fd, 400, "`max_tokens` must be positive.");
    return;
  }
  int maximum = maxval ? (int)maxval->num : s->c.max_tokens;
  if (maximum > s->c.max_tokens)
    maximum = s->c.max_tokens;
  double temp = jnum(body, "temperature", .7), top_p = jnum(body, "top_p", .9);
  if (!isfinite(temp) || temp < 0 || temp > 2 || !isfinite(top_p) ||
      top_p <= 0 || top_p > 1) {
    api_error(fd, 400, "Invalid generation options.");
    return;
  }
  buf prompt = {0};
  int thinking = s->c.default_thinking;
  for (int i = 0; i < s->c.hidden_model_alias_count; i++)
    if (!strcmp(model, s->c.hidden_model_aliases[i]))
      thinking = 0;
  jval *effort = jget(body, "reasoning_effort");
  if (effort) {
    if (effort->t != J_STR) {
      b_free(&prompt);
      api_error(fd, 400, "Invalid reasoning effort.");
      return;
    }
    thinking = strcmp(effort->str, "none") != 0;
  }
  jval *enable = jget(body, "enable_thinking");
  if (!enable)
    enable = jget(body, "think");
  if (enable) {
    if (enable->t != J_BOOL) {
      b_free(&prompt);
      api_error(fd, 400, "Thinking control must be boolean.");
      return;
    }
    thinking = enable->boolean;
  }
  jval *thinking_obj = jget(body, "thinking");
  if (thinking_obj) {
    const char *type = jstr(thinking_obj, "type", "");
    if (thinking_obj->t != J_OBJ ||
        (strcmp(type, "enabled") && strcmp(type, "disabled"))) {
      b_free(&prompt);
      api_error(fd, 400, "`thinking.type` must be enabled or disabled.");
      return;
    }
    thinking = !strcmp(type, "enabled");
  }
  if (chat ? render_chat(&prompt, body, thinking)
           : (jget(body, "prompt") && jget(body, "prompt")->t == J_STR
                  ? (b_add(&prompt, jget(body, "prompt")->str), 0)
                  : -1)) {
    b_free(&prompt);
    api_error(fd, 400,
              chat ? "`messages` must be a non-empty text-message array."
                   : "`prompt` must be a string.");
    return;
  }
  if (!chat && prompt.n == 0) {
    b_free(&prompt);
    api_error(fd, 400, "`prompt` must not be empty.");
    return;
  }
  if (s->c.context_length) {
    size_t limit = (size_t)(s->c.context_length - maximum > 0
                                ? s->c.context_length - maximum
                                : 1) *
                   16;
    if (prompt.n > limit) {
      b_free(&prompt);
      api_error(fd, 400,
                "Rendered prompt exceeds the configured context window.");
      return;
    }
  }
  jval *slotv = jget(body, "cache_slot");
  int desired = -1;
  if (slotv) {
    if (slotv->t != J_NUM || slotv->num < 0 || slotv->num >= s->c.kv_slots ||
        floor(slotv->num) != slotv->num) {
      b_free(&prompt);
      api_error(fd, 400, "Invalid cache slot.");
      return;
    }
    desired = (int)slotv->num;
  }
  jval *tools = jget(body, "tools");
  if (!tools)
    tools = jget(body, "functions");
  int stream = jbool(body, "stream", 0);
  char rid[80];
  snprintf(rid, sizeof(rid), chat ? "chatcmpl-%lld-%lu" : "cmpl-%lld-%lu",
           (long long)time(NULL), (unsigned long)pthread_self());
  openai_stream stream_ctx = {.fd = fd,
                              .chat = chat,
                              .id = rid,
                              .model = s->c.model_id};
  stream_ctx.semantic.thinking = thinking;
  stream_ctx.semantic.tools =
      tools && tools->t == J_ARR && tools->len ? tools : NULL;
  stream_ctx.semantic.emit = openai_semantic_emit;
  stream_ctx.semantic.opaque = &stream_ctx;
  if (stream) {
    if (send_stream_head(fd, "text/event-stream")) {
      b_free(&prompt);
      return;
    }
    if (chat) {
      openai_delta(&stream_ctx, "role", "assistant", 9);
      openai_delta(&stream_ctx, "content", "", 0);
    }
  }
  int slot, ad = schedule_enter(s, desired, &slot);
  if (ad) {
    b_free(&prompt);
    if (!stream)
      api_error(fd, 429,
                ad == -1 ? "The inference queue is full."
                         : "Timed out waiting for the inference engine.");
    return;
  }
  request r;
  if (hr->watchdog) {
    pthread_mutex_lock(&s->sched_mu);
    s->watchdog_active++;
    pthread_mutex_unlock(&s->sched_mu);
  }
  int rc = engine_generate(&s->e, prompt.p, maximum, temp, top_p, slot, &r,
                           stream ? openai_on_data : NULL,
                           stream ? &stream_ctx : NULL);
  schedule_leave(s, slot);
  if (hr->watchdog) {
    pthread_mutex_lock(&s->sched_mu);
    s->watchdog_active--;
    pthread_mutex_unlock(&s->sched_mu);
  }
  b_free(&prompt);
  if (rc) {
    if (!stream)
      api_error(fd, 500, "The inference engine failed.");
    else if (!stream_ctx.failed)
      write_all(fd,
                "data: {\"error\":{\"message\":\"The inference engine "
                "failed.\"}}\n\n",
                strlen("data: {\"error\":{\"message\":\"The inference engine "
                       "failed.\"}}\n\n"));
    request_destroy(&r);
    semantic_free(&stream_ctx.semantic);
    return;
  }
  tool_call calls[16];
  buf visible = {0};
  int call_n = chat && tools && tools->t == J_ARR && tools->len
                   ? parse_tool_calls(r.data.p ? r.data.p : "", tools, &visible,
                                      calls, 16)
                   : 0;
  if (!chat || !tools || tools->t != J_ARR || !tools->len)
    b_add(&visible, r.data.p ? r.data.p : "");
  buf reasoning = {0}, answer = {0};
  split_reasoning(visible.p ? visible.p : "", thinking, &reasoning, &answer);
  if (stream) {
    semantic_finish(&stream_ctx.semantic);
    buf payload = {0}, wire = {0};
    b_add(&payload, "{\"id\":");
    b_json(&payload, rid);
    b_printf(&payload, ",\"object\":\"%s\",\"created\":%lld,\"model\":",
             chat ? "chat.completion.chunk" : "text_completion",
             (long long)time(NULL));
    b_json(&payload, s->c.model_id);
    if (chat) {
      b_add(&payload, ",\"choices\":[{\"index\":0,\"delta\":{");
      b_printf(&payload, "},\"logprobs\":null,\"finish_reason\":\"%s\"}]}",
               call_n             ? "tool_calls"
               : r.length_limited ? "length"
                                  : "stop");
    } else {
      b_add(&payload, ",\"choices\":[{\"index\":0,\"text\":");
      b_json(&payload, "");
      b_printf(&payload, ",\"logprobs\":null,\"finish_reason\":\"%s\"}]}",
               r.length_limited ? "length" : "stop");
    }
    b_add(&wire, "data: ");
    b_addn(&wire, payload.p, payload.n);
    b_add(&wire, "\n\ndata: [DONE]\n\n");
    if (!stream_ctx.failed)
      write_all(fd, wire.p, wire.n);
    b_free(&payload);
    b_free(&wire);
  }
  buf out = {0};
  if (chat) {
    b_add(&out, "{\"id\":");
    b_json(&out, rid);
    b_printf(&out,
             ",\"object\":\"chat.completion\",\"created\":%lld,\"model\":",
             (long long)time(NULL));
    b_json(&out, s->c.model_id);
    b_add(&out, ",\"choices\":[{\"index\":0,\"message\":{\"role\":"
                "\"assistant\",\"content\":");
    b_json(&out, answer.p ? answer.p : "");
    if (reasoning.n) {
      b_add(&out, ",\"reasoning_content\":");
      b_json(&out, reasoning.p);
    }
    if (call_n) {
      b_add(&out, ",\"tool_calls\":[");
      for (int i = 0; i < call_n; i++) {
        if (i)
          b_add(&out, ",");
        b_add(&out, "{\"id\":");
        b_json(&out, calls[i].id);
        b_add(&out, ",\"type\":\"function\",\"function\":{\"name\":");
        b_json(&out, calls[i].name);
        b_add(&out, ",\"arguments\":");
        b_json(&out, calls[i].args.p);
        b_add(&out, "}}");
      }
      b_add(&out, "]");
    }
    b_printf(&out, "},\"logprobs\":null,\"finish_reason\":\"%s\"}],",
             call_n             ? "tool_calls"
             : r.length_limited ? "length"
                                : "stop");
  } else {
    b_add(&out, "{\"id\":");
    b_json(&out, rid);
    b_printf(&out,
             ",\"object\":\"text_completion\",\"created\":%lld,\"model\":",
             (long long)time(NULL));
    b_json(&out, s->c.model_id);
    b_add(&out, ",\"choices\":[{\"index\":0,\"text\":");
    b_json(&out, answer.p ? answer.p : "");
    b_printf(&out, ",\"logprobs\":null,\"finish_reason\":\"%s\"}],",
             r.length_limited ? "length" : "stop");
  }
  b_printf(&out,
           "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_"
           "tokens\":%d}}",
           r.prompt_tokens, r.completion_tokens,
           r.prompt_tokens + r.completion_tokens);
  if (!stream)
    send_json(fd, 200, &out);
  b_free(&out);
  b_free(&visible);
  b_free(&reasoning);
  b_free(&answer);
  semantic_free(&stream_ctx.semantic);
  free_tool_calls(calls, call_n);
  request_destroy(&r);
}

static int protocol_options(server *s, jval *body, int anthropic, int *maximum,
                            double *temp, double *top_p) {
  if (!valid_optional_type(body, "model", J_STR) ||
      !valid_optional_type(body, "stream", J_BOOL) ||
      !valid_optional_type(body, "temperature", J_NUM) ||
      !valid_optional_type(body, "top_p", J_NUM))
    return 400;
  const char *model = jstr(body, "model", s->c.model_id);
  if (!model_allowed(s, model))
    return 404;
  jval *mv = jget(body, anthropic ? "max_tokens" : "max_output_tokens");
  if (!mv)
    mv = jget(body, "max_tokens");
  if (anthropic && !mv)
    return 400;
  if (mv && !valid_positive_integer(mv))
    return 400;
  *maximum = mv ? (int)mv->num : s->c.max_tokens;
  if (*maximum > s->c.max_tokens)
    *maximum = s->c.max_tokens;
  *temp = jnum(body, "temperature", .7);
  *top_p = jnum(body, "top_p", .9);
  return isfinite(*temp) && *temp >= 0 && *temp <= 2 && isfinite(*top_p) &&
                 *top_p > 0 && *top_p <= 1
             ? 0
             : 400;
}

typedef struct {
  int fd, index, next_index, block_kind, failed;
  semantic_stream semantic;
} anthropic_stream;

static int anthropic_event(int fd, const char *name, buf *payload) {
  buf wire = {0};
  b_add(&wire, "event: ");
  b_add(&wire, name);
  b_add(&wire, "\ndata: ");
  b_addn(&wire, payload->p, payload->n);
  b_add(&wire, "\n\n");
  int rc = write_all(fd, wire.p, wire.n);
  b_free(&wire);
  return rc;
}

static int anthropic_close_block(anthropic_stream *stream) {
  if (!stream->block_kind)
    return 0;
  buf stop = {0};
  b_printf(&stop, "{\"type\":\"content_block_stop\",\"index\":%d}",
           stream->index);
  stream->failed = anthropic_event(stream->fd, "content_block_stop", &stop);
  b_free(&stop);
  stream->block_kind = 0;
  return stream->failed;
}
static int anthropic_semantic_emit(void *opaque, semantic_kind kind,
                                   const char *data, size_t n,
                                   const tool_call *call) {
  anthropic_stream *stream = opaque;
  if (stream->failed)
    return 1;
  int wanted = kind == SEM_REASONING ? 1 : kind == SEM_TEXT ? 2 : 3;
  if (kind == SEM_TOOL) {
    if (anthropic_close_block(stream))
      return 1;
    stream->index = stream->next_index++;
    buf start = {0};
    b_printf(&start,
             "{\"type\":\"content_block_start\",\"index\":%d,"
             "\"content_block\":{\"type\":\"tool_use\",\"id\":",
             stream->index);
    b_json(&start, call->id);
    b_add(&start, ",\"name\":");
    b_json(&start, call->name);
    b_add(&start, ",\"input\":{}}}");
    stream->failed = anthropic_event(stream->fd, "content_block_start", &start);
    b_free(&start);
    if (stream->failed)
      return 1;
    buf delta = {0};
    b_printf(&delta,
             "{\"type\":\"content_block_delta\",\"index\":%d,"
             "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":",
             stream->index);
    b_json(&delta, call->args.p ? call->args.p : "{}");
    b_add(&delta, "}}");
    stream->failed = anthropic_event(stream->fd, "content_block_delta", &delta);
    b_free(&delta);
    stream->block_kind = 3;
    return anthropic_close_block(stream);
  }
  if (stream->block_kind != wanted) {
    if (anthropic_close_block(stream))
      return 1;
    stream->index = stream->next_index++;
    buf start = {0};
    b_printf(&start,
             "{\"type\":\"content_block_start\",\"index\":%d,"
             "\"content_block\":{\"type\":\"%s\",",
             stream->index, kind == SEM_REASONING ? "thinking" : "text");
    b_add(&start, kind == SEM_REASONING
                      ? "\"thinking\":\"\",\"signature\":\"\"}}"
                      : "\"text\":\"\"}}");
    stream->failed = anthropic_event(stream->fd, "content_block_start", &start);
    b_free(&start);
    stream->block_kind = wanted;
    if (stream->failed)
      return 1;
  }
  char *copy = strndup(data, n);
  buf delta = {0};
  b_printf(&delta,
           "{\"type\":\"content_block_delta\",\"index\":%d,"
           "\"delta\":{\"type\":\"%s\",\"%s\":",
           stream->index, kind == SEM_REASONING ? "thinking_delta" : "text_delta",
           kind == SEM_REASONING ? "thinking" : "text");
  b_json(&delta, copy);
  b_add(&delta, "}}");
  stream->failed = anthropic_event(stream->fd, "content_block_delta", &delta);
  b_free(&delta);
  free(copy);
  return stream->failed;
}
static int anthropic_on_data(const char *data, size_t n, void *opaque) {
  anthropic_stream *stream = opaque;
  return semantic_feed(&stream->semantic, data, n);
}

static void anthropic(server *s, int fd, jval *body) {
  int max;
  double temp, top_p;
  int bad = protocol_options(s, body, 1, &max, &temp, &top_p);
  if (bad) {
    api_error(fd, bad,
              bad == 404 ? "The requested model is not available."
                         : "Invalid Anthropic request.");
    return;
  }
  jval *think = jget(body, "thinking");
  int thinking =
      think && think->t == J_OBJ && !strcmp(jstr(think, "type", ""), "enabled");
  buf prompt = {0};
  if (render_anthropic(&prompt, body, thinking)) {
    b_free(&prompt);
    api_error(fd, 400, "Anthropic messages require supported text content.");
    return;
  }
  jval *tools = jget(body, "tools");
  int has_tools = tools && tools->t == J_ARR && tools->len;
  int stream = jbool(body, "stream", 0);
  char id[80];
  snprintf(id, sizeof(id), "msg_%lld_%lu", (long long)time(NULL),
           (unsigned long)pthread_self());
  anthropic_stream live = {.fd = fd};
  live.semantic.thinking = thinking;
  live.semantic.tools = has_tools ? tools : NULL;
  live.semantic.emit = anthropic_semantic_emit;
  live.semantic.opaque = &live;
  if (stream) {
    if (send_stream_head(fd, "text/event-stream")) {
      b_free(&prompt);
      return;
    }
    buf start = {0};
    b_add(&start, "{\"type\":\"message_start\",\"message\":{\"id\":");
    b_json(&start, id);
    b_add(&start, ",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
    b_json(&start, s->c.model_id);
    b_add(&start, ",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,"
                  "\"usage\":{\"input_tokens\":0,\"output_tokens\":0,"
                  "\"cache_creation_input_tokens\":0,"
                  "\"cache_read_input_tokens\":0}}}");
    live.failed = anthropic_event(fd, "message_start", &start);
    b_free(&start);
    if (live.failed) {
      b_free(&prompt);
      return;
    }
  }
  request r;
  int rc = generate_prompt_cb(
      s, prompt.p, max, temp, top_p, -1, &r,
      stream ? anthropic_on_data : NULL, stream ? &live : NULL);
  b_free(&prompt);
  if (rc) {
    if (rc == -4)
      request_destroy(&r);
    if (!stream)
      api_error(fd, rc == -1 || rc == -2 ? 429 : 500,
                "Inference request failed.");
    else if (!live.failed) {
      buf error = {0};
      b_add(&error, "{\"type\":\"error\",\"error\":{\"type\":\"api_error\","
                    "\"message\":\"The colibri engine failed to process the "
                    "request.\"}}");
      anthropic_event(fd, "error", &error);
      b_free(&error);
    }
    semantic_free(&live.semantic);
    return;
  }
  tool_call calls[16];
  buf visible = {0};
  int call_n = tools && tools->t == J_ARR && tools->len
                   ? parse_tool_calls(r.data.p ? r.data.p : "", tools, &visible,
                                      calls, 16)
                   : 0;
  if (!tools || tools->t != J_ARR || !tools->len)
    b_add(&visible, r.data.p ? r.data.p : "");
  buf reasoning = {0}, content = {0};
  split_reasoning(visible.p ? visible.p : "", thinking, &reasoning, &content);
  buf out = {0};
  b_add(&out, "{\"id\":");
  b_json(&out, id);
  b_add(&out, ",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
  b_json(&out, s->c.model_id);
  b_add(&out, ",\"content\":[");
  int comma = 0;
  if (reasoning.n) {
    b_add(&out, "{\"type\":\"thinking\",\"thinking\":");
    b_json(&out, reasoning.p);
    b_add(&out, ",\"signature\":\"\"}");
    comma = 1;
  }
  if (content.n || (!comma && !call_n)) {
    if (comma)
      b_add(&out, ",");
    b_add(&out, "{\"type\":\"text\",\"text\":");
    b_json(&out, content.p ? content.p : "");
    b_add(&out, "}");
    comma = 1;
  }
  for (int i = 0; i < call_n; i++) {
    if (comma)
      b_add(&out, ",");
    b_add(&out, "{\"type\":\"tool_use\",\"id\":");
    b_json(&out, calls[i].id);
    b_add(&out, ",\"name\":");
    b_json(&out, calls[i].name);
    b_add(&out, ",\"input\":");
    b_add(&out, calls[i].args.p);
    b_add(&out, "}");
    comma = 1;
  }
  b_printf(&out,
           "],\"stop_reason\":\"%s\",\"stop_sequence\":null,\"usage\":{\"input_"
           "tokens\":%d,\"output_tokens\":%d,\"cache_creation_input_tokens\":0,"
           "\"cache_read_input_tokens\":0}}",
           call_n             ? "tool_use"
           : r.length_limited ? "max_tokens"
                              : "end_turn",
           r.prompt_tokens, r.completion_tokens);
  if (!stream)
    send_json(fd, 200, &out);
  else {
    semantic_finish(&live.semantic);
    anthropic_close_block(&live);
    buf wire = {0};
    b_printf(&wire,
             "event: message_delta\ndata: {\"type\":\"message_delta\","
             "\"delta\":{\"stop_reason\":\"%s\",\"stop_sequence\":null},"
             "\"usage\":{\"input_tokens\":%d,\"output_tokens\":%d,"
             "\"cache_creation_input_tokens\":0,"
             "\"cache_read_input_tokens\":0}}\n\n"
             "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
             call_n             ? "tool_use"
             : r.length_limited ? "max_tokens"
                                : "end_turn",
             r.prompt_tokens, r.completion_tokens);
    if (!live.failed)
      write_all(fd, wire.p, wire.n);
    b_free(&wire);
  }
  semantic_free(&live.semantic);
  b_free(&out);
  b_free(&visible);
  b_free(&reasoning);
  b_free(&content);
  free_tool_calls(calls, call_n);
  request_destroy(&r);
}

static void anthropic_count_tokens(server *s, int fd, jval *body) {
  const char *model = jstr(body, "model", s->c.model_id);
  if (!model_allowed(s, model)) {
    api_error(fd, 404, "The requested model is not available.");
    return;
  }
  buf prompt = {0};
  if (render_anthropic(&prompt, body, 0)) {
    b_free(&prompt);
    api_error(fd, 400, "Anthropic messages require supported text content.");
    return;
  }
  size_t tokens = (prompt.n + 3) / 4;
  if (!tokens)
    tokens = 1;
  buf out = {0};
  b_printf(&out, "{\"input_tokens\":%zu}", tokens);
  send_body(fd, 200, "application/json", out.p, out.n,
            "x-colibri-token-count: estimated\r\n");
  b_free(&out);
  b_free(&prompt);
}

typedef struct {
  int fd, failed;
  semantic_stream semantic;
} responses_stream;

static int responses_event(responses_stream *stream, buf *payload) {
  buf wire = {0};
  b_add(&wire, "data: ");
  b_addn(&wire, payload->p, payload->n);
  b_add(&wire, "\n\n");
  stream->failed = write_all(stream->fd, wire.p, wire.n) != 0;
  b_free(&wire);
  return stream->failed;
}

static int responses_semantic_emit(void *opaque, semantic_kind kind,
                                   const char *data, size_t n,
                                   const tool_call *call) {
  responses_stream *stream = opaque;
  if (stream->failed)
    return 1;
  buf event = {0};
  if (kind == SEM_TOOL) {
    int index = stream->semantic.tool_count;
    b_add(&event, "{\"type\":\"response.output_item.added\",\"output_index\":");
    b_printf(&event, "%d,\"item\":{\"type\":\"function_call\",\"id\":", index);
    b_json(&event, call->id);
    b_add(&event, ",\"call_id\":");
    b_json(&event, call->id);
    b_add(&event, ",\"name\":");
    b_json(&event, call->name);
    b_add(&event, ",\"arguments\":\"\",\"status\":\"in_progress\"}}");
    if (responses_event(stream, &event)) {
      b_free(&event);
      return 1;
    }
    b_free(&event);

    b_add(&event,
          "{\"type\":\"response.function_call_arguments.delta\",\"delta\":");
    b_json(&event, call->args.p ? call->args.p : "{}");
    b_add(&event, ",\"item_id\":");
    b_json(&event, call->id);
    b_printf(&event, ",\"output_index\":%d}", index);
    if (responses_event(stream, &event)) {
      b_free(&event);
      return 1;
    }
    b_free(&event);

    b_add(&event,
          "{\"type\":\"response.function_call_arguments.done\",\"arguments\":");
    b_json(&event, call->args.p ? call->args.p : "{}");
    b_add(&event, ",\"item_id\":");
    b_json(&event, call->id);
    b_printf(&event, ",\"output_index\":%d}", index);
    if (responses_event(stream, &event)) {
      b_free(&event);
      return 1;
    }
    b_free(&event);

    b_add(&event, "{\"type\":\"response.output_item.done\",\"output_index\":");
    b_printf(&event, "%d,\"item\":{\"type\":\"function_call\",\"id\":", index);
    b_json(&event, call->id);
    b_add(&event, ",\"call_id\":");
    b_json(&event, call->id);
    b_add(&event, ",\"name\":");
    b_json(&event, call->name);
    b_add(&event, ",\"arguments\":");
    b_json(&event, call->args.p ? call->args.p : "{}");
    b_add(&event, ",\"status\":\"completed\"}}");
  } else {
    b_add(&event, kind == SEM_REASONING
                      ? "{\"type\":\"response.reasoning_text.delta\",\"delta\":"
                      : "{\"type\":\"response.output_text.delta\",\"delta\":");
    char *copy = strndup(data, n);
    b_json(&event, copy);
    free(copy);
    b_add(&event, "}");
  }
  int rc = responses_event(stream, &event);
  b_free(&event);
  return rc;
}

static int responses_on_data(const char *data, size_t n, void *opaque) {
  responses_stream *stream = opaque;
  return semantic_feed(&stream->semantic, data, n);
}

static void responses(server *s, int fd, jval *body) {
  int max;
  double temp, top_p;
  int bad = protocol_options(s, body, 0, &max, &temp, &top_p);
  if (bad) {
    api_error(fd, bad,
              bad == 404 ? "The requested model is not available."
                         : "Invalid Responses request.");
    return;
  }
  jval *reason = jget(body, "reasoning");
  int thinking = s->c.default_thinking;
  if (reason && reason->t == J_OBJ) {
    const char *effort = jstr(reason, "effort", "");
    if (!strcmp(effort, "none"))
      thinking = 0;
    else if (*effort)
      thinking = 1;
  }
  buf prompt = {0};
  if (render_responses(&prompt, body, thinking)) {
    b_free(&prompt);
    api_error(fd, 400, "`input` must be a string or supported message array.");
    return;
  }
  const char *previous_id = jstr(body, "previous_response_id", NULL);
  char *previous = history_get(s, previous_id);
  if (previous_id && !previous) {
    b_free(&prompt);
    api_error(fd, 404, "The previous response does not exist.");
    return;
  }
  if (previous) {
    buf joined = {0};
    b_add(&joined, previous);
    const char *current = prompt.p;
    size_t prefix = strlen("[gMASK]<sop>");
    if (!strncmp(current, "[gMASK]<sop>", prefix))
      current += prefix;
    b_add(&joined, current);
    b_free(&prompt);
    prompt = joined;
    free(previous);
  }
  buf history_prompt = {0};
  b_add(&history_prompt, prompt.p);
  strip_tool_preamble(&history_prompt);
  jval *tools = jget(body, "tools");
  int stream = jbool(body, "stream", 0);
  char id[80], oid[80];
  snprintf(id, sizeof(id), "resp_%lld_%lu", (long long)time(NULL),
           (unsigned long)pthread_self());
  snprintf(oid, sizeof(oid), "msg_%lld_%lu", (long long)time(NULL),
           (unsigned long)pthread_self());
  responses_stream live = {.fd = fd};
  live.semantic.thinking = thinking;
  live.semantic.tools = tools && tools->t == J_ARR && tools->len ? tools : NULL;
  live.semantic.emit = responses_semantic_emit;
  live.semantic.opaque = &live;
  if (stream) {
    if (send_stream_head(fd, "text/event-stream")) {
      b_free(&prompt);
      b_free(&history_prompt);
      return;
    }
    buf created = {0};
    b_add(&created, "{\"type\":\"response.created\",\"response\":{\"id\":");
    b_json(&created, id);
    b_printf(&created,
             ",\"object\":\"response\",\"created_at\":%lld,\"status\":"
             "\"in_progress\",\"model\":",
             (long long)time(NULL));
    b_json(&created, s->c.model_id);
    b_add(&created, ",\"output\":[]}}");
    responses_event(&live, &created);
    b_free(&created);
  }
  request r;
  int rc = generate_prompt_cb(s, prompt.p, max, temp, top_p, -1, &r,
                              stream ? responses_on_data : NULL,
                              stream ? &live : NULL);
  b_free(&prompt);
  if (rc) {
    if (rc == -4)
      request_destroy(&r);
    b_free(&history_prompt);
    if (!stream)
      api_error(fd, rc == -1 || rc == -2 ? 429 : 500,
                "Inference request failed.");
    else if (!live.failed)
      write_all(fd, "data: {\"type\":\"error\",\"error\":{\"message\":"
                    "\"Inference request failed.\"}}\n\n",
                strlen("data: {\"type\":\"error\",\"error\":{\"message\":"
                       "\"Inference request failed.\"}}\n\n"));
    semantic_free(&live.semantic);
    return;
  }
  tool_call calls[16];
  buf visible = {0};
  int call_n = tools && tools->t == J_ARR && tools->len
                   ? parse_tool_calls(r.data.p ? r.data.p : "", tools, &visible,
                                      calls, 16)
                   : 0;
  if (!tools || tools->t != J_ARR || !tools->len)
    b_add(&visible, r.data.p ? r.data.p : "");
  buf reasoning = {0}, content = {0};
  split_reasoning(visible.p ? visible.p : "", thinking, &reasoning, &content);
  buf out = {0};
  b_add(&out, "{\"id\":");
  b_json(&out, id);
  b_printf(&out,
           ",\"object\":\"response\",\"created_at\":%lld,\"status\":"
           "\"completed\",\"model\":",
           (long long)time(NULL));
  b_json(&out, s->c.model_id);
  b_add(&out, ",\"output\":[");
  if (call_n) {
    for (int i = 0; i < call_n; i++) {
      if (i)
        b_add(&out, ",");
      b_add(&out, "{\"type\":\"function_call\",\"id\":");
      b_json(&out, calls[i].id);
      b_add(&out, ",\"call_id\":");
      b_json(&out, calls[i].id);
      b_add(&out, ",\"name\":");
      b_json(&out, calls[i].name);
      b_add(&out, ",\"arguments\":");
      b_json(&out, calls[i].args.p);
      b_add(&out, ",\"status\":\"completed\"}");
    }
  } else {
    b_add(&out, "{\"id\":");
    b_json(&out, oid);
    b_add(&out,
          ",\"type\":\"message\",\"status\":\"completed\",\"role\":"
          "\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":");
    b_json(&out, content.p ? content.p : "");
    b_add(&out, ",\"annotations\":[]}]}");
  }
  b_add(&out, "],\"usage\":{\"input_tokens\":");
  b_printf(&out, "%d,\"output_tokens\":%d,\"total_tokens\":%d},\"error\":null}",
           r.prompt_tokens, r.completion_tokens,
           r.prompt_tokens + r.completion_tokens);
  if (!stream)
    send_json(fd, 200, &out);
  else {
    semantic_finish(&live.semantic);
    buf completed = {0};
    b_add(&completed, "{\"type\":\"response.completed\",\"response\":");
    b_addn(&completed, out.p, out.n);
    b_add(&completed, "}");
    if (!live.failed)
      responses_event(&live, &completed);
    if (!live.failed)
      write_all(fd, "data: [DONE]\n\n", strlen("data: [DONE]\n\n"));
    b_free(&completed);
  }
  b_free(&out);
  b_add(&history_prompt, r.data.p ? r.data.p : "");
  history_put(s, id, history_prompt.p);
  b_free(&history_prompt);
  b_free(&visible);
  b_free(&reasoning);
  b_free(&content);
  free_tool_calls(calls, call_n);
  semantic_free(&live.semantic);
  request_destroy(&r);
}

static void ollama_discovery(server *s, int fd, int ps) {
  /* Ollama's list command abbreviates digests with digest[:12].  Keep this a
     full, sha256-shaped value even though Colibri does not manage Ollama
     manifests, so older clients do not panic on a short compatibility ID. */
  static const char digest[] =
      "sha256:bb43a640f04c8e5504a8fbc8c6980455029f3e8fc1dedff10bcd04f94c4f4319";
  char modified[32] = "1970-01-01T00:00:00Z";
  struct tm utc;
  if (s->model_modified && gmtime_r(&s->model_modified, &utc))
    strftime(modified, sizeof(modified), "%Y-%m-%dT%H:%M:%SZ", &utc);
  buf b = {0};
  b_add(&b, "{\"models\":[{");
  if (ps) {
    b_add(&b, "\"name\":");
    b_json(&b, s->c.model_id);
    b_add(&b, ",\"model\":");
    b_json(&b, s->c.model_id);
    b_printf(&b, ",\"size\":%llu,\"digest\":", s->model_size);
    b_json(&b, digest);
    b_add(&b, ",\"expires_at\":\"9999-12-31T23:59:59Z\",\"size_vram\":0");
  } else {
    b_add(&b, "\"name\":");
    b_json(&b, s->c.model_id);
    b_add(&b, ",\"model\":");
    b_json(&b, s->c.model_id);
    b_add(&b, ",\"modified_at\":");
    b_json(&b, modified);
    b_printf(&b, ",\"size\":%llu,\"digest\":", s->model_size);
    b_json(&b, digest);
    b_add(&b,
          ",\"details\":{\"format\":\"safetensors\",\"family\":\"glm\","
          "\"parameter_size\":\"744B\",\"quantization_level\":\"Q4\"}");
  }
  b_add(&b, "}]}");
  send_json(fd, 200, &b);
  b_free(&b);
}

static int hex_digit(char c) {
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  return -1;
}
static int decode_path(const char *path, char *out, size_t cap) {
  size_t n = 0;
  for (size_t i = 0; path[i] && path[i] != '?'; i++) {
    unsigned char c = (unsigned char)path[i];
    if (c == '%' && path[i + 1] && path[i + 2]) {
      int a = hex_digit(path[i + 1]), b = hex_digit(path[i + 2]);
      if (a < 0 || b < 0)
        return -1;
      c = (unsigned char)((a << 4) | b);
      i += 2;
    }
    if (!c || n + 1 >= cap)
      return -1;
    out[n++] = (char)c;
  }
  out[n] = 0;
  return 0;
}
static const char *mime_type(const char *path) {
  const char *dot = strrchr(path, '.');
  if (!dot)
    return "application/octet-stream";
  if (!strcmp(dot, ".html"))
    return "text/html; charset=utf-8";
  if (!strcmp(dot, ".js"))
    return "text/javascript; charset=utf-8";
  if (!strcmp(dot, ".css"))
    return "text/css; charset=utf-8";
  if (!strcmp(dot, ".svg"))
    return "image/svg+xml";
  if (!strcmp(dot, ".png"))
    return "image/png";
  return "application/octet-stream";
}
static int static_file(server *s, int fd, const char *url) {
  if (!s->c.web_root || !*s->c.web_root)
    return 0;
  char decoded[PATH_MAX], root[PATH_MAX], candidate[PATH_MAX],
      resolved[PATH_MAX];
  if (decode_path(url, decoded, sizeof(decoded)) ||
      !realpath(s->c.web_root, root))
    return 0;
  const char *relative = decoded[0] == '/' ? decoded + 1 : decoded;
  if (!*relative)
    relative = "index.html";
  if (snprintf(candidate, sizeof(candidate), "%s/%s", root, relative) >=
      (int)sizeof(candidate))
    return 0;
  if (!realpath(candidate, resolved)) {
    if (snprintf(candidate, sizeof(candidate), "%s/index.html", root) >=
            (int)sizeof(candidate) ||
        !realpath(candidate, resolved))
      return 0;
  }
  size_t rn = strlen(root);
  if (strncmp(resolved, root, rn) || (resolved[rn] && resolved[rn] != '/'))
    return 0;
  struct stat st;
  if (stat(resolved, &st) || !S_ISREG(st.st_mode) || st.st_size < 0 ||
      st.st_size > (32 << 20))
    return 0;
  FILE *f = fopen(resolved, "rb");
  if (!f)
    return 0;
  char *data = malloc((size_t)st.st_size);
  size_t got = fread(data, 1, (size_t)st.st_size, f);
  fclose(f);
  if (got != (size_t)st.st_size) {
    free(data);
    return 0;
  }
  send_body(fd, 200, mime_type(resolved), data, got, NULL);
  free(data);
  return 1;
}

typedef struct {
  int fd, chat, failed;
  const char *model;
  semantic_stream semantic;
} ollama_stream;
static int ollama_semantic_emit(void *opaque, semantic_kind kind,
                                const char *data, size_t n,
                                const tool_call *call) {
  ollama_stream *s = opaque;
  if (s->failed)
    return 1;
  buf out = {0};
  b_add(&out, "{\"model\":");
  b_json(&out, s->model);
  b_add(&out, ",\"created_at\":\"1970-01-01T00:00:00Z\",");
  if (s->chat) {
    b_add(&out, "\"message\":{\"role\":\"assistant\",");
    if (kind == SEM_TOOL) {
      b_add(&out, "\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":");
      b_json(&out, call->name);
      b_add(&out, ",\"arguments\":");
      b_add(&out, call->args.p ? call->args.p : "{}");
      b_add(&out, "}}]}");
    } else {
      b_add(&out, kind == SEM_REASONING ? "\"thinking\":" : "\"content\":");
      char *copy = strndup(data, n);
      b_json(&out, copy);
      free(copy);
      b_add(&out, "}");
    }
  } else {
    b_add(&out, "\"response\":");
    char *copy = strndup(data ? data : "", n);
    b_json(&out, copy);
    free(copy);
  }
  b_add(&out, ",\"done\":false}\n");
  s->failed = write_all(s->fd, out.p, out.n) != 0;
  b_free(&out);
  return s->failed;
}
static int ollama_on_data(const char *data, size_t n, void *opaque) {
  ollama_stream *s = opaque;
  return semantic_feed(&s->semantic, data, n);
}

static void ollama_generate(server *s, int fd, jval *body, int chat) {
  const char *model = jstr(body, "model", s->c.model_id);
  if (!model_allowed(s, model)) {
    api_error(fd, 404, "model not found");
    return;
  }
  jval *opts = jget(body, "options");
  int max = opts && opts->t == J_OBJ
                ? (int)jnum(opts, "num_predict", s->c.max_tokens)
                : s->c.max_tokens;
  if (max < 1)
    max = s->c.max_tokens;
  if (max > s->c.max_tokens)
    max = s->c.max_tokens;
  double temp = opts && opts->t == J_OBJ ? jnum(opts, "temperature", .7) : .7,
         top = opts && opts->t == J_OBJ ? jnum(opts, "top_p", .9) : .9;
  buf prompt = {0};
  if (chat ? render_chat(&prompt, body, jbool(body, "think", 0))
           : (jget(body, "prompt") && jget(body, "prompt")->t == J_STR
                  ? (b_add(&prompt, jget(body, "prompt")->str), 0)
                  : -1)) {
    b_free(&prompt);
    api_error(fd, 400, "invalid Ollama request");
    return;
  }
  jval *tools = jget(body, "tools");
  int stream = jbool(body, "stream", 1);
  ollama_stream live = {.fd = fd, .chat = chat, .model = s->c.model_id};
  live.semantic.thinking = jbool(body, "think", 0);
  live.semantic.tools =
      chat && tools && tools->t == J_ARR && tools->len ? tools : NULL;
  live.semantic.emit = ollama_semantic_emit;
  live.semantic.opaque = &live;
  if (stream && send_stream_head(fd, "application/x-ndjson")) {
    b_free(&prompt);
    return;
  }
  request r;
  int rc = generate_prompt_cb(s, prompt.p, max, temp, top, -1, &r,
                              stream ? ollama_on_data : NULL,
                              stream ? &live : NULL);
  b_free(&prompt);
  if (rc) {
    if (rc == -4)
      request_destroy(&r);
    if (!stream)
      api_error(fd, 500, "inference failed");
    semantic_free(&live.semantic);
    return;
  }
  tool_call calls[16];
  buf visible = {0};
  int call_n = chat && tools && tools->t == J_ARR && tools->len
                   ? parse_tool_calls(r.data.p ? r.data.p : "", tools, &visible,
                                      calls, 16)
                   : 0;
  if (!chat || !tools || tools->t != J_ARR || !tools->len)
    b_add(&visible, r.data.p ? r.data.p : "");
  if (stream)
    semantic_finish(&live.semantic);
  buf out = {0};
  b_add(&out, "{\"model\":");
  b_json(&out, s->c.model_id);
  b_add(&out, ",\"created_at\":\"1970-01-01T00:00:00Z\",");
  if (chat) {
    b_add(&out, "\"message\":{\"role\":\"assistant\",\"content\":");
    b_json(&out, stream ? "" : (visible.p ? visible.p : ""));
    if (call_n && !stream) {
      b_add(&out, ",\"tool_calls\":[");
      for (int i = 0; i < call_n; i++) {
        if (i)
          b_add(&out, ",");
        b_add(&out, "{\"function\":{\"name\":");
        b_json(&out, calls[i].name);
        b_add(&out, ",\"arguments\":");
        b_add(&out, calls[i].args.p);
        b_add(&out, "}}");
      }
      b_add(&out, "]");
    }
    b_add(&out, "},");
  } else {
    b_add(&out, "\"response\":");
    b_json(&out, stream ? "" : (visible.p ? visible.p : ""));
    b_add(&out, ",");
  }
  b_printf(&out,
           "\"done\":true,\"done_reason\":\"%s\",\"prompt_eval_count\":%d,"
           "\"eval_count\":%d}",
           r.length_limited ? "length" : "stop", r.prompt_tokens,
           r.completion_tokens);
  if (!stream)
    send_json(fd, 200, &out);
  else {
    b_add(&out, "\n");
    if (!live.failed)
      write_all(fd, out.p, out.n);
  }
  b_free(&out);
  b_free(&visible);
  free_tool_calls(calls, call_n);
  semantic_free(&live.semantic);
  request_destroy(&r);
}

typedef struct {
  server *s;
  int fd;
} client_arg;
static int cors_allowed(server *s, const char *origin) {
  static const char *defaults[] = {
      "http://127.0.0.1:8000",  "http://localhost:8000",
      "http://127.0.0.1:5173",  "http://localhost:5173",
      "http://tauri.localhost", "tauri://localhost"};
  if (!origin || !*origin)
    return 0;
  if (s->c.cors_origin_count) {
    for (int i = 0; i < s->c.cors_origin_count; i++)
      if (!strcmp(s->c.cors_origins[i], "*") ||
          !strcmp(s->c.cors_origins[i], origin))
        return 1;
    return 0;
  }
  for (size_t i = 0; i < sizeof(defaults) / sizeof(*defaults); i++)
    if (!strcmp(defaults[i], origin))
      return 1;
  return 0;
}
static void *client_main(void *arg) {
  client_arg *a = arg;
  server *s = a->s;
  int fd = a->fd;
  free(a);
  response_origin[0] = 0;
  response_path[0] = 0;
  snprintf(response_request_id, sizeof(response_request_id), "req_%lld_%lu",
           (long long)time(NULL), (unsigned long)pthread_self());
  http_req r = {0};
  int pr = parse_request(fd, &r);
  if (pr) {
    api_error(fd, pr == -2 ? 400 : 400, "Invalid HTTP request.");
    goto done;
  }
  if (http_debug_enabled())
    fprintf(stderr,
            "[http] id=%s method=%s path=%s bytes=%zu auth=%s origin=%s\n",
            response_request_id, r.method, r.path, r.content_length,
            r.authorization[0] || r.api_key[0] ? "present" : "absent",
            r.origin[0] ? r.origin : "-");
  if (cors_allowed(s, r.origin))
    snprintf(response_origin, sizeof(response_origin), "%s", r.origin);
  snprintf(response_path, sizeof(response_path), "%s", r.path);
  /* HTTP request targets may carry query parameters.  Routing is based on the
   * path component; Anthropic clients currently append ?beta=true. */
  char *query = strchr(r.path, '?');
  if (query)
    *query = 0;
  if (!auth_ok(s, &r)) {
    api_error(fd, 401, "Invalid or missing API key.");
    goto done;
  }
  if (!strcmp(r.method, "OPTIONS")) {
    send_body(fd, 204, "text/plain", "", 0,
              "Access-Control-Allow-Methods: GET, POST, "
              "OPTIONS\r\nAccess-Control-Allow-Headers: Authorization, "
              "Content-Type, x-api-key, anthropic-version, anthropic-beta\r\n");
    goto done;
  }
  if (!strcmp(r.method, "HEAD")) {
    if (!strcmp(r.path, "/") || !strcmp(r.path, "/v1/models") ||
        !strncmp(r.path, "/v1/models/", 11))
      send_body(fd, 200, "text/plain", "", 0, NULL);
    else
      api_error(fd, 404, "Not found.");
    goto done;
  }
  if (!strcmp(r.method, "GET")) {
    if (!strcmp(r.path, "/health"))
      health(s, fd);
    else if (!strcmp(r.path, "/profile"))
      profile(s, fd);
    else if (!strcmp(r.path, "/v1/models"))
      models(s, fd);
    else if (!strncmp(r.path, "/v1/models/", 11))
      model_get(s, fd, r.path + 11);
    else if (!strcmp(r.path, "/api/version"))
      send_body(fd, 200, "application/json", "{\"version\":\"colibri-native\"}",
                28, NULL);
    else if (!strcmp(r.path, "/api/tags"))
      ollama_discovery(s, fd, 0);
    else if (!strcmp(r.path, "/api/ps"))
      ollama_discovery(s, fd, 1);
    else if (static_file(s, fd, r.path))
      ;
    else
      api_error(fd, 404, "Not found.");
    goto done;
  }
  if (strcmp(r.method, "POST")) {
    api_error(fd, 405, "Method not allowed.");
    goto done;
  }
  if (!r.body || !r.content_length) {
    api_error(fd, 400, "Request body must be valid JSON.");
    goto done;
  }
  char *arena = NULL;
  jval *body = json_parse(r.body, &arena);
  if (!body || body->t != J_OBJ) {
    api_error(fd, 400, "Request body must be a JSON object.");
    json_free(body);
    goto done;
  }
  if (http_debug_enabled()) {
    jval *messages = jget(body, "messages"), *tools = jget(body, "tools");
    int deferred = 0;
    if (tools && tools->t == J_ARR)
      for (int i = 0; i < tools->len; i++)
        deferred += jbool(tools->kids[i], "defer_loading", 0) ? 1 : 0;
    jval *maximum = jget(body, "max_tokens");
    fprintf(stderr,
            "[http] id=%s request model=%s stream=%s messages=%d tools=%d "
            "deferred_tools=%d max_tokens=%.0f\n",
            response_request_id, jstr(body, "model", "(default)"),
            jbool(body, "stream", 0) ? "true" : "false",
            messages && messages->t == J_ARR ? messages->len : 0,
            tools && tools->t == J_ARR ? tools->len : 0, deferred,
            maximum && maximum->t == J_NUM ? maximum->num : 0.0);
  }
  if (!strcmp(r.path, "/v1/chat/completions"))
    completion(s, fd, &r, body, 1);
  else if (!strcmp(r.path, "/v1/completions"))
    completion(s, fd, &r, body, 0);
  else if (!strcmp(r.path, "/v1/messages"))
    anthropic(s, fd, body);
  else if (!strcmp(r.path, "/v1/messages/count_tokens"))
    anthropic_count_tokens(s, fd, body);
  else if (!strcmp(r.path, "/v1/responses"))
    responses(s, fd, body);
  else if (!strcmp(r.path, "/api/chat"))
    ollama_generate(s, fd, body, 1);
  else if (!strcmp(r.path, "/api/generate"))
    ollama_generate(s, fd, body, 0);
  else if (!strcmp(r.path, "/api/show"))
    send_body(fd, 200, "application/json",
              "{\"capabilities\":[\"completion\",\"tools\"]}",
              sizeof("{\"capabilities\":[\"completion\",\"tools\"]}") - 1,
              NULL);
  else
    api_error(fd, 404, "Not found.");
  json_free(body);
done:
  free(r.body);
  shutdown(fd, SHUT_RDWR);
  close(fd);
  pthread_mutex_lock(&s->clients_mu);
  s->clients--;
  pthread_cond_broadcast(&s->clients_cv);
  pthread_mutex_unlock(&s->clients_mu);
  return NULL;
}

static void on_signal(int sig) {
  (void)sig;
  stopping = 1;
  if (signal_listener >= 0)
    close((int)signal_listener);
}
int coli_server_run(const coli_server_config *c) {
  server s = {0};
  s.c = *c;
  s.created = time(NULL);
  model_metadata(s.c.model, &s.model_size, &s.model_modified);
  if (s.model_modified > s.created)
    s.model_modified = s.created;
  pthread_mutex_init(&s.sched_mu, NULL);
  pthread_cond_init(&s.sched_cv, NULL);
  pthread_mutex_init(&s.history_mu, NULL);
  pthread_mutex_init(&s.clients_mu, NULL);
  pthread_cond_init(&s.clients_cv, NULL);
  s.slot_busy = calloc((size_t)c->kv_slots, sizeof(int));
  struct addrinfo hints = {0}, *ai = NULL;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  char port[20];
  snprintf(port, sizeof(port), "%d", c->port);
  if (getaddrinfo(c->host, port, &hints, &ai)) {
    fprintf(stderr, "cannot resolve listen address\n");
    return 2;
  }
  for (struct addrinfo *p = ai; p; p = p->ai_next) {
    s.listener = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (s.listener < 0)
      continue;
    int one = 1;
    setsockopt(s.listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (!bind(s.listener, p->ai_addr, p->ai_addrlen) && !listen(s.listener, 64))
      break;
    close(s.listener);
    s.listener = -1;
  }
  freeaddrinfo(ai);
  if (s.listener < 0) {
    perror("bind");
    return 2;
  }
  if (engine_start(&s.e, c)) {
    fprintf(stderr, "colibri engine exited while loading\n");
    close(s.listener);
    return 2;
  }
  fprintf(stderr, "OpenAI-compatible API listening on http://%s:%d/v1\n",
          c->host, c->port);
  signal_listener = s.listener;
  signal(SIGTERM, on_signal);
  signal(SIGINT, on_signal);
  while (!stopping) {
    struct sockaddr_storage peer;
    socklen_t pn = sizeof(peer);
    int fd = accept(s.listener, (struct sockaddr *)&peer, &pn);
    if (fd < 0) {
      if (errno == EINTR)
        continue;
      break;
    }
    struct timeval recv_timeout = {.tv_sec = 30};
    struct timeval send_timeout = {.tv_sec = 10};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &recv_timeout,
               sizeof(recv_timeout));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &send_timeout,
               sizeof(send_timeout));
    client_arg *a = malloc(sizeof(*a));
    a->s = &s;
    a->fd = fd;
    pthread_t th;
    pthread_mutex_lock(&s.clients_mu);
    s.clients++;
    pthread_mutex_unlock(&s.clients_mu);
    if (!pthread_create(&th, NULL, client_main, a))
      pthread_detach(th);
    else {
      pthread_mutex_lock(&s.clients_mu);
      s.clients--;
      pthread_mutex_unlock(&s.clients_mu);
      close(fd);
      free(a);
    }
  }
  close(s.listener);
  pthread_mutex_lock(&s.sched_mu);
  pthread_cond_broadcast(&s.sched_cv);
  pthread_mutex_unlock(&s.sched_mu);
  pthread_mutex_lock(&s.clients_mu);
  while (s.clients)
    pthread_cond_wait(&s.clients_cv, &s.clients_mu);
  pthread_mutex_unlock(&s.clients_mu);
  engine_stop(&s.e);
  response_entry *entry = s.history_head;
  while (entry) {
    response_entry *next = entry->next;
    free(entry->prompt);
    free(entry);
    entry = next;
  }
  free(s.slot_busy);
  return 0;
}
