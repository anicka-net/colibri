#define _GNU_SOURCE
#include "native_chat.h"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static int write_all_fd(int fd, const void *p, size_t n) {
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
static int read_marker(FILE *f, const char *marker, int emit) {
  size_t mn = strlen(marker), n = 0;
  char *pending = malloc(mn + 1);
  int ch;
  while ((ch = fgetc(f)) != EOF) {
    pending[n++] = (char)ch;
    if (n >= mn && !memcmp(pending + n - mn, marker, mn)) {
      size_t body = n - mn;
      if (emit && body)
        fwrite(pending, 1, body, stdout);
      free(pending);
      return 0;
    }
    if (n == mn) {
      if (emit)
        fputc(pending[0], stdout);
      memmove(pending, pending + 1, --n);
    }
  }
  free(pending);
  return -1;
}
int coli_chat_run(const char *engine, const char *model, int cap, int eb,
                  int db, int max_tokens, int context) {
  int in[2], out[2];
  if (pipe(in) || pipe(out)) {
    perror("pipe");
    return 2;
  }
  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    return 2;
  }
  if (!pid) {
    dup2(in[0], 0);
    dup2(out[1], 1);
    close(in[0]);
    close(in[1]);
    close(out[0]);
    close(out[1]);
    setenv("SNAP", model, 1);
    setenv("SERVE", "1", 1);
    unsetenv("SERVE_BATCH");
    char v[32];
    snprintf(v, sizeof(v), "%d", max_tokens);
    setenv("NGEN", v, 1);
    if (context > 0) {
      snprintf(v, sizeof(v), "%d", context);
      setenv("CTX", v, 1);
    }
    char a[20], b[20], c[20];
    snprintf(a, sizeof(a), "%d", cap);
    snprintf(b, sizeof(b), "%d", eb);
    snprintf(c, sizeof(c), "%d", db);
    execl(engine, engine, a, b, c, (char *)NULL);
    _exit(127);
  }
  close(in[0]);
  close(out[1]);
  FILE *stream = fdopen(out[0], "rb");
  if (!stream || read_marker(stream, "\1\1READY\1\1\n", 0)) {
    fprintf(stderr, "engine exited while loading\n");
    kill(pid, SIGTERM);
    return 2;
  }
  char line[1024];
  fgets(line, sizeof(line), stream);
  fgets(line, sizeof(line), stream);
  fprintf(stderr, "colibrì chat ready · :reset clears memory · :more continues "
                  "· :q exits\n");
  char *input = NULL;
  size_t input_cap = 0;
  while (1) {
    if (isatty(0)) {
      fputs("› ", stdout);
      fflush(stdout);
    }
    ssize_t n = getline(&input, &input_cap, stdin);
    if (n < 0)
      break;
    while (n && (input[n - 1] == '\n' || input[n - 1] == '\r'))
      input[--n] = 0;
    if (!strcmp(input, ":q") || !strcmp(input, ":quit") ||
        !strcmp(input, "exit"))
      break;
    const char *send = input;
    if (!strcmp(input, ":reset"))
      send = "\2RESET";
    else if (!strcmp(input, ":more"))
      send = "\2MORE";
    if (write_all_fd(in[1], send, strlen(send)) || write_all_fd(in[1], "\n", 1))
      break;
    if (read_marker(stream, "\1\1END\1\1\n", strcmp(input, ":reset"))) {
      fprintf(stderr, "engine exited unexpectedly\n");
      break;
    }
    if (fgets(line, sizeof(line), stream)) {
      if (!strcmp(input, ":reset"))
        fprintf(stderr, "memory cleared\n");
      else if (!strncmp(line, "STAT ", 5))
        fprintf(stderr, "[%s]", line + 5);
    }
    fflush(stdout);
  }
  free(input);
  close(in[1]);
  kill(pid, SIGTERM);
  int status;
  waitpid(pid, &status, 0);
  fclose(stream);
  return 0;
}
