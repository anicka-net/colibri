#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
  (void)getenv("SNAP");
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("\1\1READY\1\1\nSTAT 0 0.00 0.0 0.01\n");
  if (!getenv("SERVE_BATCH")) {
    printf("TIERS 2 3 4 5.0 6.0\n");
    char input[256];
    while (fgets(input, sizeof(input), stdin)) {
      if (!strncmp(input, "\2RESET", 6))
        printf("\1\1END\1\1\nSTAT 0 0 0 0\n");
      else
        printf("chat:%s\1\1END\1\1\nSTAT 2 10.0 80.0 0.02\n", input);
    }
    return 0;
  }
  printf(
      "TIERS 2 3 4 5.0 6.0\nHWINFO 8 16.0 12.0 1 24.0 Test CPU | Test GPU\n");
  char line[256];
  while (fgets(line, sizeof(line), stdin)) {
    unsigned long long id;
    int slot, max;
    size_t bytes;
    double temp, top_p;
    if (sscanf(line, "SUBMIT %llu %d %zu %d %lf %lf", &id, &slot, &bytes, &max,
               &temp, &top_p) == 6) {
      char *prompt = malloc(bytes + 1);
      if (fread(prompt, 1, bytes, stdin) != bytes)
        return 2;
      prompt[bytes] = 0;
      (void)fgetc(stdin);
      if (strstr(prompt, "exit-engine")) {
        free(prompt);
        return 0;
      }
      char ctx_reply[64];
      snprintf(ctx_reply, sizeof(ctx_reply), "CTX=%s",
               getenv("CTX") ? getenv("CTX") : "unset");
      char bind_reply[64];
      snprintf(bind_reply, sizeof(bind_reply), "OMP_PROC_BIND=%s",
               getenv("OMP_PROC_BIND") ? getenv("OMP_PROC_BIND") : "unset");
      const char *reply =
          strstr(prompt, "check headers")
              ? (strstr(prompt, "x-anthropic-") ? "headers-leaked"
                                                   : "headers-stripped")
          : strstr(prompt, "check authored system")
              ? (strstr(prompt, "Authorization: keep this") &&
                         strstr(prompt, "x-user-authored: keep this")
                     ? "system-preserved"
                     : "system-lost")
          : strstr(prompt, "show bind") ? bind_reply
          : strstr(prompt, "show ctx") ? ctx_reply
          : strstr(prompt, "check reference")
              ? (strstr(prompt, "DEFERRED_SENTINEL") ? "reference-expanded"
                                                     : "reference-missing")
          : strstr(prompt, "check defer")
              ? (strstr(prompt, "DEFERRED_SENTINEL") ? "deferred-leaked"
                                                     : "deferred-ok")
          : strstr(prompt, "native tool syntax")
              ? "I'll look that up.<tool_call>lookup(**q**: \"finch\")"
          : strstr(prompt, "# Tools")
              ? "<tool_call>lookup<arg_key>q</arg_key><arg_value>bird</"
                "arg_value></tool_call>"
          : (strstr(prompt, "<think>") && !strstr(prompt, "<think></think>"))
              ? "Reasoning</think>Answer"
          : strstr(prompt, "hello") ? "Hello from C"
                                    : "OK";
      if (strstr(prompt, "split utf8")) {
        printf("DATA %llu 1\n\xc3\n", id);
        printf("DATA %llu 1\n\xa9\n", id);
        reply = NULL;
      } else if (strstr(prompt, "fragmented tool")) {
        const char *parts[] = {
            "Before ", "<tool_", "call>lookup<arg_key>q</arg_key><arg_",
            "value>bird</arg_value></tool_call>", " After"};
        for (int i = 0; i < 5; i++) {
          printf("DATA %llu %zu\n%s\n", id, strlen(parts[i]), parts[i]);
          if (i == 0)
            usleep(500000);
        }
        reply = NULL;
      } else if (strstr(prompt, "slow")) {
        printf("DATA %llu 5\nfirst\n", id);
        usleep(500000);
        printf("DATA %llu 6\nsecond\n", id);
        reply = NULL;
      }
      if (reply)
        printf("DATA %llu %zu\n%s\n", id, strlen(reply), reply);
      printf("DONE %llu STAT 3 9.5 80.0 0.02 7 0\n", id);
      printf("PROF 1.0 7 3 .1 .2 .3 .4 .5 4\n");
      free(prompt);
    } else if (!strncmp(line, "CANCEL ", 7)) {
      sscanf(line, "CANCEL %llu", &id);
      printf("ERROR %llu CANCELLED\n", id);
    }
  }
  return 0;
}
