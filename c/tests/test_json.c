#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../json.h"

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,         \
              #condition);                                                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(void) {
  jval *root = json_parse(
      "{\"name\":\"Colibri\\nCPU\",\"enabled\":true,\"empty\":null,"
      "\"values\":[1,-2.5,3e2],\"unicode\":\"\\u03bb \\uD83D\\uDE80\"}",
      NULL);

  CHECK(root && root->t == J_OBJ);
  CHECK(strcmp(json_get(root, "name")->str, "Colibri\nCPU") == 0);
  CHECK(json_get(root, "enabled")->boolean == 1);
  CHECK(json_get(root, "empty")->t == J_NULL);
  CHECK(json_get(root, "missing") == NULL);

  jval *values = json_get(root, "values");
  CHECK(values->t == J_ARR && values->len == 3);
  CHECK(values->kids[0]->num == 1.0);
  CHECK(values->kids[1]->num == -2.5);
  CHECK(values->kids[2]->num == 300.0);
  CHECK(strcmp(json_get(root, "unicode")->str, "λ 🚀") == 0);
  json_free(root);

  const char *invalid[] = {
      "{",           "{\"x\":1,}",         "[1,]",        "{\"x\":tru}",
      "1 trailing",  "\"unterminated",     "\"bad\\q\"",  "\"bad\ncontrol\"",
      "\"\\uD800\"", "\"\\uD800\\u0041\"", "\"\\uDC00\"", "NaN",
      "Infinity"};
  for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++)
    CHECK(json_parse(invalid[i], NULL) == NULL);

  puts("json tests: ok");
  return 0;
}
