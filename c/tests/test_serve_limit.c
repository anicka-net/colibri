#include "../serve_limit.h"
#include <stdio.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main(void) {
    int effective = coli_effective_generation_limit(32000, 9063);
    CHECK(effective == 9063);
    CHECK(!coli_generation_limit_reached(24, effective));
    CHECK(coli_generation_limit_reached(9063, effective));
    CHECK(coli_generation_limit_reached(0, 0));
    CHECK(coli_effective_generation_limit(64, 9063) == 64);
    puts("serve generation-limit semantics: ok");
    return 0;
}
