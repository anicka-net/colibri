#include "../expert_vram_plan.h"
#include <math.h>
#include <stdio.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main(void) {
    double a = coli_vram_device_budget(150e9, 90e9, 180e9);
    double b = coli_vram_device_budget(150e9, 90e9, 180e9);
    CHECK(fabs(a - 75e9) < 1.0);
    CHECK(fabs(b - 75e9) < 1.0);
    a = coli_vram_device_budget(120e9, 60e9, 150e9);
    b = coli_vram_device_budget(120e9, 90e9, 150e9);
    CHECK(fabs(a - 48e9) < 1.0);
    CHECK(fabs(b - 72e9) < 1.0);
    CHECK(coli_vram_device_budget(10, 0, 10) == 0);
    puts("expert VRAM per-device budget: ok");
    return 0;
}
