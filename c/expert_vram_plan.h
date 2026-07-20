#ifndef COLI_EXPERT_VRAM_PLAN_H
#define COLI_EXPERT_VRAM_PLAN_H

/* Split a total expert budget proportionally across the safe capacity of each
   device. This prevents routing heat from consuming another device's scratch
   reserve while still supporting heterogeneous GPUs. */
static inline double coli_vram_device_budget(double total_budget,
                                              double safe_device,
                                              double safe_total) {
    if (total_budget <= 0 || safe_device <= 0 || safe_total <= 0) return 0;
    double share = total_budget * safe_device / safe_total;
    return share < safe_device ? share : safe_device;
}

static inline double coli_vram_ram_charge(double bytes, int integrated) {
    return bytes > 0 && integrated ? bytes : 0;
}

#endif
