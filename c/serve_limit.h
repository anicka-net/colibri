#ifndef COLI_SERVE_LIMIT_H
#define COLI_SERVE_LIMIT_H

static inline int coli_effective_generation_limit(int requested, int room) {
    return requested < room ? requested : room;
}

static inline int coli_generation_limit_reached(int emitted, int effective) {
    return effective <= 0 || emitted >= effective;
}

#endif
