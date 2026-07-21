#ifndef COLI_REMOTE_EXPERT_PROTOCOL_H
#define COLI_REMOTE_EXPERT_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#define COLI_REMOTE_MAGIC 0x434f4c52u
#define COLI_REMOTE_VERSION 1u
#define COLI_REMOTE_PACK_MAGIC "COLIRXP1"
#define COLI_REMOTE_MAX_EXPERTS 32
#define COLI_REMOTE_MAX_ROWS 32
#define COLI_REMOTE_MAX_S 4
#define COLI_REMOTE_HIDDEN 6144
#define COLI_REMOTE_INTER 2048

typedef struct {
    uint32_t magic, version, seq, layer, S, count, total_rows;
    int32_t eids[COLI_REMOTE_MAX_EXPERTS];
    int32_t rows[COLI_REMOTE_MAX_EXPERTS];
    int32_t tokrow[COLI_REMOTE_MAX_ROWS];
} ColiRemoteRequest;

typedef struct {
    uint32_t magic, version, seq, status, count, total_rows;
} ColiRemoteResponse;

typedef struct {
    char magic[8];
    uint32_t version, hidden, inter, count;
    uint64_t records_offset, data_offset;
} ColiRemotePackHeader;

typedef struct {
    uint16_t layer, eid;
    uint32_t reserved;
    uint64_t offsets[6];
    uint64_t sizes[6];
} ColiRemotePackRecord;

#define COLI_REMOTE_REQUEST_BYTES \
    (sizeof(ColiRemoteRequest) + \
     (size_t)COLI_REMOTE_MAX_S * COLI_REMOTE_HIDDEN * sizeof(float))
#define COLI_REMOTE_RESPONSE_BYTES \
    (sizeof(ColiRemoteResponse) + \
     (size_t)COLI_REMOTE_MAX_ROWS * COLI_REMOTE_HIDDEN * sizeof(float))

#endif
