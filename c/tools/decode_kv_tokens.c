#include "../tok.h"

#include <errno.h>
#include <stdint.h>

static int seek64(FILE *f, int64_t off, int whence) {
#ifdef _WIN32
    return _fseeki64(f, off, whence);
#else
    return fseeko(f, (off_t)off, whence);
#endif
}

static int64_t tell64(FILE *f) {
#ifdef _WIN32
    return _ftelli64(f);
#else
    return (int64_t)ftello(f);
#endif
}

static int parse_nonnegative(const char *s, int *out) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end || v < 0 || v > INT_MAX)
        return 0;
    *out = (int)v;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s TOKENIZER TOK_FILE START COUNT\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 1; }
    char magic[8];
    int32_t header[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "COLIKT1\0", 8) ||
        fread(header, 4, 8, f) != 8 || header[6] < 0) {
        fprintf(stderr, "invalid Colibri token checkpoint\n");
        fclose(f);
        return 1;
    }
    int start, count;
    if (!parse_nonnegative(argv[3], &start) ||
        !parse_nonnegative(argv[4], &count) || start > header[6]) {
        fprintf(stderr, "invalid token range (stored=%d)\n", header[6]);
        fclose(f);
        return 1;
    }
    if (seek64(f, 0, SEEK_END) || tell64(f) < 40 + (int64_t)header[6] * 4) {
        fprintf(stderr, "short token checkpoint\n");
        fclose(f);
        return 1;
    }
    if (count > header[6] - start) count = header[6] - start;
    if (count > (INT_MAX - 1) / 256) {
        fprintf(stderr, "token range is too large to decode at once\n");
        fclose(f);
        return 1;
    }
    int *ids = malloc((size_t)(count ? count : 1) * sizeof(*ids));
    if (!ids) {
        fprintf(stderr, "out of memory reading %d token ids\n", count);
        fclose(f);
        return 1;
    }
    if (seek64(f, 40 + (int64_t)start * 4, SEEK_SET) ||
        fread(ids, 4, (size_t)count, f) != (size_t)count) {
        fprintf(stderr, "short token checkpoint\n");
        fclose(f); free(ids); return 1;
    }
    fclose(f);
    size_t cap = (size_t)count * 256u + 1u;
    char *text = malloc(cap);
    if (!text) {
        fprintf(stderr, "out of memory decoding %d tokens\n", count);
        free(ids);
        return 1;
    }
    Tok tok;
    tok_load(&tok, argv[1]);
    int n = tok_decode(&tok, ids, count, text, (int)(cap - 1));
    fwrite(text, 1, (size_t)n, stdout);
    free(text); free(ids);
    return 0;
}
