#include "../tok.h"

#include <stdint.h>

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
    int start = atoi(argv[3]), count = atoi(argv[4]);
    if (start < 0 || count < 0 || start > header[6]) {
        fprintf(stderr, "invalid token range (stored=%d)\n", header[6]);
        fclose(f);
        return 1;
    }
    if (count > header[6] - start) count = header[6] - start;
    int *ids = malloc((size_t)(count ? count : 1) * sizeof(*ids));
    if (fseek(f, 40L + (long)start * 4L, SEEK_SET) ||
        fread(ids, 4, (size_t)count, f) != (size_t)count) {
        fprintf(stderr, "short token checkpoint\n");
        fclose(f); free(ids); return 1;
    }
    fclose(f);
    Tok tok;
    tok_load(&tok, argv[1]);
    size_t cap = (size_t)count * 256u + 1u;
    char *text = malloc(cap);
    int n = tok_decode(&tok, ids, count, text, (int)(cap - 1));
    fwrite(text, 1, (size_t)n, stdout);
    free(text); free(ids);
    return 0;
}
