// Minimal libllama tool for scripts/conversion_check.py (llama.cpp 3466812d).
// Built by conversion_check.py:
//   gcc -O2 -o build/llama_logits scripts/llama_logits.c -I ~/llama.cpp/include
//       -I ~/llama.cpp/ggml/include -L ~/llama.cpp/build/bin -lllama -Wl,-rpath,~/llama.cpp/build/bin
// Usage:
//   llama_logits MODEL NGL tokenize IN.txt  OUT.i32      text -> token ids (no BOS, special parsed)
//   llama_logits MODEL NGL logits   IN.i32  OUT.f32      float32 logits [n_tokens x n_vocab], every position
//   llama_logits MODEL NGL greedy   IN.i32  OUT.i32 N    N argmax tokens after the prompt (no stop on EOS)
// Writes n_vocab to stdout (logits mode) so the caller can reshape.
#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char * read_all(const char * path, long * size) {
    FILE * f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char * buf = malloc(*size + 1);
    if (fread(buf, 1, *size, f) != (size_t) *size) { perror("fread"); exit(1); }
    buf[*size] = 0;
    fclose(f);
    return buf;
}

static void write_all(const char * path, const void * data, size_t bytes) {
    FILE * f = fopen(path, "wb");
    if (!f || fwrite(data, 1, bytes, f) != bytes) { perror(path); exit(1); }
    fclose(f);
}

static int argmax(const float * x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: see header\n"); return 2; }
    const char * mode = argv[3];
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[2]);
    struct llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    long size;
    char * in = read_all(argv[4], &size);

    if (strcmp(mode, "tokenize") == 0) {
        int n = -llama_tokenize(vocab, in, (int) size, NULL, 0, false, true);
        llama_token * toks = malloc(sizeof(llama_token) * n);
        if (llama_tokenize(vocab, in, (int) size, toks, n, false, true) != n) { fprintf(stderr, "tokenize\n"); return 1; }
        write_all(argv[5], toks, sizeof(llama_token) * n);
        printf("%d\n", n);
        return 0;
    }

    int n = (int) (size / sizeof(llama_token));
    llama_token * toks = (llama_token *) in;
    int n_gen = (strcmp(mode, "greedy") == 0) ? atoi(argv[6]) : 0;

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + n_gen + 16;
    cp.n_batch = n + 16;
    cp.n_ubatch = n + 16;
    cp.n_seq_max = 1;
    cp.no_perf = true;
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    struct llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = strcmp(mode, "logits") == 0 || i == n - 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    if (strcmp(mode, "logits") == 0) {
        FILE * f = fopen(argv[5], "wb");
        for (int i = 0; i < n; i++) fwrite(llama_get_logits_ith(ctx, i), sizeof(float), n_vocab, f);
        fclose(f);
        printf("%d\n", n_vocab);
    } else if (strcmp(mode, "greedy") == 0) {
        llama_token * out = malloc(sizeof(llama_token) * n_gen);
        llama_token t = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        for (int k = 0; k < n_gen; k++) {
            out[k] = t;
            if (k == n_gen - 1) break;
            if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            t = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        }
        write_all(argv[5], out, sizeof(llama_token) * n_gen);
    } else {
        fprintf(stderr, "bad mode %s\n", mode);
        return 2;
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
