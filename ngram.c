/*
Compile and run:
clang -O3 -Wall -Wextra -Wpedantic -fsanitize=address -fsanitize=undefined -o ngram ngram.c && ./ngram
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

// ----------------------------------------------------------------------------
// utils

FILE *fopen_check(const char *path, const char *mode, const char *file, int line) {
    FILE *fp = fopen(path, mode);
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
        exit(EXIT_FAILURE);
    }
    return fp;
}
#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

size_t powi(int base, int exp) {
    // integer exponentiation utility
    size_t result = 1;
    for (int i = 0; i < exp; i++) {
        result *= base;
    }
    return result;
}

// ----------------------------------------------------------------------------
// random number generation

uint32_t random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * 0x2545F4914F6CDD1Dull) >> 32);
}

float random_f32(uint64_t *state) {
    // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

// ----------------------------------------------------------------------------
// sampling

int sample_discrete(const float* probs, const int n, const float coinf) {
    // sample from a discrete distribution
    assert(coinf >= 0.0f && coinf < 1.0f);
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        float probs_i = probs[i];
        assert(probs_i >= 0.0f && probs_i <= 1.0f);
        cdf += probs_i;
        if (coinf < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// tokenizer: convert strings <---> 1D integer sequences

// 26 lowercase letters + 1 end-of-text token
#define NUM_TOKENS 27
#define EOT_TOKEN 0

int tokenizer_encode(const char c) {
    // characters a-z are encoded as 1-26, and '\n' is encoded as 0
    assert(c == '\n' || ('a' <= c && c <= 'z'));
    int token = (c == '\n') ? EOT_TOKEN : (c - 'a' + 1);
    return token;
}

char tokenizer_decode(const int token) {
    // tokens 0-25 are decoded as a-z, and token 26 is decoded as '\n'
    assert(token >= 0 && token < NUM_TOKENS);
    char c = (token == EOT_TOKEN) ? '\n' : 'a' + (token - 1);
    return c;
}

// ----------------------------------------------------------------------------
// ngram model

typedef struct {
    // hyperparameters
    int seq_len;
    int vocab_size;
    float smoothing;
    // parameters
    size_t num_counts; // size_t because int would only handle up to 2^31-1 ~= 2 billion counts
    uint32_t* counts;
    // internal buffer for ravel_index
    int* ravel_buffer;
} NgramModel;

void ngram_init(NgramModel *model, const int vocab_size, const int seq_len, const float smoothing) {
    // sanity check and store the hyperparameters
    assert(vocab_size > 0);
    assert(seq_len >= 1 && seq_len <= 6); // sanity check max ngram size we'll handle
    model->vocab_size = vocab_size;
    model->seq_len = seq_len;
    model->smoothing = smoothing;
    // allocate and init memory for counts (np.zeros in numpy)
    model->num_counts = powi(vocab_size, seq_len);
    model->counts = (uint32_t*)mallocCheck(model->num_counts * sizeof(uint32_t));
    for (size_t i = 0; i < model->num_counts; i++) {
        model->counts[i] = 0;
    }
    // allocate buffer we will use for ravel_index
    model->ravel_buffer = (int*)mallocCheck(seq_len * sizeof(int));
}

size_t ravel_index(const int* index, const int n, const int dim) {
    // convert an n-dimensional index into a 1D index (ravel_multi_index in numpy)
    // each index[i] is in the range [0, dim)
    size_t index1d = 0;
    size_t multiplier = 1;
    for (int i = n - 1; i >= 0; i--) {
        int ix = index[i];
        assert(ix >= 0 && ix < dim);
        index1d += multiplier * ix;
        multiplier *= dim;
    }
    return index1d;
}

void ngram_train(NgramModel *model, const int* tape) {
    // tape here is of length `seq_len`, and we want to update the counts
    size_t offset = ravel_index(tape, model->seq_len, model->vocab_size);
    assert(offset >= 0 && offset < model->num_counts);
    model->counts[offset]++;
}

void ngram_inference(NgramModel *model, const int* tape, float* probs) {
    // here, tape is of length `seq_len - 1`, and we want to predict the next token
    // probs should be a pre-allocated buffer of size `vocab_size`

    // copy the tape into the buffer and set the last element to zero
    for (int i = 0; i < model->seq_len - 1; i++) {
        model->ravel_buffer[i] = tape[i];
    }
    model->ravel_buffer[model->seq_len - 1] = 0;
    // find the offset into the counts array based on the context
    size_t offset = ravel_index(model->ravel_buffer, model->seq_len, model->vocab_size);
    // seek to the row of counts for this context
    uint32_t* counts_row = model->counts + offset;

    // calculate the sum of counts in the row
    float row_sum = model->vocab_size * model->smoothing;
    for (int i = 0; i < model->vocab_size; i++) {
        row_sum += counts_row[i];
    }
    if (row_sum == 0.0f) {
        // the entire row of counts is zero, so let's set uniform probabilities
        float uniform_prob = 1.0f / model->vocab_size;
        for (int i = 0; i < model->vocab_size; i++) {
            probs[i] = uniform_prob;
        }
    } else {
        // normalize the row of counts into probabilities
        float scale = 1.0f / row_sum;
        for (int i = 0; i < model->vocab_size; i++) {
            float counts_i = counts_row[i] + model->smoothing;
            probs[i] = scale * counts_i;
        }
    }
}

void ngram_free(NgramModel *model) {
    free(model->counts);
    free(model->ravel_buffer);
}

// ----------------------------------------------------------------------------
// tape stores a fixed window of tokens, functions like a finite queue

typedef struct {
    int n;
    int length;
    int* buffer;
} Tape;

void tape_init(Tape *tape, const int length) {
    // we will allow a buffer of length 0, useful for the Unigram model
    assert(length >= 0);
    tape->length = length;
    tape->n = 0; // counts the number of elements in the buffer up to max
    tape->buffer = NULL;
    if (length > 0) {
        tape->buffer = (int*)mallocCheck(length * sizeof(int));
    }
}

void tape_set(Tape *tape, const int val) {
    for (int i = 0; i < tape->length; i++) {
        tape->buffer[i] = val;
    }
}

int tape_update(Tape *tape, const int token) {
    // returns 1 if the tape is ready/full, 0 otherwise
    if (tape->length == 0) {
        return 1; // unigram tape is always ready
    }
    // shift all elements to the left by one
    for (int i = 0; i < tape->length - 1; i++) {
        tape->buffer[i] = tape->buffer[i + 1];
    }
    // add the new token to the end (on the right)
    tape->buffer[tape->length - 1] = token;
    // keep track of when we've filled the tape
    if (tape->n < tape->length) {
        tape->n++;
    }
    return (tape->n == tape->length);
}

void tape_free(Tape *tape) {
    free(tape->buffer);
}

// ----------------------------------------------------------------------------
// dataloader: iterates all windows of a given length in a text file

typedef struct {
    FILE *file;
    int seq_len;
    Tape tape;
} DataLoader;

void dataloader_init(DataLoader *dataloader, const char *path, const int seq_len) {
    dataloader->file = fopenCheck(path, "r");
    dataloader->seq_len = seq_len;
    tape_init(&dataloader->tape, seq_len);
}

int dataloader_next(DataLoader *dataloader) {
    // returns 1 if a new window was read, 0 if the end of the file was reached
    int c;
    while (1) {
        c = fgetc(dataloader->file);
        if (c == EOF) {
            break;
        }
        int token = tokenizer_encode(c);
        int ready = tape_update(&dataloader->tape, token);
        if (ready) {
            return 1;
        }
    }
    return 0;
}

void dataloader_free(DataLoader *dataloader) {
    fclose(dataloader->file);
    tape_free(&dataloader->tape);
}

// ----------------------------------------------------------------------------

void error_usage(void) {
    fprintf(stderr, "Usage:   ./ngram [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -n <int>    n-gram model arity (default 4)\n");
    fprintf(stderr, "  -s <float>  smoothing factor (default 0.1)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // the arity of the n-gram model (1 = unigram, 2 = bigram, 3 = trigram, ...)
    int seq_len = 4;
    float smoothing = 0.1f;

    // simple argparse, example usage: ./ngram -n 4 -s 0.1
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (!(strlen(argv[i]) == 2)) { error_usage(); } // must be -x (one dash, one letter)
        if (argv[i][1] == 'n') { seq_len = atoi(argv[i+1]); }
        else if (argv[i][1] == 's') { smoothing = atof(argv[i+1]); }
        else { error_usage(); }
    }

    // init the model
    NgramModel model;
    ngram_init(&model, NUM_TOKENS, seq_len, smoothing);

    // train the model
    DataLoader train_loader;
    dataloader_init(&train_loader, "data/train.txt", seq_len);
    while (dataloader_next(&train_loader)) {
        ngram_train(&model, train_loader.tape.buffer);
    }
    dataloader_free(&train_loader);

    // allocate probs buffer for inference
    float* probs = (float*)mallocCheck(NUM_TOKENS * sizeof(float));

    // sample from the model for 200 time steps
    Tape sample_tape;
    tape_init(&sample_tape, seq_len - 1);
    tape_set(&sample_tape, EOT_TOKEN); // fill with EOT tokens to init
    uint64_t rng = 1337;
    for (int i = 0; i < 200; i++) {
        ngram_inference(&model, sample_tape.buffer, probs);
        float coinf = random_f32(&rng);
        int token = sample_discrete(probs, NUM_TOKENS, coinf);
        tape_update(&sample_tape, token);
        char c = tokenizer_decode(token);
        printf("%c", c);
    }
    printf("\n");

    // evaluate the test split loss
    DataLoader test_loader;
    dataloader_init(&test_loader, "data/test.txt", seq_len);
    float sum_loss = 0.0f;
    int count = 0;
    while (dataloader_next(&test_loader)) {
        // note that ngram_inference will only use the first seq_len - 1 tokens in buffer
        ngram_inference(&model, test_loader.tape.buffer, probs);
        // and the last token in the tape buffer is the label
        int target = test_loader.tape.buffer[seq_len - 1];
        // negative log likelihood loss
        sum_loss += -logf(probs[target]);
        count++;
    }
    dataloader_free(&test_loader);
    float mean_loss = sum_loss / count;
    float test_perplexity = expf(mean_loss);
    printf("test_loss %f, test_perplexity %f\n", mean_loss, test_perplexity);

    // clean ups
    ngram_free(&model);
    free(probs);
    tape_free(&sample_tape);
    return EXIT_SUCCESS;
}
