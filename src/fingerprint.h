#pragma once
#include "dsp.h"
#include <stddef.h>

#define FINGERPRINT_FFT 4096
#define FINGERPRINT_MIN_BIN (FINGERPRINT_FFT / 128)
#define FINGERPRINT_MAX_BIN (FINGERPRINT_FFT * 15 / 32)
#define FINGERPRINT_PEAKS 4
#define FINGERPRINT_FRAMES (WINDOW_FRAMES - FINGERPRINT_FFT / FRAME_SIZE + 1)
#define FINGERPRINT_CANDIDATES 16

typedef struct {
    int32_t offset_frames;
    uint16_t reference, class_index;
    uint8_t votes, second_votes;
} catalog_evidence_t;

typedef struct {
    float window[FINGERPRINT_FFT];
    float real[FINGERPRINT_FFT], imag[FINGERPRINT_FFT];
    int16_t pcm[FINGERPRINT_FFT];
    unsigned frames;
} fingerprint_context_t;

void fingerprint_init(fingerprint_context_t *ctx);
void fingerprint_reset(fingerprint_context_t *ctx);
int fingerprint_push(fingerprint_context_t *ctx, const int16_t *pcm, uint16_t *peaks);
size_t fingerprint_extract(const int16_t *pcm, size_t samples, uint16_t *peaks);
void catalog_match(const uint16_t *peaks, float *scores);
unsigned catalog_match_detailed(const uint16_t *peaks, float *scores, catalog_evidence_t *matches);
