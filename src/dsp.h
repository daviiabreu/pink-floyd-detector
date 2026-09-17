#pragma once
#include <stdint.h>

#define SAMPLE_RATE 16000
#define FRAME_SIZE 512
#define WINDOW_FRAMES 64
#define WINDOW_SAMPLES (FRAME_SIZE * WINDOW_FRAMES)
#define MEL_BANDS 26
#define MFCC_COUNT 13
#define FRAME_FEATURES (2 + MFCC_COUNT)
#define FEATURE_COUNT (2 * FRAME_FEATURES)

typedef struct {
    float window[FRAME_SIZE];
    float real[FRAME_SIZE];
    float imag[FRAME_SIZE];
    float mel_edges[MEL_BANDS + 2];
    float dct[MFCC_COUNT][MEL_BANDS];
} dsp_context_t;

typedef struct {
    float mean[FRAME_FEATURES];
    float m2[FRAME_FEATURES];
    unsigned count;
} feature_accumulator_t;

void dsp_init(dsp_context_t *ctx);
void dsp_fft(float *real, float *imag, unsigned size);
void dsp_frame(dsp_context_t *ctx, const int16_t *pcm, float *features);
void features_reset(feature_accumulator_t *acc);
int features_push(feature_accumulator_t *acc, const float *frame, float *output);
int extract_window(const int16_t *pcm, float *output);
