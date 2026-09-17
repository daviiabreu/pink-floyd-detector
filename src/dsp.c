#include "dsp.h"
#include <math.h>
#include <string.h>

#define PI_F 3.14159265358979323846f

void dsp_init(dsp_context_t *ctx) {
    for (int i = 0; i < FRAME_SIZE; ++i)
        ctx->window[i] = 0.5f - 0.5f * cosf(2.0f * PI_F * i / (FRAME_SIZE - 1));
    float low = 2595.0f * log10f(1.0f + 80.0f / 700.0f);
    float high = 2595.0f * log10f(1.0f + 7600.0f / 700.0f);
    for (int i = 0; i < MEL_BANDS + 2; ++i) {
        float mel = low + (high - low) * i / (MEL_BANDS + 1);
        ctx->mel_edges[i] = 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    }
    for (int k = 0; k < MFCC_COUNT; ++k)
        for (int m = 0; m < MEL_BANDS; ++m)
            ctx->dct[k][m] = cosf(PI_F * k * (m + 0.5f) / MEL_BANDS) *
                             sqrtf((k == 0 ? 1.0f : 2.0f) / MEL_BANDS);
}

void dsp_fft(float *real, float *imag, unsigned size) {
    for (unsigned i = 1, j = 0; i < size; ++i) {
        unsigned bit = size >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float value = real[i]; real[i] = real[j]; real[j] = value;
        }
    }
    for (unsigned length = 2; length <= size; length <<= 1) {
        float wr = cosf(-2.0f * PI_F / length), wi = sinf(-2.0f * PI_F / length);
        for (unsigned start = 0; start < size; start += length) {
            float ur = 1.0f, ui = 0.0f;
            for (unsigned j = 0; j < length / 2; ++j) {
                int a = start + j, b = a + length / 2;
                float tr = ur * real[b] - ui * imag[b];
                float ti = ur * imag[b] + ui * real[b];
                real[b] = real[a] - tr; imag[b] = imag[a] - ti;
                real[a] += tr; imag[a] += ti;
                float next = ur * wr - ui * wi;
                ui = ur * wi + ui * wr; ur = next;
            }
        }
    }
}

void dsp_frame(dsp_context_t *ctx, const int16_t *pcm, float *features) {
    float mean = 0.0f, power = 0.0f;
    for (int i = 0; i < FRAME_SIZE; ++i) mean += pcm[i] / 32768.0f;
    mean /= FRAME_SIZE;
    for (int i = 0; i < FRAME_SIZE; ++i) {
        float sample = pcm[i] / 32768.0f - mean;
        ctx->real[i] = sample;
        power += sample * sample;
    }
    float rms = sqrtf(power / FRAME_SIZE);
    features[0] = log10f(fmaxf(rms, 1.0e-6f));
    for (int i = 0; i < FRAME_SIZE; ++i) {
        ctx->real[i] *= ctx->window[i] / fmaxf(rms, 1.0e-6f);
        ctx->imag[i] = 0.0f;
    }
    dsp_fft(ctx->real, ctx->imag, FRAME_SIZE);
    float total = 0.0f, weighted = 0.0f;
    for (int i = 0; i <= FRAME_SIZE / 2; ++i) {
        float value = ctx->real[i] * ctx->real[i] + ctx->imag[i] * ctx->imag[i];
        ctx->real[i] = value / (FRAME_SIZE * FRAME_SIZE);
        float magnitude = sqrtf(ctx->real[i]);
        total += magnitude;
        weighted += magnitude * i * SAMPLE_RATE / FRAME_SIZE;
    }
    features[1] = total > 1.0e-10f ? weighted / total : 0.0f;
    float mel[MEL_BANDS];
    for (int m = 0; m < MEL_BANDS; ++m) {
        float energy = 0.0f;
        float left = ctx->mel_edges[m], center = ctx->mel_edges[m + 1];
        float right = ctx->mel_edges[m + 2];
        for (int i = 0; i <= FRAME_SIZE / 2; ++i) {
            float hz = (float)i * SAMPLE_RATE / FRAME_SIZE;
            float weight = fmaxf(0.0f, fminf((hz - left) / (center - left),
                                           (right - hz) / (right - center)));
            energy += ctx->real[i] * weight;
        }
        mel[m] = logf(fmaxf(energy, 1.0e-10f));
    }
    for (int k = 0; k < MFCC_COUNT; ++k) {
        features[k + 2] = 0.0f;
        for (int m = 0; m < MEL_BANDS; ++m) features[k + 2] += ctx->dct[k][m] * mel[m];
    }
}

void features_reset(feature_accumulator_t *acc) { memset(acc, 0, sizeof(*acc)); }

int features_push(feature_accumulator_t *acc, const float *frame, float *output) {
    ++acc->count;
    for (int i = 0; i < FRAME_FEATURES; ++i) {
        float delta = frame[i] - acc->mean[i];
        acc->mean[i] += delta / acc->count;
        acc->m2[i] += delta * (frame[i] - acc->mean[i]);
    }
    if (acc->count < WINDOW_FRAMES) return 0;
    for (int i = 0; i < FRAME_FEATURES; ++i) {
        output[i] = acc->mean[i];
        output[i + FRAME_FEATURES] = sqrtf(fmaxf(0.0f, acc->m2[i] / WINDOW_FRAMES));
    }
    features_reset(acc);
    return 1;
}

int extract_window(const int16_t *pcm, float *output) {
    dsp_context_t ctx;
    feature_accumulator_t acc;
    float frame[FRAME_FEATURES];
    dsp_init(&ctx);
    features_reset(&acc);
    for (int i = 0; i < WINDOW_FRAMES; ++i) {
        dsp_frame(&ctx, pcm + i * FRAME_SIZE, frame);
        features_push(&acc, frame, output);
    }
    return FEATURE_COUNT;
}
