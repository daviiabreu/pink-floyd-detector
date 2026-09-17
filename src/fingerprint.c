#include "fingerprint.h"
#include <math.h>
#include <string.h>

void fingerprint_reset(fingerprint_context_t *ctx) { ctx->frames = 0; }

void fingerprint_init(fingerprint_context_t *ctx) {
    fingerprint_reset(ctx);
    for (int i = 0; i < FINGERPRINT_FFT; ++i)
        ctx->window[i] = 0.5f - 0.5f * cosf(6.2831853071795864769f * i / (FINGERPRINT_FFT - 1));
}

int fingerprint_push(fingerprint_context_t *ctx, const int16_t *pcm, uint16_t *peaks) {
    if (ctx->frames < FINGERPRINT_FFT / FRAME_SIZE) {
        memcpy(ctx->pcm + ctx->frames * FRAME_SIZE, pcm, FRAME_SIZE * sizeof(*pcm));
        if (++ctx->frames < FINGERPRINT_FFT / FRAME_SIZE) return 0;
    } else {
        memmove(ctx->pcm, ctx->pcm + FRAME_SIZE, (FINGERPRINT_FFT - FRAME_SIZE) * sizeof(*pcm));
        memcpy(ctx->pcm + FINGERPRINT_FFT - FRAME_SIZE, pcm, FRAME_SIZE * sizeof(*pcm));
    }
    float mean = 0.0f;
    for (int i = 0; i < FINGERPRINT_FFT; ++i) mean += ctx->pcm[i] / 32768.0f;
    mean /= FINGERPRINT_FFT;
    for (int i = 0; i < FINGERPRINT_FFT; ++i) {
        ctx->real[i] = (ctx->pcm[i] / 32768.0f - mean) * ctx->window[i];
        ctx->imag[i] = 0.0f;
    }
    dsp_fft(ctx->real, ctx->imag, FINGERPRINT_FFT);
    for (int i = 0; i <= FINGERPRINT_FFT / 2; ++i)
        ctx->real[i] = ctx->real[i] * ctx->real[i] + ctx->imag[i] * ctx->imag[i];
    for (int i = 0; i <= FINGERPRINT_FFT / 2; ++i) {
        ctx->imag[i] = i >= FINGERPRINT_MIN_BIN && i <= FINGERPRINT_MAX_BIN && ctx->real[i] > ctx->real[i - 1] &&
                      ctx->real[i] >= ctx->real[i + 1] ? ctx->real[i] : 0.0f;
    }
    for (int p = 0; p < FINGERPRINT_PEAKS; ++p) {
        int best = 0;
        for (int i = FINGERPRINT_MIN_BIN; i <= FINGERPRINT_MAX_BIN; ++i)
            if (ctx->imag[i] > ctx->imag[best]) best = i;
        peaks[p] = ctx->imag[best] > 1e-10f ? (uint16_t)best : 0;
        for (int i = best - 5; i <= best + 5; ++i)
            if (i >= 0 && i <= FINGERPRINT_FFT / 2) ctx->imag[i] = 0.0f;
    }
    return 1;
}

size_t fingerprint_extract(const int16_t *pcm, size_t samples, uint16_t *peaks) {
    fingerprint_context_t ctx;
    fingerprint_init(&ctx);
    size_t count = 0;
    for (size_t offset = 0; offset + FRAME_SIZE <= samples; offset += FRAME_SIZE)
        count += fingerprint_push(&ctx, pcm + offset, peaks + count * FINGERPRINT_PEAKS);
    return count;
}
