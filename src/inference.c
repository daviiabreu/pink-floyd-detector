#include "inference.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef MODEL_CATALOG_ID
#include "catalog_data.h"
_Static_assert(MODEL_CATALOG_ID == CATALOG_ID, "Rebuild the model for this catalog");
_Static_assert(MODEL_FEATURES == FEATURE_COUNT + 2 * CATALOG_SONGS, "Model/catalog feature mismatch");
#else
_Static_assert(MODEL_FEATURES == FEATURE_COUNT, "Model/DSP feature mismatch");
#endif

int model_feature_count(void) { return MODEL_FEATURES; }

#if MODEL_STORAGE == 16
static float half_to_float(uint16_t half) {
    uint32_t sign = ((uint32_t)half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 31u;
    uint32_t fraction = half & 1023u;
    uint32_t bits;
    if (exponent == 0) {
        if (fraction == 0) bits = sign;
        else {
            int shift = 0;
            while ((fraction & 1024u) == 0) { fraction <<= 1; ++shift; }
            bits = sign | ((uint32_t)(113 - shift) << 23) | ((fraction & 1023u) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (fraction << 13);
    } else bits = sign | ((exponent + 112u) << 23) | (fraction << 13);
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}
#endif

const char *model_precision_name(void) {
#if MODEL_STORAGE == 16
    return "fp16_weights";
#elif MODEL_STORAGE == -16
    return "int16";
#elif MODEL_STORAGE == 8
    return "int8";
#elif MODEL_STORAGE == 4
    return "int4_weights_int8_activations";
#else
    return "fp32";
#endif
}

void model_predict(const float *features, float *probabilities) {
    float scaled[MODEL_FEATURES];
#if MODEL_STORAGE == -16 || MODEL_STORAGE == 8 || MODEL_STORAGE == 4
    int32_t quantized[MODEL_FEATURES];
#endif
    for (int f = 0; f < MODEL_FEATURES; ++f) {
        scaled[f] = (features[f] - MODEL_MEAN[f]) / MODEL_SCALE[f];
        if (!isfinite(scaled[f])) {
            for (int c = 0; c < MODEL_CLASSES; ++c) probabilities[c] = NAN;
            return;
        }
#if MODEL_STORAGE == -16 || MODEL_STORAGE == 8 || MODEL_STORAGE == 4
        float value = fminf(MODEL_INPUT_QMAX, fmaxf(-MODEL_INPUT_QMAX, scaled[f] / MODEL_INPUT_SCALE));
        quantized[f] = (int32_t)nearbyintf(value);
#endif
    }
    float max_logit = -INFINITY;
    for (int c = 0; c < MODEL_CLASSES; ++c) {
        float logit = MODEL_BIAS[c];
#if MODEL_STORAGE == -16
        int64_t accumulator = 0;
        for (int f = 0; f < MODEL_FEATURES; ++f)
            accumulator += (int64_t)quantized[f] * MODEL_WEIGHTS_Q[f][c];
        logit += (float)accumulator * MODEL_INPUT_SCALE * MODEL_WEIGHT_SCALE[c];
#elif MODEL_STORAGE == 8 || MODEL_STORAGE == 4
        int32_t accumulator = 0;
        for (int f = 0; f < MODEL_FEATURES; ++f) {
#if MODEL_STORAGE == 4
            unsigned index = f * MODEL_CLASSES + c;
            unsigned nibble = (MODEL_WEIGHTS_PACKED[index / 2] >> (4 * (index % 2))) & 15u;
            int32_t weight = nibble >= 8 ? (int32_t)nibble - 16 : (int32_t)nibble;
#else
            int32_t weight = MODEL_WEIGHTS_Q[f][c];
#endif
            accumulator += quantized[f] * weight;
        }
        logit += (float)accumulator * MODEL_INPUT_SCALE * MODEL_WEIGHT_SCALE[c];
#elif MODEL_STORAGE == 16
        for (int f = 0; f < MODEL_FEATURES; ++f)
            logit += scaled[f] * half_to_float(MODEL_WEIGHTS_HALF[f][c]);
#else
        for (int f = 0; f < MODEL_FEATURES; ++f)
            logit += scaled[f] * MODEL_WEIGHTS[f][c];
#endif
        probabilities[c] = logit;
        max_logit = fmaxf(max_logit, logit);
    }
    float total = 0.0f, correction = 0.0f;
    for (int c = 0; c < MODEL_CLASSES; ++c) {
        probabilities[c] = expf(probabilities[c] - max_logit);
        float adjusted = probabilities[c] - correction;
        float next = total + adjusted;
        correction = (next - total) - adjusted;
        total = next;
    }
    for (int c = 0; c < MODEL_CLASSES; ++c) probabilities[c] /= total;
}

void model_predict_batch(const float *features, float *probabilities, size_t rows) {
    for (size_t row = 0; row < rows; ++row)
        model_predict(features + row * MODEL_FEATURES, probabilities + row * MODEL_CLASSES);
}

float model_benchmark(const float *features, size_t rows, unsigned repetitions) {
    float output[MODEL_CLASSES], checksum = 0.0f;
    for (unsigned i = 0; i < repetitions; ++i)
        for (size_t row = 0; row < rows; ++row) {
            model_predict(features + row * MODEL_FEATURES, output);
            checksum += output[i % MODEL_CLASSES];
        }
    return checksum;
}

int model_candidate(const float *features, const float *probabilities) {
    int best = 0;
    float second = 0.0f;
    for (int c = 0; c < MODEL_CLASSES; ++c) {
        if (!isfinite(probabilities[c])) return -1;
        if (probabilities[c] > probabilities[best]) best = c;
    }
    for (int c = 0; c < MODEL_CLASSES; ++c)
        if (c != best) second = fmaxf(second, probabilities[c]);
    if (!isfinite(features[0]) || features[0] < MODEL_MIN_LOG_RMS ||
        best == MODEL_BACKGROUND_CLASS || probabilities[best] < MODEL_THRESHOLD ||
        probabilities[best] - second < MODEL_MARGIN) return -1;
    return best;
}
