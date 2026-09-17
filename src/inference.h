#pragma once
#include "dsp.h"
#include <stddef.h>
#if MODEL_PRECISION == 16
#include "../model/variants/fp16.h"
#elif MODEL_PRECISION == -16
#include "../model/variants/int16.h"
#elif MODEL_PRECISION == 8
#include "../model/variants/int8.h"
#elif MODEL_PRECISION == 4
#include "../model/variants/int4.h"
#else
#include "model_data.h"
#endif
#ifndef MODEL_STORAGE
#define MODEL_STORAGE 32
#endif
void model_predict(const float *features, float *probabilities);
int model_candidate(const float *features, const float *probabilities);
void model_predict_batch(const float *features, float *probabilities, size_t rows);
float model_benchmark(const float *features, size_t rows, unsigned repetitions);
const char *model_precision_name(void);
int model_feature_count(void);
