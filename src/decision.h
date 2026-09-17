#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "fingerprint.h"

#define CONFIRM_WINDOWS 3
#define EVIDENCE_WINDOWS 8
#define MIN_MATCHED_FRAMES 3
#define MIN_PEAK_MARGIN 2
#define MIN_TOTAL_MATCHED_FRAMES 18
#define ALIGNMENT_TOLERANCE_FRAMES 2
#define WINDOW_DURATION_US ((int64_t)WINDOW_SAMPLES * 1000000 / SAMPLE_RATE)
#define MAX_WINDOW_AGE_US (WINDOW_DURATION_US + 1000000)
#define ALERT_TIMEOUT_US (WINDOW_DURATION_US + 1500000)

typedef struct {
    int model_candidate;
    uint32_t sequence;
    int64_t started_us;
    unsigned count;
    catalog_evidence_t matches[FINGERPRINT_CANDIDATES];
} decision_window_t;

typedef struct {
    decision_window_t history[EVIDENCE_WINDOWS];
    unsigned next, filled, support_windows, aligned_frames;
    int64_t first_start_us;
} decision_state_t;

void decision_reset(decision_state_t *state);
int decision_update(decision_state_t *state, int candidate,
                    const catalog_evidence_t *matches, unsigned count, uint32_t sequence,
                    int64_t window_start_us, int64_t now_us);
