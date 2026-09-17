#include "decision.h"
#include <string.h>

static bool usable(const catalog_evidence_t *match) {
    return match->votes >= MIN_MATCHED_FRAMES &&
           match->votes >= match->second_votes + MIN_PEAK_MARGIN;
}

void decision_reset(decision_state_t *state) {
    memset(state, 0, sizeof(*state));
}

int decision_update(decision_state_t *state, int candidate,
                    const catalog_evidence_t *matches, unsigned count, uint32_t sequence,
                    int64_t window_start_us, int64_t now_us) {
    if (count > FINGERPRINT_CANDIDATES || (count && !matches) ||
        now_us < window_start_us || now_us - window_start_us > MAX_WINDOW_AGE_US) {
        decision_reset(state);
        return -1;
    }
    if (state->filled && sequence - state->history[(state->next + EVIDENCE_WINDOWS - 1) % EVIDENCE_WINDOWS].sequence != WINDOW_FRAMES)
        decision_reset(state);
    decision_window_t *current = &state->history[state->next];
    *current = (decision_window_t){
        .model_candidate = candidate, .sequence = sequence, .started_us = window_start_us, .count = count,
    };
    if (count) memcpy(current->matches, matches, count * sizeof(*matches));
    state->next = (state->next + 1) % EVIDENCE_WINDOWS;
    if (state->filled < EVIDENCE_WINDOWS) ++state->filled;
    state->support_windows = state->aligned_frames = 0;
    state->first_start_us = 0;
    int accepted = -1;
    for (unsigned i = 0; i < count; ++i) {
        const catalog_evidence_t *match = &current->matches[i];
        if (!usable(match)) continue;
        unsigned support = 0, votes = 0;
        bool model_agrees = false;
        int64_t first = window_start_us;
        for (unsigned h = 0; h < state->filled; ++h) {
            const decision_window_t *past = &state->history[h];
            uint32_t elapsed_frames = sequence - past->sequence;
            if (elapsed_frames > (EVIDENCE_WINDOWS - 1) * WINDOW_FRAMES) continue;
            for (unsigned j = 0; j < past->count; ++j) {
                const catalog_evidence_t *other = &past->matches[j];
                int64_t drift = (int64_t)match->offset_frames - other->offset_frames - elapsed_frames;
                if (other->reference != match->reference || other->class_index != match->class_index ||
                    !usable(other) || drift < -ALIGNMENT_TOLERANCE_FRAMES || drift > ALIGNMENT_TOLERANCE_FRAMES)
                    continue;
                ++support;
                votes += other->votes;
                model_agrees |= past->model_candidate == match->class_index;
                if (past->started_us < first) first = past->started_us;
                break;
            }
        }
        if (support < CONFIRM_WINDOWS || votes < MIN_TOTAL_MATCHED_FRAMES || !model_agrees) continue;
        if (accepted >= 0 && accepted != match->class_index) {
            state->support_windows = state->aligned_frames = 0;
            state->first_start_us = 0;
            return -1;
        }
        accepted = match->class_index;
        if (votes > state->aligned_frames) {
            state->support_windows = support;
            state->aligned_frames = votes;
            state->first_start_us = first;
        }
    }
    return accepted;
}
