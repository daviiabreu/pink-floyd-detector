#include "fingerprint.h"
#include "catalog_data.h"
#include <string.h>

#define QUERY_HASHES (FINGERPRINT_FRAMES * 4 * 27)
_Static_assert(FINGERPRINT_FRAMES < 255, "Frame counters require fewer than 255 frames");
_Static_assert(CATALOG_BUCKET_LIMIT < 256, "Posting counts must fit one byte");
_Static_assert(CATALOG_SHORTLIST == FINGERPRINT_CANDIDATES, "Evidence capacity must match shortlist");

static uint8_t votes[CATALOG_SCRATCH_BINS], last_frame[CATALOG_SCRATCH_BINS];
static uint32_t hit_start[QUERY_HASHES];
static uint16_t hit_info[QUERY_HASHES];
static uint8_t strength[CATALOG_REFERENCES], selected[CATALOG_REFERENCES];

static void find_hash(uint64_t hash, unsigned frame, unsigned *hits) {
    size_t low = 0, high = CATALOG_ENTRIES;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if ((CATALOG_INDEX[middle] >> 24) < hash) low = middle + 1;
        else high = middle;
    }
    size_t end = low;
    while (end < CATALOG_ENTRIES && (CATALOG_INDEX[end] >> 24) == hash) ++end;
    if (end == low) return;
    hit_start[*hits] = (uint32_t)low;
    hit_info[(*hits)++] = (uint16_t)((frame << 8) | (end - low));
    for (size_t index = low; index < end; ++index) {
        uint32_t entry = (uint32_t)CATALOG_INDEX[index] & 0xffffffu;
        unsigned reference = entry >> 16;
        unsigned offset = (entry & 65535u) + FINGERPRINT_FRAMES - frame;
        for (unsigned delta = 0; delta < 2; ++delta) {
            unsigned bin = CATALOG_COARSE_BASE[reference] + (offset + delta) / CATALOG_COARSE_WIDTH;
            if (last_frame[bin] != frame) {
                last_frame[bin] = (uint8_t)frame;
                ++votes[bin];
                if (votes[bin] > strength[reference]) strength[reference] = votes[bin];
            }
        }
    }
}

unsigned catalog_match_detailed(const uint16_t *peaks, float *scores, catalog_evidence_t *matches) {
    memset(scores, 0, 2 * CATALOG_SONGS * sizeof(*scores));
    memset(votes, 0, sizeof(votes));
    memset(last_frame, 255, sizeof(last_frame));
    memset(strength, 0, sizeof(strength));
    memset(selected, 0, sizeof(selected));
    unsigned hits = 0;
    for (unsigned frame = 0; frame < FINGERPRINT_FRAMES; ++frame) {
        for (int omitted = 0; omitted < FINGERPRINT_PEAKS; ++omitted) {
            uint16_t p[3];
            int count = 0;
            for (int i = 0; i < FINGERPRINT_PEAKS; ++i)
                if (i != omitted) p[count++] = peaks[frame * FINGERPRINT_PEAKS + i];
            for (int i = 0; i < 2; ++i)
                for (int j = i + 1; j < 3; ++j)
                    if (p[i] > p[j]) { uint16_t temp = p[i]; p[i] = p[j]; p[j] = temp; }
            if (p[0] < FINGERPRINT_MIN_BIN || p[2] > FINGERPRINT_MAX_BIN) continue;
            for (int a = -1; a <= 1; ++a)
                for (int b = -1; b <= 1; ++b)
                    for (int c = -1; c <= 1; ++c) {
                        uint64_t hash = ((uint64_t)(p[0] + a) << 22) |
                                        ((uint64_t)(p[1] + b) << 11) | (uint64_t)(p[2] + c);
                        find_hash(hash, frame, &hits);
                    }
        }
    }
    unsigned matched = 0;
    for (unsigned rank = 0; rank < CATALOG_SHORTLIST && rank < CATALOG_REFERENCES; ++rank) {
        unsigned reference = 0;
        for (unsigned i = 1; i < CATALOG_REFERENCES; ++i)
            if (!selected[i] && (selected[reference] || strength[i] > strength[reference])) reference = i;
        if (!strength[reference]) break;
        selected[reference] = 1;
        unsigned bins = CATALOG_REFERENCE_FRAMES[reference] + FINGERPRINT_FRAMES + 1;
        memset(votes, 0, bins);
        memset(last_frame, 255, bins);
        unsigned best = 0;
        for (unsigned hit = 0; hit < hits; ++hit) {
            unsigned frame = hit_info[hit] >> 8, count = hit_info[hit] & 255u;
            for (unsigned i = hit_start[hit]; i < hit_start[hit] + count; ++i) {
                uint32_t entry = (uint32_t)CATALOG_INDEX[i] & 0xffffffu;
                if ((entry >> 16) != reference) continue;
                unsigned offset = (entry & 65535u) + FINGERPRINT_FRAMES - frame;
                for (unsigned delta = 0; delta < 2; ++delta) {
                    unsigned bin = offset + delta;
                    if (last_frame[bin] != frame) {
                        last_frame[bin] = (uint8_t)frame;
                        ++votes[bin];
                        if (votes[bin] > votes[best] || (votes[bin] == votes[best] && bin < best)) best = bin;
                    }
                }
            }
        }
        uint8_t second = 0;
        for (unsigned bin = 0; bin < bins; ++bin)
            if ((bin + WINDOW_FRAMES < best || bin > best + WINDOW_FRAMES) && votes[bin] > second) second = votes[bin];
        unsigned song = CATALOG_CLASS[reference];
        if (matches) matches[matched] = (catalog_evidence_t){
            .offset_frames = (int32_t)best - FINGERPRINT_FRAMES,
            .reference = (uint16_t)reference, .class_index = (uint16_t)(song + 1),
            .votes = votes[best], .second_votes = second,
        };
        ++matched;
        float score = votes[best] / (float)FINGERPRINT_FRAMES;
        float prominence = (votes[best] - second) / (float)FINGERPRINT_FRAMES;
        if (score > scores[2 * song] || (score == scores[2 * song] && prominence > scores[2 * song + 1])) {
            scores[2 * song] = score;
            scores[2 * song + 1] = prominence;
        }
    }
    return matched;
}

void catalog_match(const uint16_t *peaks, float *scores) {
    catalog_match_detailed(peaks, scores, NULL);
}
