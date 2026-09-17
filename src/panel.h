#pragma once
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    bool awaiting_window;
    int64_t last_window_us;
} panel_t;

void panel_init(panel_t *panel, int64_t now_us);
void panel_note_window(panel_t *panel, int64_t now_us);
bool panel_audio_timeout(const panel_t *panel, int64_t now_us);
bool panel_red_led(const panel_t *panel, int64_t now_us);
