#include "panel.h"
#include "decision.h"

void panel_init(panel_t *panel, int64_t now_us) {
    *panel = (panel_t){.awaiting_window = true, .last_window_us = now_us};
}

void panel_note_window(panel_t *panel, int64_t now_us) {
    panel->last_window_us = now_us;
    panel->awaiting_window = false;
}

bool panel_audio_timeout(const panel_t *panel, int64_t now_us) {
    int64_t timeout = ALERT_TIMEOUT_US + (panel->awaiting_window ? WINDOW_DURATION_US : 0);
    return now_us - panel->last_window_us >= timeout;
}

bool panel_red_led(const panel_t *panel, int64_t now_us) {
    return panel_audio_timeout(panel, now_us) && (now_us / 250000) % 2;
}
