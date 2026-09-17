#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "driver/uart.h"
#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/ringbuf.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "decision.h"
#include "dsp.h"
#include "inference.h"
#include "fingerprint.h"
#include "panel.h"

#define MIC_BCLK GPIO_NUM_26
#define MIC_WS GPIO_NUM_25
#define MIC_DATA GPIO_NUM_33
#define ALERT_LED GPIO_NUM_27
#define STATUS_LED GPIO_NUM_14
#ifndef FEATURE_DELAY_MS
#define FEATURE_DELAY_MS 0
#endif

static void leds_init(void) {
    gpio_config_t leds = {
        .pin_bit_mask = (1ULL << ALERT_LED) | (1ULL << STATUS_LED),
        .mode = GPIO_MODE_OUTPUT,
    };
    ESP_ERROR_CHECK(gpio_config(&leds));
    gpio_set_level(ALERT_LED, 0);
    gpio_set_level(STATUS_LED, 0);
}

#ifdef WIRING_TEST
static void wiring_task(void *argument) {
    (void)argument;
    printf("{\"type\":\"wiring_test\",\"note\":\"LEDs alternam a cada 500 ms; sem captura de audio\"}\n");
    for (;;) {
        bool green = (esp_timer_get_time() / 500000) % 2;
        gpio_set_level(ALERT_LED, green);
        gpio_set_level(STATUS_LED, !green);
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}
#else
static i2s_chan_handle_t microphone;
static portMUX_TYPE dma_lock = portMUX_INITIALIZER_UNLOCKED;
static uint32_t dma_overflows;

static bool IRAM_ATTR on_dma_overflow(i2s_chan_handle_t handle,
                                     i2s_event_data_t *event, void *context) {
    (void)handle; (void)event; (void)context;
    portENTER_CRITICAL_ISR(&dma_lock);
    ++dma_overflows;
    portEXIT_CRITICAL_ISR(&dma_lock);
    return false;
}

static uint32_t overflow_count(void) {
    portENTER_CRITICAL(&dma_lock);
    uint32_t result = dma_overflows;
    portEXIT_CRITICAL(&dma_lock);
    return result;
}

static void microphone_init(void) {
    i2s_chan_config_t channel = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    channel.dma_desc_num = 8;
    channel.dma_frame_num = 256;
    ESP_ERROR_CHECK(i2s_new_channel(&channel, NULL, &microphone));
    i2s_std_config_t config = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {.mclk = I2S_GPIO_UNUSED, .bclk = MIC_BCLK, .ws = MIC_WS,
                     .dout = I2S_GPIO_UNUSED, .din = MIC_DATA},
    };
    config.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(microphone, &config));
    i2s_event_callbacks_t callbacks = {.on_recv_q_ovf = on_dma_overflow};
    ESP_ERROR_CHECK(i2s_channel_register_event_callback(microphone, &callbacks, NULL));
}

static bool read_frame(int16_t *pcm) {
    int32_t raw[FRAME_SIZE];
    size_t total = 0;
    int64_t deadline = esp_timer_get_time() + 200000;
    while (total < sizeof(raw)) {
        size_t count = 0;
        if (esp_timer_get_time() >= deadline) return false;
        esp_err_t result = i2s_channel_read(microphone, (uint8_t *)raw + total,
                                           sizeof(raw) - total, &count, 50);
        total += count;
        if (result != ESP_OK && result != ESP_ERR_TIMEOUT) return false;
    }
    for (int i = 0; i < FRAME_SIZE; ++i) pcm[i] = (int16_t)(raw[i] >> 16);
    return true;
}

#ifdef RECORD_AUDIO
static void record_task(void *argument) {
    (void)argument;
    struct { char magic[4]; uint32_t sequence; int16_t pcm[FRAME_SIZE]; } packet = {
        .magic = {'A', 'U', 'D', '0'},
    };
    ESP_ERROR_CHECK(uart_set_baudrate(UART_NUM_0, 921600));
    ESP_ERROR_CHECK(i2s_channel_enable(microphone));
    uint32_t previous_overflows = overflow_count();
    for (;;) {
        ++packet.sequence;
        if (!read_frame(packet.pcm)) continue;
        uint32_t current = overflow_count();
        if (current != previous_overflows) {
            ++packet.sequence;
            previous_overflows = current;
            continue;
        }
        uart_write_bytes(UART_NUM_0, &packet, sizeof(packet));
    }
}
#else
#ifndef MODEL_CATALOG_ID
#error "The detector requires a model trained with the fingerprint catalog"
#endif
typedef struct {
    uint32_t sequence, epoch;
    int64_t started_us, ready_us;
    int16_t pcm[FRAME_SIZE];
} audio_frame_t;

typedef struct {
    uint32_t sequence;
    int64_t started_us, captured_us, queued_us;
    int64_t capture_max_us, feature_sum_us, feature_max_us, ring_wait_max_us;
    float values[FEATURE_COUNT];
    uint16_t peaks[FINGERPRINT_FRAMES][FINGERPRINT_PEAKS];
} feature_packet_t;

typedef struct {
    feature_packet_t features;
    int64_t queue_wait_us, fingerprint_us, inference_us, e2e_us, confirmation_us;
    int candidate, alert;
    unsigned support_windows, aligned_frames;
    float confidence;
} result_packet_t;

typedef struct {
    uint32_t frames, read_errors, ring_drops, feature_drops, log_drops, windows;
    uint32_t dsp_deadlines, stale_windows;
} metrics_t;

static RingbufHandle_t audio_ring;
static QueueHandle_t feature_queue, result_queue;
static SemaphoreHandle_t metrics_mutex;
static metrics_t metrics;
static TaskHandle_t capture_handle, feature_handle, detect_handle;

static void increment(uint32_t *counter) {
    configASSERT(xSemaphoreTake(metrics_mutex, portMAX_DELAY) == pdTRUE);
    ++*counter;
    xSemaphoreGive(metrics_mutex);
}

static void capture_task(void *argument) {
    (void)argument;
    audio_frame_t frame = {0};
    ESP_ERROR_CHECK(i2s_channel_enable(microphone));
    uint32_t previous_epoch = overflow_count();
    for (;;) {
        ++frame.sequence;
        uint32_t epoch = overflow_count();
        frame.started_us = esp_timer_get_time();
        if (!read_frame(frame.pcm)) {
            increment(&metrics.read_errors);
            continue;
        }
        frame.ready_us = esp_timer_get_time();
        frame.epoch = overflow_count();
        increment(&metrics.frames);
        if (frame.epoch != epoch || frame.epoch != previous_epoch) {
            previous_epoch = frame.epoch;
            ++frame.sequence;
            continue;
        }
        if (xRingbufferSend(audio_ring, &frame, sizeof(frame), 0) != pdTRUE)
            increment(&metrics.ring_drops);
    }
}

static void feature_task(void *argument) {
    (void)argument;
    static dsp_context_t dsp;
    fingerprint_context_t *fingerprint = heap_caps_malloc(
        sizeof(*fingerprint), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    configASSERT(fingerprint);
    fingerprint_init(fingerprint);
    feature_accumulator_t accumulator;
    feature_packet_t packet = {0};
    uint32_t last_sequence = 0, last_epoch = 0;
    dsp_init(&dsp);
    features_reset(&accumulator);
    for (;;) {
        size_t size;
        audio_frame_t *frame = xRingbufferReceive(audio_ring, &size, portMAX_DELAY);
        configASSERT(frame && size == sizeof(*frame));
        if (frame->sequence - last_sequence != 1 || frame->epoch != last_epoch)
            features_reset(&accumulator);
        last_sequence = frame->sequence;
        last_epoch = frame->epoch;
        if (accumulator.count == 0) {
            packet = (feature_packet_t){.started_us = frame->started_us};
            fingerprint_reset(fingerprint);
        }
        int64_t start = esp_timer_get_time();
        int64_t wait = start - frame->ready_us;
        if (wait > packet.ring_wait_max_us) packet.ring_wait_max_us = wait;
        int64_t capture = frame->ready_us - frame->started_us;
        if (capture > packet.capture_max_us) packet.capture_max_us = capture;
        float values[FRAME_FEATURES];
        dsp_frame(&dsp, frame->pcm, values);
        unsigned warmup = FINGERPRINT_FFT / FRAME_SIZE - 1;
        unsigned peak_row = accumulator.count < warmup ? 0 : accumulator.count - warmup;
        fingerprint_push(fingerprint, frame->pcm, packet.peaks[peak_row]);
        packet.sequence = frame->sequence;
        packet.captured_us = frame->ready_us;
        vRingbufferReturnItem(audio_ring, frame);
        if (FEATURE_DELAY_MS > 0) vTaskDelay(pdMS_TO_TICKS(FEATURE_DELAY_MS));
        int ready = features_push(&accumulator, values, packet.values);
        int64_t elapsed = esp_timer_get_time() - start;
        packet.feature_sum_us += elapsed;
        if (elapsed > packet.feature_max_us) packet.feature_max_us = elapsed;
        if (elapsed > 32000) increment(&metrics.dsp_deadlines);
        if (ready) {
            packet.queued_us = esp_timer_get_time();
            if (xQueueSend(feature_queue, &packet, 0) != pdTRUE)
                increment(&metrics.feature_drops);
        }
    }
}

static bool update_status(panel_t *panel, int64_t now) {
    bool timeout = panel_audio_timeout(panel, now);
    gpio_set_level(STATUS_LED, panel_red_led(panel, now));
    if (timeout) gpio_set_level(ALERT_LED, 0);
    return timeout;
}

static void detect_task(void *argument) {
    (void)argument;
    decision_state_t state;
    decision_reset(&state);
    int previous_alert = -1;
    panel_t panel;
    panel_init(&panel, esp_timer_get_time());
    for (;;) {
        if (update_status(&panel, esp_timer_get_time())) {
            decision_reset(&state);
            previous_alert = -1;
        }
        result_packet_t result = {0};
        if (xQueueReceive(feature_queue, &result.features, pdMS_TO_TICKS(50)) != pdTRUE)
            continue;
        int64_t start = esp_timer_get_time();
        result.queue_wait_us = start - result.features.queued_us;
        float probabilities[MODEL_CLASSES];
        float input[MODEL_FEATURES];
        catalog_evidence_t matches[FINGERPRINT_CANDIDATES];
        memcpy(input, result.features.values, sizeof(result.features.values));
        unsigned match_count = catalog_match_detailed(
            &result.features.peaks[0][0], input + FEATURE_COUNT, matches);
        int64_t model_start = esp_timer_get_time();
        result.fingerprint_us = model_start - start;
        model_predict(input, probabilities);
        result.candidate = model_candidate(input, probabilities);
        result.inference_us = esp_timer_get_time() - model_start;
        int best = 0;
        for (int c = 1; c < MODEL_CLASSES; ++c)
            if (probabilities[c] > probabilities[best]) best = c;
        result.confidence = probabilities[best];
        int64_t now = esp_timer_get_time();
        if (update_status(&panel, now)) {
            decision_reset(&state);
            previous_alert = -1;
        }
        if (now >= result.features.started_us && now - result.features.started_us <= MAX_WINDOW_AGE_US)
            panel_note_window(&panel, now);
        result.alert = decision_update(&state, result.candidate, matches, match_count, result.features.sequence,
                                       result.features.started_us, now);
        result.support_windows = state.support_windows;
        result.aligned_frames = state.aligned_frames;
        gpio_set_level(ALERT_LED, result.alert >= 0);
        gpio_set_level(STATUS_LED, panel_red_led(&panel, now));
        result.e2e_us = esp_timer_get_time() - result.features.started_us;
        result.confirmation_us = result.alert >= 0 && result.alert != previous_alert ? now - state.first_start_us : 0;
        previous_alert = result.alert;
        increment(&metrics.windows);
        if (result.e2e_us > MAX_WINDOW_AGE_US) increment(&metrics.stale_windows);
        if (xQueueSend(result_queue, &result, 0) != pdTRUE) increment(&metrics.log_drops);
    }
}

static const char *label(int index) { return index < 0 ? "desconhecida" : MODEL_LABELS[index]; }

static void telemetry_task(void *argument) {
    (void)argument;
    printf("{\"type\":\"boot\",\"model\":\"%s\",\"dataset\":\"%s\",\"precision\":\"%s\"}\n",
           MODEL_ID, MODEL_DATASET, model_precision_name());
    int64_t last_stats = 0;
    for (;;) {
        result_packet_t result;
        if (xQueueReceive(result_queue, &result, pdMS_TO_TICKS(500)) == pdTRUE) {
            feature_packet_t *f = &result.features;
            printf("{\"type\":\"result\",\"precision\":\"%s\",\"sequence\":%" PRIu32 ",\"candidate\":\"%s\","
                   "\"alert\":\"%s\",\"confidence\":%.5f,\"log_rms\":%.5f,"
                   "\"capture_max_us\":%" PRId64 ",\"capture_window_us\":%" PRId64 ","
                   "\"feature_sum_us\":%" PRId64 ",\"feature_max_us\":%" PRId64 ","
                   "\"ring_wait_max_us\":%" PRId64 ",\"queue_wait_us\":%" PRId64 ","
                   "\"fingerprint_us\":%" PRId64 ",\"inference_us\":%" PRId64 ",\"software_e2e_us\":%" PRId64 ","
                   "\"confirmation_us\":%" PRId64 ",\"support_windows\":%u,\"aligned_frames\":%u}\n",
                   model_precision_name(), f->sequence, label(result.candidate), label(result.alert), (double)result.confidence,
                   (double)f->values[0], f->capture_max_us, f->captured_us - f->started_us,
                   f->feature_sum_us, f->feature_max_us, f->ring_wait_max_us,
                   result.queue_wait_us, result.fingerprint_us, result.inference_us, result.e2e_us, result.confirmation_us,
                   result.support_windows, result.aligned_frames);
        }
        int64_t now = esp_timer_get_time();
        if (now - last_stats >= 1000000) {
            last_stats = now;
            configASSERT(xSemaphoreTake(metrics_mutex, portMAX_DELAY) == pdTRUE);
            metrics_t snapshot = metrics;
            xSemaphoreGive(metrics_mutex);
            printf("{\"type\":\"stats\",\"uptime_us\":%" PRId64 ",\"frames\":%" PRIu32 ","
                   "\"read_errors\":%" PRIu32 ",\"ring_drops\":%" PRIu32 ",\"feature_drops\":%" PRIu32 ","
                   "\"log_drops\":%" PRIu32 ",\"dma_overflows\":%" PRIu32 ",\"windows\":%" PRIu32 ","
                   "\"dsp_deadlines\":%" PRIu32 ",\"stale_windows\":%" PRIu32 ",\"free_heap\":%u,"
                   "\"stack_capture\":%u,\"stack_features\":%u,\"stack_detect\":%u}\n",
                   now, snapshot.frames, snapshot.read_errors, snapshot.ring_drops,
                   snapshot.feature_drops, snapshot.log_drops, overflow_count(), snapshot.windows,
                   snapshot.dsp_deadlines, snapshot.stale_windows,
                   (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
                   (unsigned)uxTaskGetStackHighWaterMark(capture_handle),
                   (unsigned)uxTaskGetStackHighWaterMark(feature_handle),
                   (unsigned)uxTaskGetStackHighWaterMark(detect_handle));
        }
    }
}
#endif
#endif

void app_main(void) {
    leds_init();
#ifdef WIRING_TEST
    configASSERT(xTaskCreatePinnedToCore(wiring_task, "wiring", 4096, NULL, 5, NULL, 1) == pdPASS);
#else
    microphone_init();
#ifdef RECORD_AUDIO
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 256, 2048, 0, NULL, 0));
    configASSERT(xTaskCreatePinnedToCore(record_task, "record", 6144, NULL, 20, NULL, 0) == pdPASS);
#else
    audio_ring = xRingbufferCreate(12 * (sizeof(audio_frame_t) + 8), RINGBUF_TYPE_NOSPLIT);
    feature_queue = xQueueCreate(3, sizeof(feature_packet_t));
    result_queue = xQueueCreate(8, sizeof(result_packet_t));
    metrics_mutex = xSemaphoreCreateMutex();
    configASSERT(audio_ring && feature_queue && result_queue && metrics_mutex);
    configASSERT(xTaskCreatePinnedToCore(feature_task, "features", 6144, NULL, 10, &feature_handle, 1) == pdPASS);
    configASSERT(xTaskCreatePinnedToCore(detect_task, "detect", 12288, NULL, 5, &detect_handle, 1) == pdPASS);
    configASSERT(xTaskCreatePinnedToCore(capture_task, "capture", 6144, NULL, 20, &capture_handle, 0) == pdPASS);
    configASSERT(xTaskCreatePinnedToCore(telemetry_task, "telemetry", 4096, NULL, 2, NULL, 0) == pdPASS);
#endif
#endif
}
