idf_component_register(SRCS "main.c" "dsp.c" "fingerprint.c" "catalog.c" "inference.c" "decision.c" "panel.c"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_i2s esp_driver_gpio esp_driver_uart esp_timer esp_ringbuf)
