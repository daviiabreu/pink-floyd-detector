#include "wokwi-api.h"

void chip_init(void) {
    pin_init("VDD", INPUT);
    pin_init("GND", INPUT);
    pin_init("L/R", INPUT);
    pin_init("SCK", INPUT);
    pin_init("WS", INPUT);
    pin_init("SD", INPUT);
}
