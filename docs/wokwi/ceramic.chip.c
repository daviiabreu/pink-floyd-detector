#include "wokwi-api.h"

void chip_init(void) {
    pin_init("1", INPUT);
    pin_init("2", INPUT);
}
