// myproject.h
#ifndef MYPROJECT_H
#define MYPROJECT_H

#define NUM_CHANNELS 6
#define SEQ_LEN 60       // your WINDOW_SIZE
#define NUM_CLASSES 4

typedef int16_t input_t;    // integer input type
typedef float float_t;    // internal float computation

void cnn_forward(const input_t input[NUM_CHANNELS][SEQ_LEN], float_t output[NUM_CLASSES]);

#endif
