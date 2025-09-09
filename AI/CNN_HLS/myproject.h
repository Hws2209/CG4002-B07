// myproject.h
#ifndef MYPROJECT_H
#define MYPROJECT_H

#include <hls_stream.h>

typedef int32_t input_t;    // integer input type
typedef float float_t;    // internal float computation

void cnn_forward(
    hls::stream<input_t> &input_stream,
    hls::stream<float_t> &output_stream
);

#endif
