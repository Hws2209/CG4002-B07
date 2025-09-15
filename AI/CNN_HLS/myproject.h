#ifndef MYPROJECT_H
#define MYPROJECT_H

#include <hls_stream.h>

typedef int32_t input_t;
typedef float float_t;

void cnn_forward(
    hls::stream<input_t> &input_stream,
    hls::stream<float_t> &output_stream
);

#endif
