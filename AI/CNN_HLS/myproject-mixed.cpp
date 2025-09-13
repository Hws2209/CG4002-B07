#include <hls_stream.h>
#include <hls_math.h>
#include "headers/conv1_weight.h"
#include "headers/conv1_bias.h"
#include "headers/conv2_weight.h"
#include "headers/conv2_bias.h"
#include "headers/fc1_weight.h"
#include "headers/fc1_bias.h"
#include "headers/fc2_weight.h"
#include "headers/fc2_bias.h"

#define NUM_CHANNELS 6
#define SEQ_LEN 60 // WINDOW_SIZE
#define CONV1_OUT 6
#define CONV2_OUT 3
#define KERNEL_SIZE 3
#define POOL_SIZE 2
#define FC1_NEURONS 64
#define NUM_CLASSES 4

typedef int32_t input_t; // integer input type
typedef float float_t; // internal float computation

// ---------------- ReLU ----------------
float_t relu(float_t x) {
    return (x > 0.0f) ? x : 0.0f;
}

// ---------------- Conv1D ----------------
void conv1d_layer1(
    const input_t input[][SEQ_LEN],
    float_t output[][SEQ_LEN],
    const float_t weight[],
    const float_t bias[],
    int in_channels,
    int out_channels
) {
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    
    Conv1_Loop_OC: for (int oc = 0; oc < out_channels; oc++) {
        Conv1_Loop_I: for (int i = 0; i < SEQ_LEN; i++) {
            #pragma HLS PIPELINE II=1
            float_t sum = bias[oc];
            Conv1_Loop_IC: for (int ic = 0; ic < in_channels; ic++) {
                Conv1_Loop_K: for (int k = 0; k < KERNEL_SIZE; k++) {
                    int idx = i + k - 1; // padding='same'
                    if (idx >= 0 && idx < SEQ_LEN) {
                        sum += float_t(input[ic][idx]) * weight[oc*in_channels*KERNEL_SIZE + ic*KERNEL_SIZE + k];
                    }
                }
            }
            output[oc][i] = relu(sum);
        }
    }
}


void conv1d_layer2(
    const float_t input[][SEQ_LEN],
    float_t output[][SEQ_LEN],
    const float_t weight[],
    const float_t bias[],
    int in_channels,
    int out_channels
) {
    Conv2_Loop_OC: for (int oc = 0; oc < out_channels; oc++) {
        Conv2_Loop_I: for (int i = 0; i < SEQ_LEN; i++) {
            #pragma HLS PIPELINE II=1
            float_t sum = bias[oc];
            Conv2_Loop_IC: for (int ic = 0; ic < in_channels; ic++) {
                Conv2_Loop_K: for (int k = 0; k < KERNEL_SIZE; k++) {
                    int idx = i + k - 1; // padding='same'
                    if (idx >= 0 && idx < SEQ_LEN) {
                        sum += input[ic][idx] * weight[oc*in_channels*KERNEL_SIZE + ic*KERNEL_SIZE + k];
                    }
                }
            }
            output[oc][i] = relu(sum);
        }
    }
}

// ---------------- MaxPool1D ----------------
void maxpool1d(
    const float_t input[][SEQ_LEN],
    float_t output[][SEQ_LEN/POOL_SIZE],
    int channels
) {
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1

    MaxPool_Loop_C: for (int c = 0; c < channels; c++) {
        MaxPool_Loop_I: for (int i = 0; i < SEQ_LEN; i += POOL_SIZE) {
            #pragma HLS PIPELINE II=1
            float_t max_val = input[c][i];
            MaxPool_Loop_P: for (int j = 1; j < POOL_SIZE; j++) {
                if (input[c][i+j] > max_val) max_val = input[c][i+j];
            }
            output[c][i/POOL_SIZE] = max_val;
        }
    }
}

// ---------------- Fully Connected ----------------
void fc(
    hls::stream<float_t> &in_stream,
    hls::stream<float_t> &out_stream,
    const float_t weight[],
    const float_t bias[],
    int in_size,
    int out_size,
    bool should_relu
) {
    float_t sum[FC1_NEURONS];
    #pragma HLS ARRAY_PARTITION variable=sum complete dim=1

    // Initialize partial sums with bias
    for (int o = 0; o < out_size; o++) {
        #pragma HLS PIPELINE II=1
        sum[o] = bias[o];
    }

    // Streaming accumulation
    FC_Loop_I: for (int i = 0; i < in_size; i++) {
        float_t val = in_stream.read();
        FC_Loop_O: for (int o = 0; o < out_size; o++) {
            #pragma HLS PIPELINE II=1
            sum[o] += val * weight[o*in_size + i];
        }
    }

    // Write results
    for (int o = 0; o < out_size; o++) {
        #pragma HLS PIPELINE II=1
        if (should_relu) {
            out_stream.write(relu(sum[o]));
        } else {
            out_stream.write(sum[o]);
        }
    }
}

// ---------------- CNN Forward ----------------
void cnn_forward(
    hls::stream<input_t> &input_stream,
    hls::stream<float_t> &output_stream
) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

    static input_t input[NUM_CHANNELS][SEQ_LEN];
    static float_t output[NUM_CLASSES];
    static float_t conv1_out[CONV1_OUT][SEQ_LEN];
    static float_t conv2_out[CONV2_OUT][SEQ_LEN];
    static float_t pool_out[CONV2_OUT][SEQ_LEN/POOL_SIZE];
    hls::stream<float_t> pool_stream("pool_stream"); // [CONV2_OUT][SEQ_LEN/POOL_SIZE]
    hls::stream<float_t> fc1_stream("fc1_stream"); // [FC1_NEURONS]

    Read_Input: for(int c=0; c<NUM_CHANNELS; c++) {
        for(int i=0; i<SEQ_LEN; i++) {
            #pragma HLS PIPELINE II=1
            input[c][i] = input_stream.read();
        }
    }

    conv1d_layer1(input, conv1_out, conv1_weight, conv1_bias, NUM_CHANNELS, CONV1_OUT);
    conv1d_layer2(conv1_out, conv2_out, conv2_weight, conv2_bias, CONV1_OUT, CONV2_OUT);
    maxpool1d(conv2_out, pool_out, CONV2_OUT);

    Flatten_Loop_C: for (int c = 0; c < CONV2_OUT; c++) {
        Flatten_Loop_I: for (int i = 0; i < SEQ_LEN/POOL_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            pool_stream.write(pool_out[c][i]);
        }
    }
    
    fc(pool_stream, fc1_stream, fc1_weight, fc1_bias, CONV2_OUT*(SEQ_LEN/POOL_SIZE), FC1_NEURONS, true);
    fc(fc1_stream, output_stream, fc2_weight, fc2_bias, FC1_NEURONS, NUM_CLASSES, false);
}
