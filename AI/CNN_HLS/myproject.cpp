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
#define SEQ_LEN 60       // your WINDOW_SIZE
#define CONV1_OUT 6
#define CONV2_OUT 3
#define KERNEL_SIZE 3
#define POOL_SIZE 2
#define FC1_NEURONS 64
#define NUM_CLASSES 4

typedef int32_t input_t;    // integer input type
typedef float float_t;    // internal float computation

// ---------------- ReLU ----------------
float_t relu(float_t x) {
    return (x > 0.0f) ? x : 0.0f;
}

// ---------------- Conv1D ----------------
void conv1d_layer1(
    const input_t input[][SEQ_LEN],
    const float_t weight[],
    const float_t bias[],
    float_t output[][SEQ_LEN],
    int in_channels,
    int out_channels
) {
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    
    Conv1_Loop_OC: for(int oc=0; oc<out_channels; oc++) {
        Conv1_Loop_I: for(int i=0; i<SEQ_LEN; i++) {
            #pragma HLS PIPELINE II=1
            float_t sum = bias[oc];
            Conv1_Loop_IC: for(int ic=0; ic<in_channels; ic++) {
                Conv1_Loop_K: for(int k=0; k<KERNEL_SIZE; k++) {
                    int idx = i + k - 1; // padding='same'
                    if(idx >=0 && idx < SEQ_LEN) sum += float_t(input[ic][idx]) * weight[oc*in_channels*KERNEL_SIZE + ic*KERNEL_SIZE + k];
                }
            }
            output[oc][i] = relu(sum);
        }
    }
}


void conv1d_layer2(
    const float_t input[][SEQ_LEN/POOL_SIZE],
    const float_t weight[],
    const float_t bias[],
    float_t output[][SEQ_LEN/POOL_SIZE],
    int in_channels,
    int out_channels
) {
    Conv2_Loop_OC: for(int oc=0; oc<out_channels; oc++) {
        Conv2_Loop_I: for(int i=0; i<SEQ_LEN/POOL_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            float_t sum = bias[oc];
            Conv2_Loop_IC: for(int ic=0; ic<in_channels; ic++) {
                Conv2_Loop_K: for(int k=0; k<KERNEL_SIZE; k++) {
                    int idx = i + k - 1; // padding='same'
                    if(idx >=0 && idx < SEQ_LEN/POOL_SIZE) sum += input[ic][idx] * weight[oc*in_channels*KERNEL_SIZE + ic*KERNEL_SIZE + k];
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

    MaxPool_Loop_C: for(int c=0;c<channels;c++)
        MaxPool_Loop_I: for(int i=0;i<SEQ_LEN;i+=POOL_SIZE){
            #pragma HLS PIPELINE II=1
            float_t max_val = input[c][i];
            MaxPool_Loop_P: for(int j=1;j<POOL_SIZE;j++)
                if(input[c][i+j] > max_val) max_val = input[c][i+j];
            output[c][i/POOL_SIZE] = max_val;
        }
}

// ---------------- Fully Connected ----------------
void fc(
    const float_t input[],
    const float_t weight[],
    const float_t bias[],
    float_t output[],
    int in_size,
    int out_size,
    bool should_relu
) {
    FC_Loop_O: for (int o=0;o<out_size;o++) {
        #pragma HLS PIPELINE II=1
        float_t sum = bias[o];
        FC_Loop_I: for (int i=0;i<in_size;i++) {
            sum += input[i] * weight[o*in_size + i];
        }
        
        if (should_relu) {
            output[o] = relu(sum);
        } else {
            output[o] = sum;
        }
    }
}

// ---------------- Flatten ----------------
void flatten(
    const float_t input[][SEQ_LEN/POOL_SIZE],
    float_t output[],
    int channels
) {
    Flatten_Loop_C: for(int c=0;c<channels;c++)
        Flatten_Loop_I: for(int i=0;i<SEQ_LEN/POOL_SIZE;i++)
            #pragma HLS PIPELINE II=1
            output[c*(SEQ_LEN/POOL_SIZE)+i] = input[c][i];
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
    static float_t pool1_out[CONV1_OUT][SEQ_LEN/POOL_SIZE];
    static float_t conv2_out[CONV2_OUT][SEQ_LEN/POOL_SIZE];
    static float_t flatten_vec[CONV2_OUT * (SEQ_LEN/POOL_SIZE)];
    static float_t fc1_out[FC1_NEURONS];

    Read_Input: for(int c=0; c<NUM_CHANNELS; c++) {
        for(int i=0; i<SEQ_LEN; i++) {
            #pragma HLS PIPELINE II=1
            input[c][i] = input_stream.read();
        }
    }

    // Conv1
    conv1d_layer1(input, conv1_weight, conv1_bias, conv1_out, NUM_CHANNELS, CONV1_OUT);

    // MaxPool
    maxpool1d(conv1_out, pool1_out, CONV1_OUT);

    // Conv2
    conv1d_layer2(pool1_out, conv2_weight, conv2_bias, conv2_out, CONV1_OUT, CONV2_OUT);

    // Flatten
    flatten(conv2_out, flatten_vec, CONV2_OUT);

    // FC1
    fc(flatten_vec, fc1_weight, fc1_bias, fc1_out, CONV2_OUT*(SEQ_LEN/POOL_SIZE), FC1_NEURONS, true);

    // FC2
    fc(fc1_out, fc2_weight, fc2_bias, output, FC1_NEURONS, NUM_CLASSES, false);

    Write_Output: for(int i=0; i<NUM_CLASSES; i++) {
        #pragma HLS PIPELINE II=1
        output_stream.write(output[i]);
    }
}
