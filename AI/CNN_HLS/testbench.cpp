#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm> // for std::max_element
#include <cmath>     // for std::fabs
#include "myproject.h" // HLS top function
#include <unistd.h>
#include <stdio.h>
#include <hls_stream.h>

#define NUM_CHANNELS 6
#define SEQ_LEN 60
#define NUM_CLASSES 4

typedef int32_t input_t;
typedef float float_t;

// Utility: get argmax of logits
int argmax(const float_t logits[NUM_CLASSES]) {
    int max_idx = 0;
    float_t max_val = logits[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int main() {
    input_t input[NUM_CHANNELS][SEQ_LEN];
    float_t output[NUM_CLASSES];

    std::ifstream data_file("../../../../../data.txt");
    if (!data_file.is_open()) {
        std::cerr << "Failed to open data.txt\n";
        return 1;
    }

    std::ifstream golden_file("../../../../../golden_logits.txt");
    if (!golden_file.is_open()) {
        std::cerr << "Failed to open golden_logits.txt\n";
        return 1;
    }

    std::ofstream out_file("../../../../../output_logits.txt");
    if (!out_file.is_open()) {
        std::cerr << "Failed to open output_logits.txt\n";
        return 1;
    }

    std::string line;
    std::vector<std::vector<int>> current_matrix;
    std::vector<int> golden_pred_classes;
    int sample_count = 0;
    int num_failures = 0;
    int num_logit_mismatches = 0;

    auto process_sample = [&](const std::vector<std::vector<int>> &matrix) {
        // Transpose to [NUM_CHANNELS][SEQ_LEN]
        for (int t = 0; t < SEQ_LEN; t++)
            for (int ch = 0; ch < NUM_CHANNELS; ch++)
                input[ch][t] = static_cast<input_t>(matrix[t][ch]);

        // Call HLS CNN
        hls::stream<input_t> input_stream;
        hls::stream<float_t> output_stream;

        for (int ch = 0; ch < NUM_CHANNELS; ch++)
            for (int t = 0; t < SEQ_LEN; t++)
                input_stream.write(input[ch][t]);

        cnn_forward(input_stream, output_stream);

        // Read output from stream
        for (int c = 0; c < NUM_CLASSES; c++)
            output[c] = output_stream.read();

        // Write logits to file
        for (int c = 0; c < NUM_CLASSES; c++) {
            out_file << output[c];
            if (c < NUM_CLASSES - 1) out_file << ", ";
        }
        out_file << "\n";

        // Read corresponding golden logits line
        if (!std::getline(golden_file, line)) {
            std::cerr << "Golden logits file has fewer samples than data.txt\n";
            return;
        }

        std::istringstream iss(line);
        float_t golden_logits[NUM_CLASSES];
        for (int c = 0; c < NUM_CLASSES; c++) {
            iss >> golden_logits[c];
            if (c < NUM_CLASSES - 1) iss.ignore(1, ','); // skip comma
        }

        // Compare predicted class
        int pred_class = argmax(output);
        int golden_class = argmax(golden_logits);
        if (pred_class != golden_class) num_failures++;

        // Logit closeness check (±0.1)
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (std::fabs(output[c] - golden_logits[c]) > 0.1f) {
                num_logit_mismatches++;
            }
        }
        sample_count++;
    };

    while (std::getline(data_file, line)) {
        if (line.empty()) {
            // End of current matrix
            if (!current_matrix.empty()) {
                process_sample(current_matrix);
                current_matrix.clear();
            }
            continue;
        }

        // Read one row
        std::vector<int> row(NUM_CHANNELS, 0);
        std::istringstream iss(line);
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            int val;
            iss >> val;
            row[ch] = val;
            if (ch < NUM_CHANNELS - 1) iss.ignore(1, ','); // skip comma
        }
        current_matrix.push_back(row);
    }

    // Handle last matrix if file does not end with empty line
    if (!current_matrix.empty()) {
        process_sample(current_matrix);
    }

    data_file.close();
    golden_file.close();
    out_file.close();

    std::cout << "Processed " << sample_count << " samples.\n";

    if (num_failures == 0)
        std::cout << "Class check passed! All predicted classes match the golden.\n";
    else
        std::cout << "Class check failed! " << num_failures << " mismatches found.\n";

    if (num_logit_mismatches == 0)
        std::cout << "Logit check passed! All logits within ±0.1 tolerance.\n";
    else
        std::cout << "Logit check failed! " << num_logit_mismatches
                  << " values exceeded ±0.1 difference.\n";

    return (num_failures == 0 && num_logit_mismatches == 0) ? 0 : 1;
}
