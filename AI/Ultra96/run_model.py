from pynq import Overlay, allocate, PL
from scipy.stats import skew
import numpy as np
import time

# TBC
action_map = {
    0: "class1",
    1: "class2",
    2: "class3",
    3: "class4"
}

MODEL_TYPE = "CNN" # "CNN" | "RNN" | "MLP" | "Simplified MLP"

NUM_CLASSES = len(action_map)

NUM_FEATURES = 8

SAMPLING_RATE = 20 # TBC
TIME_LIMIT = 3 # TBC
WINDOW_SIZE = SAMPLING_RATE * TIME_LIMIT

NUM_DATA = 6 # TBC
NUM_INPUT = NUM_FEATURES * NUM_DATA if MODEL_TYPE == "Simplified MLP" else WINDOW_SIZE * NUM_DATA


PL.reset() # Reset the programmable logic
ol = Overlay('design_1.bit') # Loads the FPGA bitstream
dma = ol.axi_dma_0 # Direct memory access channel between FPGA and ARM

if MODEL_TYPE == "Simplified MLP":
    input_buffer = allocate(shape=(NUM_INPUT,), dtype=np.float32)
else:
    input_buffer = allocate(shape=(NUM_INPUT,), dtype=np.int32) # To store input data to send to FPGA
output_buffer = allocate(shape=(NUM_CLASSES,), dtype=np.float32) # To store output logit from FPGA


def extract_features(input):
    features = []
    for i in range(NUM_DATA):
        axis = input[:, i]
        fft_axis = np.fft.fft(axis)
        
        features.extend([
            np.mean(axis),
            np.std(axis),
            np.max(axis),
            np.min(axis),
            np.sqrt(np.mean(axis**2)),
            skew(axis),
            np.max(np.abs(fft_axis)),
            np.max(np.angle(fft_axis))
        ])

    return np.array(features, dtype=np.float32)


def get_model_output(data_window):
    global input_buffer, output_buffer, dma, action_map

    # Prepare input data
    if MODEL_TYPE == "Simplified MLP":
        input = extract_features(np.array(data_window))
    elif MODEL_TYPE == "MLP" or MODEL_TYPE == "RNN":
        input = np.array(data_window, dtype=np.int32).flatten()
    elif MODEL_TYPE == "CNN":
        input = np.array(data_window, dtype=np.int32).T.flatten()
    else:
        raise ValueError("Invalid MODEL_TYPE")

    np.copyto(input_buffer, input)

    try:
        dma.sendchannel.transfer(input_buffer)
        dma.recvchannel.transfer(output_buffer)
        dma.sendchannel.wait()
        dma.recvchannel.wait()

        return output_buffer.copy()
    except RuntimeError as e:
        print(e)
        print("Error config: ", dma.register_map)


def main():
    data_window = []
    golden_logits_matrix = np.loadtxt("golden_logits.txt", dtype=np.float32) # Output from testing on laptop

    sample_count = 0
    num_failures = 0
    num_logit_mismatches = 0
    total_compute_time = 0.0

    interactive_input = input("Interactive mode? Y/N: ")
    interactive_mode = interactive_input.upper() == "Y"

    def classify_action():
        nonlocal sample_count, num_failures, num_logit_mismatches, total_compute_time, data_window
        
        start_time = time.time()
        pred_logits = get_model_output(data_window)
        pred_class = int(np.argmax(pred_logits))

        end_time = time.time()
        total_compute_time += (end_time - start_time)

        golden_logits = golden_logits_matrix[sample_count]
        golden_class = int(np.argmax(golden_logits))

        # Compare output from Ultra96 and laptop
        if pred_class != golden_class:
            num_failures += 1

        if np.any(np.abs(pred_logits - golden_logits) > 0.1):
            num_logit_mismatches += 1

        data_window.clear()
        sample_count += 1

        if sample_count % 50 == 0:
            print(f"Processed {sample_count} samples so far...")

        if not interactive_mode:
            return True
        
        action = action_map.get(pred_class, "unknown") # Get name of class
        print("Action: ", action)

        continue_signal = input("Continue? Y/N: ")
        return continue_signal.upper() == "Y"
    
    with open("data.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line: # Empty line indicates end of a matrix
                if data_window:
                    if not classify_action():
                        break
                continue
            else:
                data_window.append([int(x) for x in line.split(",")])
        if data_window: # Handle last matrix if file does not end with empty line
            classify_action()

    # Print summary
    print(f"Processed {sample_count} samples")
    
    if sample_count > 0:
        print(f"Average time per prediction: {total_compute_time/sample_count:.6f} seconds")

    if num_failures == 0:
        print("Class check passed! All predicted classes match the golden.")
    else:
        print(f"Class check failed! {num_failures} mismatches found.")

    if num_logit_mismatches == 0:
        print("Logit check passed! All logits within ±0.1 tolerance.")
    else:
        print(f"Logit check failed! {num_logit_mismatches} values exceeded ±0.1 difference.")
        

if __name__ == "__main__":
    main()
