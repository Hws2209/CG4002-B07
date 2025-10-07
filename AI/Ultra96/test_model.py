from pynq import Overlay, allocate, PL
from scipy.stats import skew
import numpy as np
import time
import logging

# To be updated
DATA_LABELS = ["0-idle", "1-wave", "2-updown", "3-rotate"]
NUM_CLASSES = len(DATA_LABELS)

MODEL_TYPE = "CNN" # "CNN" | "RNN" | "MLP" | "Simplified MLP"

NUM_FEATURES = 8
WINDOW_SIZE = 20
NUM_DATA = 6
NUM_SENSORS = 1 # To be updated

NUM_INPUT = NUM_FEATURES * NUM_DATA * NUM_SENSORS if MODEL_TYPE == "Simplified MLP" else WINDOW_SIZE * NUM_DATA * NUM_SENSORS


logger = logging.getLogger("ai_engine")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


PL.reset() # Reset the programmable logic
logger.info("Programmable Logic has been reset.")

ol = Overlay('design_1.bit') # Loads the FPGA bitstream
logger.info("Overlay loaded: %s", ol)

dma = ol.axi_dma_0 # Direct memory access channel between FPGA and ARM
logger.info("DMA object: %s", dma)

if MODEL_TYPE == "Simplified MLP":
    input_buffer = allocate(shape=(NUM_INPUT,), dtype=np.float32)
else:
    input_buffer = allocate(shape=(NUM_INPUT,), dtype=np.int32) # To store input data to send to FPGA
output_buffer = allocate(shape=(NUM_CLASSES,), dtype=np.float32) # To store output logit from FPGA

logger.info("Input buffer allocated with shape %s", input_buffer.shape)
logger.info("Output buffer allocated with shape %s", output_buffer.shape)


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


def get_model_output(input_array):
    global input_buffer, output_buffer, dma, DATA_LABELS
    logger.info("Preparing input buffer...")
    np.copyto(input_buffer, input_array)

    try:
        logger.info("Starting DMA send transfer...")
        dma.sendchannel.transfer(input_buffer)
        dma.recvchannel.transfer(output_buffer)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        logger.info("DMA receive completed.")

        return output_buffer.copy()
    except RuntimeError as e:
        print(e)
        print("Error config:\n", dma.register_map)


def main():
    buckets = [[] for _ in range(NUM_SENSORS)]
    golden_logits_matrix = np.loadtxt("golden_logits.txt", dtype=np.float32) # Output from testing on laptop

    sample_count = 0
    num_failures = 0
    num_logit_mismatches = 0
    total_compute_time = 0.0

    interactive_input = input("Interactive mode? Y/N: ")
    interactive_mode = interactive_input.upper() == "Y"

    def classify_action():
        nonlocal sample_count, num_failures, num_logit_mismatches, total_compute_time, buckets
        logger.info("Received new input data")

        # Form input_array
        if MODEL_TYPE == "Simplified MLP":
            input_array = np.array([extract_features(np.array(bucket)) for bucket in buckets], dtype=np.float32).ravel()
        elif MODEL_TYPE == "MLP" or MODEL_TYPE == "RNN":
            input_array = np.concatenate(buckets, axis=0).ravel().astype(np.int32)
        elif MODEL_TYPE == "CNN":
            input_array = np.concatenate(buckets, axis=0).T.ravel().astype(np.int32)
        else:
            raise ValueError("Invalid MODEL_TYPE")
        
        start_time = time.time()
        pred_logits = get_model_output(input_array)
        pred_class = int(np.argmax(pred_logits))

        logger.info("Prediction logits: %s", pred_logits)
        logger.info("Predicted class: %s", DATA_LABELS[pred_class])

        end_time = time.time()
        total_compute_time += (end_time - start_time)

        golden_logits = golden_logits_matrix[sample_count]
        golden_class = int(np.argmax(golden_logits))

        # Compare output from Ultra96 and laptop
        if pred_class != golden_class:
            num_failures += 1

        if np.any(np.abs(pred_logits - golden_logits) > 0.01):
            num_logit_mismatches += 1

        buckets = [[] for _ in range(NUM_SENSORS)]
        sample_count += 1

        if sample_count % 50 == 0:
            print(f"Processed {sample_count} samples so far...")

        if not interactive_mode:
            return True
        
        continue_signal = input("Continue? Y/N: ")
        return continue_signal.upper() == "Y"
    
    with open("data.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line: # Empty line indicates end of a matrix
                if any(buckets):
                    if not classify_action():
                        break
                continue
            else:
                # OLD Data Format, To be removed
                buckets[0].append([int(x) for x in line.split(" ")])

                # TODO: NEW Data Format
                # line_values = [int(x) for x in line.split(" ")]
                # device_id = line_values[0]
                # sensor_values = line_values[1:]
                # buckets[device_id - 1].append(sensor_values)
                
        if any(buckets): # Handle last matrix if file does not end with empty line
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
        print("Logit check passed! All logits within 0.01 tolerance.")
    else:
        print(f"Logit check failed! {num_logit_mismatches} values exceeded 0.01 difference.")
        

if __name__ == "__main__":
    main()
