from pynq import Overlay, allocate, PL
from scipy.stats import skew
import numpy as np
import time
import logging


MODEL_TYPE = "CNN" # "CNN" | "RNN" | "MLP" | "Simplified MLP"


logger = logging.getLogger("ai_engine")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


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
    total_compute_time = 0.0

    interactive_input = input("Interactive mode? Y/N: ")
    interactive_mode = interactive_input.upper() == "Y"

    def classify_action():
        nonlocal sample_count, num_failures, total_compute_time, buckets
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
        end_time = time.time()
        total_compute_time += (end_time - start_time)

        logger.info("Prediction logits: %s", pred_logits)
        logger.info("Predicted class: %d %s", pred_class, DATA_LABELS[pred_class])

        golden_logits = golden_logits_matrix[sample_count]
        golden_class = int(np.argmax(golden_logits))

        # Compare output from Ultra96 and laptop
        if pred_class != golden_class:
            num_failures += 1

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
                line_values = [int(x) for x in line.split(" ")]
                device_id = line_values[0]
                sensor_values = line_values[1:]
                buckets[device_id - 1].append(sensor_values)
                
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
        

def setup_ai():
    global DATA_LABELS, MODEL_TYPE, NUM_DATA, NUM_SENSORS
    global input_buffer, output_buffer, dma

    mode = int(input("Enter a number: "))

    if mode == 1:
        DATA_LABELS = ["idle", "raise_left", "raise_right", "raise_both", "wave_left", "wave_right", 
               "wave_both", "circle_left", "circle_right", "circle_both", "clap", "jump"]
    else:
        DATA_LABELS = ["class0", "class1", "class2", "class3", "class4", "class5", "class6", 
                       "class7", "class8", "class9", "class10", "class11"]

    NUM_CLASSES = len(DATA_LABELS)

    NUM_FEATURES = 8
    WINDOW_SIZE = 20
    NUM_DATA = 6
    NUM_SENSORS = 2

    NUM_INPUT = NUM_FEATURES * NUM_DATA * NUM_SENSORS if MODEL_TYPE == "Simplified MLP" else WINDOW_SIZE * NUM_DATA * NUM_SENSORS

    PL.reset() # Reset the programmable logic
    logger.info("Programmable Logic has been reset.")

    if mode == 1:
        ol = Overlay('design_1.bit') # Loads the FPGA bitstream
    else:
        ol = Overlay('design_2.bit')

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


if __name__ == "__main__":
    setup_ai()
    main()
