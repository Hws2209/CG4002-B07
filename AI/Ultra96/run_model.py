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

SAMPLING_RATE = 10
TIME_LIMIT = 2
WINDOW_SIZE = SAMPLING_RATE * TIME_LIMIT

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


def get_model_output(data_window):
    global input_buffer, output_buffer, dma, DATA_LABELS
    logger.info("Preparing input buffer.")

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
        logger.info("Starting DMA send transfer.")
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
    data_window = []

    def classify_action():
        nonlocal data_window
        logger.info("Received new input data")

        # TODO: Pre-processing by identifier
        
        start_time = time.time()
        pred_logits = get_model_output(data_window)
        pred_class = int(np.argmax(pred_logits))
        end_time = time.time()

        logger.info("Prediction logits: %s", pred_logits)
        logger.info("Predicted class: %s", DATA_LABELS[pred_class])
        logger.info("Time taken: %s", end_time - start_time)

        data_window.clear()
        return pred_class
    
    while True:
        print("Comms code not yet here")
        return
        
        # TODO: Receive data
        
        pred_class = classify_action()


if __name__ == "__main__":
    main()
