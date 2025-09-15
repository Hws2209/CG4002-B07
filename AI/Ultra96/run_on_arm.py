from pynq import Overlay, allocate, PL
from scipy.stats import skew
import numpy as np

NUM_OUTPUT = 1
NUM_FEATURES = 8
NUM_DATA = 6 # TBC
NUM_INPUT = NUM_FEATURES * NUM_DATA

SAMPLING_RATE = 20 # TBC
TIME_LIMIT = 3 # TBC
WINDOW_SIZE = SAMPLING_RATE * TIME_LIMIT

MODEL_TYPE = "CNN"  # "CNN" | "RNN" | "MLP" | "Simplified MLP"

PL.reset() # Reset the programmable logic
ol = Overlay('design_1.bit') # Loads the FPGA bitstream
cnn_ip = ol.cnn_forward_0
dma = ol.axi_dma_0 # Direct memory access channel between FPGA and ARM

if MODEL_TYPE == "Simplified MLP":
    input_buffer = allocate(shape=(NUM_INPUT,), dtype=np.float32)
else:
    input_buffer = allocate(shape=(NUM_INPUT,), dtype=np.int32) # To store input features to send to FPGA
output_buffer = allocate(shape=(NUM_OUTPUT,), dtype=np.float32) # To store predicted action from FPGA

# TBC
action_map = {
    0: "raise_left_arm",
    1: "raise_right_arm",
    2: "kick_left_leg",
    3: "kick_right_leg",
    4: "wave_left_hand",
    5: "wave_right_hand",
    6: "step_left",
    7: "step_right",
    8: "jump",
    9: "clap",
    10: "touch_toes",
    11: "none"
}


# Check if overlay loaded successfully
print("Overlay loaded:", ol.is_loaded())

# Print available IP blocks (should show cnn_forward, DMA, etc.)
print("Available IPs:", list(ol.ip_dict.keys()))

# Print available memory-mapped registers
print("Available MMIO regions:", ol.ip_dict.keys())

# Check DMA blocks
print("Available DMA blocks:", ol.dma_dict.keys())


# Store data after start signal
data_window = []


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


def classify_action(data_window):
    global input_buffer, output_buffer, dma, action_map

    window_matrix = np.array(data_window)
    features = extract_features(window_matrix)

    input_buffer[:] = features

    print("Initial config:\n", dma.register_map)
    try:
        dma.sendchannel.transfer(input_buffer)
        dma.recvchannel.transfer(output_buffer)
        dma.sendchannel.wait()
        dma.recvchannel.wait()

        action = output_buffer[0]
        return action_map.get(action, "unknown")
    except RuntimeError as e:
        print(e)
        print("Error config: ", dma.register_map)


def main():
    global data_window
    while True:
        input_signal = input("Key In 1 to start: ")

        if input_signal == "1": # TBC, Start signal received
            data_window = [] # Clear previous data

            while len(data_window) < WINDOW_SIZE:
                # TBC, Replace this with real IMU data sampling
                sample = np.random.randint(-1000, 1000, size=(NUM_DATA,))
                data_window.append(sample)
            
            action = classify_action(data_window)
            print("Detected action:", action)


if __name__ == "__main__":
    main()
