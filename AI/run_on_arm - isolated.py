from scipy.stats import skew
import numpy as np

NUM_OUTPUT = 1
NUM_FEATURES = 8
NUM_DATA = 6 # TBC
NUM_INPUT = NUM_FEATURES * NUM_DATA

SAMPLING_RATE = 20 # TBC
TIME_LIMIT = 3 # TBC
WINDOW_SIZE = SAMPLING_RATE * TIME_LIMIT

input_buffer = np.zeros(NUM_INPUT, dtype=np.int32)
output_buffer = np.zeros(NUM_OUTPUT, dtype=np.int32)


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

    return np.array(features, dtype=np.int32)


def classify_action(data_window):
    global input_buffer, output_buffer, dma, action_map

    window_matrix = np.array(data_window)
    features = extract_features(window_matrix)

    input_buffer[:] = features

    try:
        print(window_matrix)
        print(input_buffer)
        output_buffer[0] = int(input("Output: "))

        action = output_buffer[0]
        return action_map.get(action, "unknown")
    except RuntimeError as e:
        print(e)


def main():
    global data_window
    while True:
        input_signal = input("Key In 1 to start: ")

        if input_signal == "1": # Start signal received
            data_window = [] # Clear previous data

            while len(data_window) < WINDOW_SIZE:
                sample = np.random.randint(-1000, 1000, size=(NUM_DATA,))
                data_window.append(sample)
            
            action = classify_action(data_window)
            print("Detected action:", action)


if __name__ == "__main__":
    main()
