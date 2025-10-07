import struct
import numpy as np
from scipy.stats import skew

# Constants
PACKET_SIZE = 16
NUM_SENSORS = 4
WINDOW_SIZE = 5
TOTAL_WINDOW_SIZE = WINDOW_SIZE * NUM_SENSORS
CHUNK_SIZE = TOTAL_WINDOW_SIZE * PACKET_SIZE

# Format string for unpacking (skip 2-byte header)
# H = uint16, h = int16
fmt = '<Hhhhhhh'  # '>': big-endian, '<': little-endian

MODEL_TYPE = "Simplified MLP" # "CNN" | "RNN" | "MLP" | "Simplified MLP"


def extract_features(matrix):
    features = []
    for i in range(matrix.shape[1]): # Each axis
        axis = matrix[:, i]
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

def main():
    count = 0
    with open("packets.bin", "rb") as in_file:
        while True:
            # Try to read one full window worth of data
            chunk = in_file.read(CHUNK_SIZE)
            if not chunk or len(chunk) < CHUNK_SIZE:
                break  # stop if file is finished
            
            data_window = []
            buckets = [[] for _ in range(NUM_SENSORS)]

            # Sort by identifiers
            for i in range(TOTAL_WINDOW_SIZE):
                start = i * PACKET_SIZE
                end = start + PACKET_SIZE
                packet_bytes = chunk[start:end]

                # Unpack (skip first 2 bytes = header)
                device_id, ax, ay, az, gx, gy, gz = struct.unpack(fmt, packet_bytes[2:])
                buckets[device_id - 1].append([ax, ay, az, gx, gy, gz])
            
            # Form data_window
            if MODEL_TYPE == "Simplified MLP":
                data_window = [extract_features(np.array(bucket)).tolist() for bucket in buckets]
            else:
                data_window = [row for bucket in buckets for row in bucket]
                
            count += 1
            print(f"\nNumber {count} data window:", data_window)
            print("Shape:", len(data_window), len(data_window[0]))

            if MODEL_TYPE == "Simplified MLP":
                input_array = np.array([extract_features(np.array(bucket)) for bucket in buckets], dtype=np.float32).ravel()
            elif MODEL_TYPE == "MLP" or MODEL_TYPE == "RNN":
                input_array = np.concatenate(buckets, axis=0).ravel().astype(np.int32)
            elif MODEL_TYPE == "CNN":
                input_array = np.concatenate(buckets, axis=0).T.ravel().astype(np.int32)
            else:
                raise ValueError("Invalid MODEL_TYPE")
            
            print(f"\nNumber {count} input array:", input_array)
            print("Type:", type(input_array))
            print("Shape:", input_array.shape)

            return # Only need one window for now

if __name__ == "__main__":
    main()
