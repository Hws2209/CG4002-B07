from pynq import Overlay, allocate, PL
from scipy.stats import skew
import numpy as np
import time
import logging
from socket import *
import struct
import sys
from Crypto.Cipher import AES

DATA_LABELS = ["idle", "raise_left", "raise_right", "raise_both", "wave_left", "wave_right", 
               "wave_both", "circle_left", "circle_right", "circle_both", "clap", "jump"] # TBC
NUM_CLASSES = len(DATA_LABELS)

MODEL_TYPE = "CNN" # "CNN" | "RNN" | "MLP" | "Simplified MLP"

NUM_FEATURES = 8
WINDOW_SIZE = 20
NUM_DATA = 6
NUM_SENSORS = 2

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


###COMMS SET UP :)###
PACKET_SIZE = 20 #bytes
NUM_OF_PACKETS = 20 #expected num of packets per action
HEADER = b'\x55\xAA'   # little-endian of 0xAA55

key = bytes([
    0x00, 0x01, 0x02, 0x03,
    0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B,
    0x0C, 0x0D, 0x0E, 0x0F
])
cipher = AES.new(key, AES.MODE_ECB)

ultraName = 'localhost'
ultraPort = 8887 

ultraSocket = socket(AF_INET, SOCK_STREAM)
print("trying to connect to server")
ultraSocket.connect((ultraName, ultraPort))
print("Successfully connected to server")
message = "HELLO"
ultraSocket.send(message.encode())
receivedMsg = ultraSocket.recv(3)
if receivedMsg == b"ACK":
  print('received ACK from Laptop')
else:
  print('did not receive ACK from Laptop')
  sys.exit(1)

data = ultraSocket.recv(1)  # 4 bytes for unsigned int
numESPs = data[0]
print("Number of ESPs:", numESPs)
#######################################COMMS#################


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


def classify_action(input_array):
    start_time = time.time()
    pred_logits = get_model_output(input_array)
    pred_class = int(np.argmax(pred_logits))
    end_time = time.time()

    logger.info("Prediction logits: %s", pred_logits)
    logger.info("Predicted class: %d %s", pred_class, DATA_LABELS[pred_class])
    logger.info("Time taken for dma + model: %s", end_time - start_time)

    return pred_class


def main():
    while True:
        receivedMsg = ultraSocket.recv(1)
        if receivedMsg != b"a":
            continue
        startTime= time.time()
        packetCount = 0
        buffer = b''
        buckets = [[] for _ in range(NUM_SENSORS)]

        while packetCount < (NUM_OF_PACKETS * numESPs):
            buffer = ultraSocket.recv(PACKET_SIZE) # read up to number of bytes
            while len(buffer) >= PACKET_SIZE:
                idx = buffer.find(HEADER)
                if idx != -1 and len(buffer) >= PACKET_SIZE:
                    # extract aligned packet
                    dataPacket = buffer[idx: idx + PACKET_SIZE]
                    # keep leftover for next call (if streaming)
                    buffer = buffer[idx + PACKET_SIZE:]
                    header, device_id = struct.unpack("<H H", dataPacket[:4])
                    print(header, device_id)

                    if header != 0xAA55:
                        print("incorrect header! Resync needed")
                        packetCount += 1
                        continue

                    packetCount += 1
                    print(packetCount)

                    encryptedPayload = dataPacket[4:]
                    decryptedPayload = cipher.decrypt(encryptedPayload)
                    ax, ay, az, gx, gy, gz, padding = struct.unpack("<hhh hhh I", decryptedPayload)
                    print(ax, ay, az, gx, gy, gz, padding)
                    buckets[device_id - 1].append([ax, ay, az, gx, gy, gz])


                else:
                    print("not enough packet or header not found")
                    continue
            

        # Received all packets of data
        print("time taken: ", time.time() - startTime)
        logger.info("Received new set of input data. Preprocessing...")
        
        # Form input_array
        if MODEL_TYPE == "Simplified MLP":
            input_array = np.array([extract_features(np.array(bucket)) for bucket in buckets], dtype=np.float32).ravel()
        elif MODEL_TYPE == "MLP" or MODEL_TYPE == "RNN":
            input_array = np.concatenate(buckets, axis=0).ravel().astype(np.int32)
        elif MODEL_TYPE == "CNN":
            input_array = np.concatenate(buckets, axis=0).T.ravel().astype(np.int32)
        else:
            raise ValueError("Invalid MODEL_TYPE")
        
        pred_class = classify_action(input_array)
        ultraSocket.send(bytes([pred_class]))


if __name__ == "__main__":
    main()
