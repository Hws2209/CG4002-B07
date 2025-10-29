from pynq import Overlay, allocate, PL
from scipy.stats import skew
import numpy as np
import time
import logging
from socket import *
import struct
import sys
from Crypto.Cipher import AES

MODEL_TYPE = "CNN" # "CNN" | "RNN" | "MLP" | "Simplified MLP"


logger = logging.getLogger("ai_engine")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

class msgTimeOutError (Exception):
    pass


def setup_comms():
    global PACKET_SIZE, NUM_OF_PACKETS, HEADER
    global cipher, ultraSocket, numESPs, mode

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

    data = ultraSocket.recv(2)
    numESPs = data[0]
    mode = data[1]
    print("Number of ESPs:", numESPs)
    print("Game mode:", mode)


def setup_ai():
    global DATA_LABELS, NUM_DATA
    global inputBuffer, outputBuffer, dma
    
    NUM_DATA = 6

    PL.reset() # Reset the programmable logic
    logger.info("Programmable Logic has been reset.")

    if mode == 1:
        ol = Overlay('design_1.bit') # Loads the FPGA bitstream
        logger.info("Overlay loaded (design_1.bit): %s", ol)
        DATA_LABELS = ["Idle", "Raise left arm", "Raise right arm", "Raise both arms", "Wave left hand", "Wave right hand", 
                       "Wave both hands", "Left arm circle", "Right arm circle", "Both arms circles", "Clap", "Star jump"]
    else:
        ol = Overlay('design_2.bit')
        logger.info("Overlay loaded (design_2.bit): %s", ol)
        DATA_LABELS = ["Idle", "Shake left hand", "Shake right hand", "Shake both hands", "Left high-five", "Right high-five", "Both high-five"]

    NUM_INPUT = 8 * NUM_DATA * 2 if MODEL_TYPE == "Simplified MLP" else NUM_OF_PACKETS * NUM_DATA * 2
    
    dma = ol.axi_dma_0 # Direct memory access channel between FPGA and ARM
    logger.info("DMA object: %s", dma)

    if MODEL_TYPE == "Simplified MLP":
        inputBuffer = allocate(shape=(NUM_INPUT,), dtype=np.float32)
    else:
        inputBuffer = allocate(shape=(NUM_INPUT,), dtype=np.int32) # To store input data to send to FPGA
    outputBuffer = allocate(shape=(len(DATA_LABELS),), dtype=np.float32) # To store output logit from FPGA

    logger.info("Input buffer allocated with shape %s", inputBuffer.shape)
    logger.info("Output buffer allocated with shape %s", outputBuffer.shape)

    message = "READY"
    ultraSocket.send(message.encode())


def extract_features(input):
    features = []
    for i in range(NUM_DATA):
        axis = input[:, i]
        fftAxis = np.fft.fft(axis)
        
        features.extend([
            np.mean(axis),
            np.std(axis),
            np.max(axis),
            np.min(axis),
            np.sqrt(np.mean(axis**2)),
            skew(axis),
            np.max(np.abs(fftAxis)),
            np.max(np.angle(fftAxis))
        ])

    return np.array(features, dtype=np.float32)


def get_model_output(inputArray):
    global inputBuffer, outputBuffer, dma, DATA_LABELS
    logger.info("Preparing input buffer...")
    np.copyto(inputBuffer, inputArray)

    try:
        logger.info("Starting DMA send transfer...")
        dma.sendchannel.transfer(inputBuffer)
        dma.recvchannel.transfer(outputBuffer)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        logger.info("DMA receive completed.")

        return outputBuffer.copy()
    except RuntimeError as e:
        print(e)
        print("Error config:\n", dma.register_map)


def classify_action(inputArray):
    startTime = time.time()
    predLogits = get_model_output(inputArray)
    predClass = int(np.argmax(predLogits))
    endTime = time.time()

    logger.info("Prediction logits: %s", predLogits)
    logger.info("Predicted class: %d %s", predClass, DATA_LABELS[predClass])
    logger.info("Time taken for dma + inference: %s", endTime - startTime)

    return predClass


def process_buckets(buckets):
    # Form inputArray
    if MODEL_TYPE == "Simplified MLP":
        inputArray = np.array([extract_features(np.array(bucket)) for bucket in buckets], dtype=np.float32).ravel()
    elif MODEL_TYPE == "MLP" or MODEL_TYPE == "RNN":
        inputArray = np.concatenate(buckets, axis=0).ravel().astype(np.int32)
    elif MODEL_TYPE == "CNN":
        inputArray = np.concatenate(buckets, axis=0).T.ravel().astype(np.int32)
    else:
        raise ValueError("Invalid MODEL_TYPE")
    return inputArray


def main():
    while True:
        try:
            receivedMsg = ultraSocket.recv(1)
            if receivedMsg != b"a":
                continue
            startTime = time.time()
            packetCount = 0
            buffer = b''
            buckets = [[] for _ in range(numESPs)]

            while packetCount < (NUM_OF_PACKETS * numESPs):
                buffer = ultraSocket.recv(PACKET_SIZE) # read up to number of bytes
                if buffer == b"ERROR":
                    print('Cancelling Ultra current round')
                    raise msgTimeOutError()
                while len(buffer) >= PACKET_SIZE:
                    idx = buffer.find(HEADER)
                    if idx != -1 and len(buffer) >= PACKET_SIZE:
                        # extract aligned packet
                        dataPacket = buffer[idx: idx + PACKET_SIZE]
                        # keep leftover for next call (if streaming)
                        buffer = buffer[idx + PACKET_SIZE:]
                        header, deviceID = struct.unpack("<H H", dataPacket[:4])
                        print(header, deviceID)

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
                        buckets[deviceID - 1].append([ax, ay, az, gx, gy, gz])


                    else:
                        print("not enough packet or header not found")
                        continue
                

            # Received all packets of data
            print("time taken to receive all data: ", time.time() - startTime)
            logger.info("Received new set of input data. Preprocessing...")
            
            if numESPs == 2:
                inputArray = process_buckets(buckets)
                predClass = classify_action(inputArray)
                ultraSocket.send(bytes([predClass]))
            else:
                player1 = buckets[:2]
                player2 = buckets[2:]
                inputArray1 = process_buckets(player1)
                inputArray2 = process_buckets(player2)
                predClass1 = classify_action(inputArray1)
                predClass2 = classify_action(inputArray2)
                ultraSocket.send(bytes([predClass1, predClass2]))
        except msgTimeOutError: 
            continue


if __name__ == "__main__":
    setup_comms()
    setup_ai()
    main()
