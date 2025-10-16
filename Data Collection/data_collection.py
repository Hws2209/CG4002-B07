from socket import *
from Crypto.Cipher import AES
import sys
import time
import threading
import struct
import numpy as np
import torch
from scipy.stats import skew
import winsound
from model_definitions import *

PACKET_SIZE = 20 # bytes
NUM_OF_PACKETS = 20 # expected num of packets per action
HEADER = b'\x55\xAA'   # little-endian of 0xAA55
NUM_CLIENTS = 2 # num of esp

IS_TESTING_MODE = True
MODEL_TYPE = "CNN"
MODEL_PATH = "model_cnn.pt"
DATA_LABELS = ["idle", "raise_left", "raise_right", "raise_both", "wave_left", "wave_right", 
               "wave_both", "circle_left", "circle_right", "circle_both", "clap", "jump"]

#encryption data
key = bytes([
    0x00, 0x01, 0x02, 0x03,
    0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B,
    0x0C, 0x0D, 0x0E, 0x0F
])
cipher = AES.new(key, AES.MODE_ECB)

#threading data
connectedClients = []   # store client connections
lock = threading.Lock()
ultraLock = threading.Lock()
startSignalSent = False
startRecevingFromESP = False
startBarrier = threading.Barrier(NUM_CLIENTS)
msgEndBarrier = threading.Barrier(NUM_CLIENTS+1)

collectedData = []
classCounts = {}

# Try loading the model once if testing mode is enabled
if IS_TESTING_MODE:
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()
    print(f"[MODEL] Loaded model from {MODEL_PATH}")

def extract_features(matrix):
    features = []
    for i in range(matrix.shape[1]): # Each axis
        axis = matrix[:, i]
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

def preprocess(buckets):
    if MODEL_TYPE == "Simplified MLP":
        # Size = (NUM_FEATURES * NUM_DATA * NUM_SENSORS)
        matrix = np.array([extract_features(np.array(bucket)) for bucket in buckets], dtype=np.float32).ravel()
    elif MODEL_TYPE == "MLP":
        # Size = (WINDOW_SIZE * NUM_DATA * NUM_SENSORS)
        matrix = np.concatenate(buckets, axis=0).ravel().astype(np.float32)
    elif MODEL_TYPE == "RNN":
        # Size = (WINDOW_SIZE * NUM_SENSORS, NUM_DATA)
        matrix = np.concatenate(buckets, axis=0).astype(np.float32)
    elif MODEL_TYPE == "CNN":
        # Size = (NUM_DATA, WINDOW_SIZE * NUM_SENSORS)
        matrix = np.concatenate(buckets, axis=0).T.astype(np.float32)
    else:
        raise ValueError("Invalid MODEL_TYPE")

    return matrix


#broadcast msg to all esp
def broadcast(message: str):
    """Send message to all connected clients"""
    with lock:
        for c in connectedClients:
            try:
                c.sendall((message + "\n").encode())
            except:
                pass

def ESP_client(conn, addr):
  global startSignalSent
  global startRecevingFromESP
  print(f"[NEW CONNECTION] {addr} connected.")

  #handshake
  message = conn.recv(18) #read upto number of bytes
  if message == b"HELLO":
    print(f"received HELLO from firebeetle {addr} ")
    msg = "ACK"
    conn.send(msg.encode())
    data = conn.recv(1)  
    deviceID = data[0]
    print("DeviceID:", deviceID)
  else:
    print('did not receive HELLO')

  with lock:
      connectedClients.append(conn)
  startBarrier.wait()
  # Main receive loop
  while True:
    try:
      if not startRecevingFromESP:
         continue
      
      #START RECEIVING DATAAA
      packetCount = 0
      buffer = b''
      while packetCount < NUM_OF_PACKETS:
        buffer = conn.recv(PACKET_SIZE)
        if not buffer:
            break
        dataPacket = buffer
        header, deviceID = struct.unpack("<H H", dataPacket[:4])
        encryptedPayload = dataPacket[4:]
        decryptedPayload = cipher.decrypt(encryptedPayload)
        ax, ay, az, gx, gy, gz, padding = struct.unpack("<hhh hhh I", decryptedPayload)

        packetCount += 1
        # print(f"ESP {deviceID}: packet {packetCount}")
        # print(deviceID, ax, ay, az, gx, gy, gz)
        with ultraLock:
          collectedData.append((deviceID, ax, ay, az, gx, gy, gz))

      msgEndBarrier.wait()
    except ConnectionResetError:
        break

  print(f"[DISCONNECTED] {addr}")
  with lock:
      if conn in connectedClients:
          connectedClients.remove(conn)
  conn.close()
#def set_up_socket(sock, port, bind):
#  sock = socket(AF_INET, SOCK_STREAM)
#  sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
#  sock.bind((bind, port))
#  sock.listen()

def start_server():
  global startRecevingFromESP
    

  #firebeetle connection
  serverPort = 2105
  serverSocket = socket(AF_INET, SOCK_STREAM)
  serverSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
  serverSocket.bind(('0.0.0.0', serverPort))
  serverSocket.listen()


  def accept_clients():
      while True:
        if len(connectedClients) < NUM_CLIENTS:
          conn, addr = serverSocket.accept()
          thread = threading.Thread(target=ESP_client, args=(conn, addr), daemon=True)
          thread.start()
  threading.Thread(target=accept_clients, daemon=True).start()

  #get ready to receive data
  #msg = "action"
  msg = "a"
  while len(connectedClients) < NUM_CLIENTS:
    continue 
  while True:
    input("press enter to receive msg")
    winsound.Beep(1000, 200)

    startTime = time.time()
    startRecevingFromESP = True
    # print(f"[BROADCAST] {msg}")
    broadcast(msg)
    msgEndBarrier.wait()
    startRecevingFromESP = False
    
    print("time taken: ", time.time() - startTime)
    winsound.Beep(600, 700)

    if IS_TESTING_MODE:
      # Organize data into per-sensor buckets
      buckets = [[] for _ in range(NUM_CLIENTS)]
      for row in collectedData:
        deviceID, ax, ay, az, gx, gy, gz = row
        buckets[deviceID - 1].append([ax, ay, az, gx, gy, gz])

      startTime = time.time()
      inputArray = preprocess(buckets)
      inputTensor = torch.tensor(inputArray, dtype=torch.float32).unsqueeze(0)
      with torch.no_grad():
        output = model(inputTensor)
      print("inference time: ", time.time() - startTime)
      print("MODEL OUTPUT:", output)
      predIdx = torch.argmax(output, dim=1).item()
      predLabel = DATA_LABELS[predIdx]
      print(f"PREDICTED CLASS: {predIdx} ({predLabel})")

      collectedData.clear()
      continue # skip to next round

    # Ask for class label
    classInput = input("Enter class (integer) for this round: ")
    if classInput.isdigit():  # valid integer
      classLabel = int(classInput)

      # Save collected data to file
      dataFilename = "data.txt"
      with open(dataFilename, "a") as f:
        for row in collectedData:
          f.write(" ".join(map(str, row)) + "\n")
        f.write("\n") # blank line between rounds

      # Save class label to file
      labelFilename = "label.txt"
      with open(labelFilename, "a") as f:
        f.write(f"{classLabel}\n")
        
      # Update counts
      if classLabel in classCounts:
        classCounts[classLabel] += 1
      else:
        classCounts[classLabel] = 1

      collectedData.clear() # clear buffer for next round
      print(f"[SAVED] Wrote {NUM_CLIENTS * NUM_OF_PACKETS} rows to {dataFilename}")
      print(f"[SAVED] Class {classLabel} written to {labelFilename}")
      print("[CLASS COUNTS]", classCounts)
    else:
      print("[SKIPPED] Invalid class. Data not saved.")
      collectedData.clear()

if __name__ == "__main__":
    start_server()
