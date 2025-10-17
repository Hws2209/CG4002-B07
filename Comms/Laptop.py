#import socket 
from socket import *
from Crypto.Cipher import AES
import sys
import time
import threading
import struct

sys.path.append("./../Interface")  # relative to where you run Laptop.py
from cli import *

PACKET_SIZE = 20 #bytes
NUM_OF_PACKETS = 20 #expected num of packets per action
HEADER = b'\x55\xAA'   # little-endian of 0xAA55
NUM_CLIENTS = 2 #num of esp
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
clientLock = threading.Lock()
ultraLock = threading.Lock()
startRecevingFromESP = False
startBarrier = threading.Barrier(NUM_CLIENTS+1)
msgEndBarrier = threading.Barrier(NUM_CLIENTS+1)
gameStarted = False

#broadcast msg to all esp
def broadcast(message: str):
    """Send message to all connected clients"""
    with clientLock:
        for c in connectedClients:
            try:
                c.sendall((message + "\n").encode())
            except:
                pass

# Play audio file
def sound_command(simonSays, expectedClass):
  simonFile = f"./../Interface/audio/simon_says.wav"
  audioFile = f"./../Interface/audio/{expectedClass}.wav"
  if simonSays:
    if os.path.exists(simonFile):
        play_audio(simonFile)
    else:
          print(f"(Audio file {simonFile} missing — skipping sound)")
  if os.path.exists(audioFile):
      play_audio(audioFile)
  else:
        print(f"(Audio file {audioFile} missing — skipping sound)")



def ESP_client(conn, addr, ultraSocket):
  global gameStarted
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

  with clientLock:
    connectedClients.append(conn)
  if not gameStarted:
    startBarrier.wait()
  # Main receive loop
  while True:
    try:
      if not startRecevingFromESP:
         continue
      
      # START RECEIVING DATAAA
      packetCount = 0
      buffer = b''
      while packetCount < NUM_OF_PACKETS:
        buffer = conn.recv(PACKET_SIZE)
        if not buffer:
            break
        print(buffer.hex())
        #dataPacket = cipher.decrypt(buffer)
        dataPacket = buffer
        print(dataPacket.hex())
        packetCount += 1
        print(packetCount)
        with ultraLock:
          ultraSocket.send(dataPacket) #send to ultra96

      msgEndBarrier.wait()
    except ConnectionResetError:
        break

  print(f"[DISCONNECTED] {addr}")
  with clientLock:
      if conn in connectedClients:
          connectedClients.remove(conn)
  conn.close()

def start_server():
  global startRecevingFromESP
  global gameStarted
  #Ultra96 connect
  ultraPort = 8887
  ultraSocket = socket(AF_INET, SOCK_STREAM)
  ultraSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
  ultraSocket.bind(('127.0.0.1', ultraPort))
  ultraSocket.listen()
  print('Waiting for Ultra to connect')

  ultraSocket, ultraAddr = ultraSocket.accept()
  print('ultra has connected')
  #handshake
  message = ultraSocket.recv(10) #read upto number of bytes
  print(message)
  if message == b"HELLO":
    print('received HELLO from ultra')
    msg = "ACK"
    ultraSocket.send(msg.encode())
    #msg = NUM_CLIENTS
    ultraSocket.send(bytes([NUM_CLIENTS]))
  else:
    print('did not receive HELLO from Ultra')
    sys.exit(1)
    

  #firebeetle connection
  serverPort = 2105
  serverSocket = socket(AF_INET, SOCK_STREAM)
  serverSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
  serverSocket.bind(('0.0.0.0', serverPort))
  #serverSocket.bind(('', serverPort))
  serverSocket.listen()
  ##print('Waiting for firebeetle to connect')
  ##arduinoSocket, clientAddr = serverSocket.accept()
  ##print('A firebeetle has connected')

  # Thread for accepting clients
  def accept_clients():
      while True:
        if len(connectedClients) < NUM_CLIENTS:
          conn, addr = serverSocket.accept()
          thread = threading.Thread(target=ESP_client, args=(conn, addr, ultraSocket), daemon=True)
          thread.start()
  threading.Thread(target=accept_clients, daemon=True).start()

  #get ready to receive data
  msg = "a"
  startBarrier.wait()
  gameStarted = True
  highScore = load_high_score()
  print(f"High score: {highScore}")
  currentScore = 0
  prevRoundCorrect = False
  while True:
    with clientLock:
      if len(connectedClients) < NUM_CLIENTS:
        continue

    if not prevRoundCorrect:
      input("Press enter to start game")

    simonSays = 1 if random.random() < 0.8 else 0
    expectedClass = random.randint(1, 11)
    sound_command(simonSays, expectedClass)
    if not simonSays:
      expectedClass = 0

    startTime = time.time()
    startRecevingFromESP = True
    #print(f"[BROADCAST] {msg}")
    broadcast(msg)
    ultraSocket.send(msg.encode())
    msgEndBarrier.wait()
    startRecevingFromESP = False
    print("time taken: ", time.time() - startTime)
    data = ultraSocket.recv(1)  # 4 bytes for unsigned int
    predictedClass = data[0]
    print("Expected Action:", DATA_LABELS[expectedClass])
    print("Action Detected:", DATA_LABELS[predictedClass])
    if predictedClass == expectedClass:
        currentScore += 1
        print(f"Correct! Current score: {currentScore}")
        prevRoundCorrect = True
        play_audio(f"./../Interface/audio/beep.wav")
    else:
        print("\nGame over!")
        print(f"Final score: {currentScore}")
        if currentScore > highScore:
            highScore = currentScore
            save_high_score(highScore)
        print(f"High score: {highScore}")
        currentScore = 0
        prevRoundCorrect = False

if __name__ == "__main__":
    start_server()
