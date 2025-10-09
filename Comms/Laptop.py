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
NUM_CLIENTS = 1 #num of esp 

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

#broadcast msg to all esp
def broadcast(message: str):
    """Send message to all connected clients"""
    with lock:
        for c in connectedClients:
            try:
                c.sendall((message + "\n").encode())
            except:
                pass

def ESP_client(conn, addr, ultraSocket):
  global startSignalSent
  global startRecevingFromESP
  print(f"[NEW CONNECTION] {addr} connected.")

  #handshake
  message = conn.recv(18) #read upto number of bytes
  if message == b"HELLO":
    print(f"received HELLO from firebeetle {addr} ")
    msg = "ACK"
    conn.send(msg.encode())
  else:
    print('did not receive HELLO')

  with lock:
      connectedClients.append(conn)
      # If 4 clients connected, send START once
      #if len(connectedClients) == NUM_CLIENTS and not startSignalSent:
      #    print("[INFO] All 4 clients connected. Sending START...")
      #    broadcast("START")
      #    startSignalSent = True
  startBarrier.wait()
  # Main receive loop
  while True:
    try:
      if not startRecevingFromESP:
         continue
      
      #START RECEIVING DATAAA
      #start_time = time.time()
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

  ###handshake
  ##message = arduinoSocket.recv(18) #read upto number of bytes
  ##print(message)
  ##if message == b"HELLO":
  ##  print('received HELLO from firebeetle')
  ##  msg = "ACK"
  ##  arduinoSocket.send(msg.encode())
  ##else:
  ##  print('did not receive HELLO')
      # Thread for accepting clients
        # Thread for accepting clients
  def accept_clients():
      while True:
        if len(connectedClients) < NUM_CLIENTS:
          conn, addr = serverSocket.accept()
          thread = threading.Thread(target=ESP_client, args=(conn, addr, ultraSocket), daemon=True)
          thread.start()
  threading.Thread(target=accept_clients, daemon=True).start()

  #get ready to receive data
  #msg = "action"
  msg = "a"
  high_score = load_high_score()
  while len(connectedClients) < NUM_CLIENTS:
    continue 
  print(f"High score: {high_score}")
  current_score = 0
  while True:
    #if len(connectedClients) < NUM_CLIENTS:
    #  continue
    input("press enter to receive msg")

    expected_class = random.randint(0, 3)
    # Play audio file
    audio_file = f"./../Interface/audio/{expected_class}.wav"
    if os.path.exists(audio_file):
        play_audio(audio_file)
    else:
        print(f"(Audio file {audio_file} missing — skipping sound)")

    startTime = time.time()
    startRecevingFromESP = True
    print(f"[BROADCAST] {msg}")
    broadcast(msg)
    ultraSocket.send(msg.encode())
    msgEndBarrier.wait()
    startRecevingFromESP = False
    print("time taken: ", time.time() - startTime)
    data = ultraSocket.recv(1)  # 4 bytes for unsigned int
    predicted_class = data[0]
    print("Expected Action:", expected_class)
    print("Action Detected:", predicted_class)
    if predicted_class == expected_class:
        current_score += 1
        print(f"Correct! Current score: {current_score}")
    else:
        print("\nGame over!")
        print(f"Final score: {current_score}")
        if current_score > high_score:
            high_score = current_score
            save_high_score(high_score)
        print(f"High score: {high_score}")
        current_score = 0
  ##  input("press enter to receive msg")
  ##    #print('inside loop')
  ##  arduinoSocket.send(msg.encode())
  ##  ultraSocket.send(msg.encode())
  ##  start_time = time.time()
  ##  packetCount = 0
  ##  buffer = b''
  ##  while packetCount < NUM_OF_PACKETS:
  ##    buffer = arduinoSocket.recv(PACKET_SIZE) #read upto number of bytes
  ##    print(buffer.hex())
  ##    dataPacket = cipher.decrypt(buffer)
  ##    print(dataPacket.hex())
  ##    packetCount += 1
  ##    print(packetCount)
  ##    ultraSocket.send(dataPacket) #send to ultra96

    
    #end of recv for 2s
    #print("time taken: ", time.time() - start_time)
    #flush_recv(arduinoSocket)

if __name__ == "__main__":
    start_server()
