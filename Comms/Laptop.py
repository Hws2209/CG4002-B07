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
DATA_LABELS = ["Idle", "Raise left arm", "Raise right arm", "Raise both arms", "Wave left hand", "Wave right hand", 
               "Wave both hands", "Left arm circle", "Right arm circle", "Both arms circles", "Clap", "Star jump"]

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
gameStarted = False
ultraSocket = None

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



def ESP_client(conn, addr):
  global gameStarted
  global startRecevingFromESP
  global ultraSocket
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
  global startRecevingFromESP, gameStarted
  global numPlayers, numESPs, startBarrier, msgEndBarrier
  global ultraSocket
  
  while True:
    try:
      numPlayers = int(input("Number of players (1 or 2): "))
      if numPlayers in (1, 2):
        break
      else:
        print("Invalid input. Please enter 1 or 2.")
    except ValueError:
      print("Invalid input. Please enter a number (1 or 2).")

  numESPs = numPlayers * 2
  startBarrier = threading.Barrier(numESPs+1)
  msgEndBarrier = threading.Barrier(numESPs+1)

  #firebeetle connection
  serverPort = 2105
  serverSocket = socket(AF_INET, SOCK_STREAM)
  serverSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
  serverSocket.bind(('0.0.0.0', serverPort))
  serverSocket.listen()
  # Thread for accepting clients
  def accept_clients():
      while True:
        if len(connectedClients) < numESPs:
          conn, addr = serverSocket.accept()
          thread = threading.Thread(target=ESP_client, args=(conn, addr), daemon=True)
          thread.start() 
  threading.Thread(target=accept_clients, daemon=True).start()

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
    ultraSocket.send(bytes([numESPs]))
  else:
    print('did not receive HELLO from Ultra')
    sys.exit(1)

  # get ready to receive data
  msg = "a"
  startBarrier.wait()
  gameStarted = True

  if numPlayers == 1:
    highScore = load_high_score()
    print(f"High score: {highScore}")
  
  currentScore = 0
  prevRoundCorrect = False

  while True:

    if not prevRoundCorrect:
      input("Press enter to start game")
      with clientLock:
        if len(connectedClients) < numESPs:
          print(f"Not enough devices connected: Only {len(connectedClients)} devices connected.")
          continue

    simonSays = 1 if random.random() < 0.8 else 0
    expectedClass = random.randint(1, 11)
    sound_command(simonSays, expectedClass)
    if not simonSays:
      expectedClass = 0

    startTime = time.time()
    startRecevingFromESP = True
    broadcast(msg)
    ultraSocket.send(msg.encode())
    msgEndBarrier.wait()
    startRecevingFromESP = False
    print("time taken:", time.time() - startTime)

    if numPlayers == 1:
      data = ultraSocket.recv(1)
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

    elif numPlayers == 2:
      data = ultraSocket.recv(2)
      player1Class, player2Class = data[0], data[1]

      print("Expected Action:", DATA_LABELS[expectedClass])
      print(f"Player 1 Action: {DATA_LABELS[player1Class]}")
      print(f"Player 2 Action: {DATA_LABELS[player2Class]}")

      if player1Class == expectedClass and player2Class == expectedClass:
        currentScore += 1
        print(f"Both correct! Current score: {currentScore}")
        play_audio(f"./../Interface/audio/beep.wav")
        prevRoundCorrect = True

      elif player1Class != expectedClass and player2Class == expectedClass:
        print("\nPlayer 1 made a mistake! Player 2 wins!")
        currentScore = 0
        prevRoundCorrect = False

      elif player2Class != expectedClass and player1Class == expectedClass:
        print("\nPlayer 2 made a mistake! Player 1 wins!")
        currentScore = 0
        prevRoundCorrect = False

      else:
        print("\nBoth players made a mistake! No winner this round.")
        currentScore = 0
        prevRoundCorrect = False

if __name__ == "__main__":
    start_server()
