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
def sound_command(simonSays, expectedClass, mode):
  simonFile = f"./../Interface/audio/simon_says.wav"
  if mode == 1:
    audioFile = f"./../Interface/audio/{expectedClass}.wav"
  else:
    audioFile = f"./../Interface/audio/{mode}{expectedClass}.wav"

  if simonSays:
    play_audio(simonFile)
  play_audio(audioFile)

def check_audio_files():
  fileError = False
  simonFile = f"./../Interface/audio/simon_says.wav"
  if not os.path.exists(simonFile):
    fileError = True
  for i in range(1,12):
    audioFile = f"./../Interface/audio/{i}.wav"
    if not os.path.exists(audioFile):
      fileError = True
  for i in range(21,27):
    audioFile = f"./../Interface/audio/{i}.wav"
    if not os.path.exists(audioFile):
      fileError = True
  audioFile = f"./../Interface/audio/beep.wav"
  if not os.path.exists(audioFile):
    fileError = True
  audioFile = f"./../Interface/audio/lose.wav"
  if not os.path.exists(audioFile):
    fileError = True
  if fileError:
    print ("sound files missing") 
    sys.exit(1)



def ESP_client(conn, addr):
  global gameStarted
  global startRecevingFromESP
  global ultraSocket
  global connectedClients

  #handshake
  message = conn.recv(5)
  if message == b"HELLO":
    msg = "ACK"
    conn.send(msg.encode())
    data = conn.recv(1)  
    deviceID = data[0]
    print("DeviceID Connected:", deviceID)

    if not gameStarted:
      startBarrier.wait()
    # Main receive loop
    while True:
      try:
        if not startRecevingFromESP:
          continue
        
        packetCount = 0
        buffer = b''
        while packetCount < NUM_OF_PACKETS:
          buffer = conn.recv(PACKET_SIZE)
          if not buffer:
            break
          #ensure amt of data received is at least packet_sized
          while len(buffer) >= PACKET_SIZE:
            idx = buffer.find(HEADER)
            if idx != -1 and len(buffer) >= PACKET_SIZE:
              # extract aligned packet
              dataPacket = buffer[idx: idx + PACKET_SIZE]
              # keep leftover for next call (if streaming)
              buffer = buffer[idx + PACKET_SIZE:]
              print(dataPacket.hex())
              packetCount += 1
              print(packetCount)
              with ultraLock:
                ultraSocket.send(dataPacket) #send to ultra96

        msgEndBarrier.wait()
      except threading.BrokenBarrierError:
        #need to flush data?
        continue
      except ConnectionResetError:
        break

  else: #if handshake is not successful
    print('did not receive HELLO')

  print(f"[DISCONNECTED] {addr}")
  with clientLock:
      if conn in connectedClients:
          connectedClients.remove(conn)
  conn.close()

def start_server():
  global startRecevingFromESP, gameStarted
  global startBarrier, msgEndBarrier
  global ultraSocket
  global connectedClients

  check_audio_files()
  
  while True:
    try:
      numPlayers = int(input("Number of players (1 or 2): "))
      if numPlayers in (1, 2):
        break
      else:
        print("Invalid input. Please enter 1 or 2.")
    except ValueError:
      print("Invalid input. Please enter a number (1 or 2).")

  if numPlayers == 2:
    while True:
      try:
        mode = int(input("Game mode (1 - Versus or 2 - Collab): "))
        if mode in (1, 2):
          break
        else:
          print("Invalid input. Please enter 1 or 2.")
      except ValueError:
        print("Invalid input. Please enter a number (1 or 2).")
  else:
     mode = 1

  if mode == 1:
    DATA_LABELS = ["Idle", "Raise left arm", "Raise right arm", "Raise both arms", "Wave left hand", "Wave right hand", 
                   "Wave both hands", "Left arm circle", "Right arm circle", "Both arms circles", "Clap", "Star jump"]
  else:
    DATA_LABELS = ["Idle", "Shake left hand", "Shake right hand", "Shake both hands", "Left high-five", "Right high-five", "Both high-five"]

  numESPs = numPlayers * 2
  print("no. of ESPs to expect: ", numESPs)
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
      global connectedClients
      while True:
        if len(connectedClients) < numESPs:
          #print("len of connectecClients: ", len(connectedClients))
          conn, addr = serverSocket.accept()
          thread = threading.Thread(target=ESP_client, args=(conn, addr), daemon=True)
          thread.start() 
          with clientLock:
            connectedClients.append(conn)
  threading.Thread(target=accept_clients, daemon=True).start()

  #Ultra96 connect
  ultraPort = 8887
  ultraSocket = socket(AF_INET, SOCK_STREAM)
  ultraSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
  ultraSocket.bind(('127.0.0.1', ultraPort))
  ultraSocket.listen()

  ultraSocket, ultraAddr = ultraSocket.accept()
  print('ultra has connected')
  #handshake
  message = ultraSocket.recv(10) #read up to number of bytes
  print(message)
  if message == b"HELLO":
    print('received HELLO from ultra')
    msg = "ACK"
    ultraSocket.send(msg.encode())
    ultraSocket.send(bytes([numESPs, mode]))

    print("Waiting for Ultra96 to load bitstream")
    readyMsg = ultraSocket.recv(5)
    if readyMsg == b"READY":
      print("Ultra96 is ready!")
    else:
       print("Did not receive READY from Ultra")
       sys.exit(1)
  else:
    print("Did not receive HELLO from Ultra")
    sys.exit(1)

  # get ready to receive data
  msg = "a"
  startBarrier.wait()
  gameStarted = True

  if numPlayers == 1 or (numPlayers == 2 and mode == 2):
    highScoreFile = f"high_score_{numPlayers}.txt"
    highScore = load_high_score(highScoreFile)
    print(f"High score: {highScore}")
  
  currentScore = 0
  prevRoundCorrect = False

  while True:
    try:
      if not prevRoundCorrect:
        input("Press enter to start game")
        with clientLock:
          if len(connectedClients) < numESPs:
            print(f"Not enough devices connected: Only {len(connectedClients)} devices connected.")
            continue

      if mode == 1:
        expectedClass = random.randint(1, 11)
      else:
        expectedClass = random.randint(1, 6)
      
      simonSays = 1 if random.random() < 0.8 else 0
      sound_command(simonSays, expectedClass, mode)
      if not simonSays:
        expectedClass = 0

      startTime = time.time()
      startRecevingFromESP = True
      msg = "a"
      broadcast(msg)
      ultraSocket.send(msg.encode())
      msgEndBarrier.wait(timeout=10)
      startRecevingFromESP = False
      espDoneTime = time.time()
      print("Time taken from broadcast message to receiving all packets:", espDoneTime - startTime)

      if numPlayers == 1:
        data = ultraSocket.recv(1)
        ultraDoneTime = time.time()
        predictedClass = data[0]
        print("Expected Action:", DATA_LABELS[expectedClass])
        print("Action Detected:", DATA_LABELS[predictedClass])
        print(f"Time taken from ESP done to Ultra96 result:", ultraDoneTime - espDoneTime)

        if predictedClass == expectedClass:
          currentScore += 1
          print(f"Correct! Current score: {currentScore}")
          prevRoundCorrect = True
          play_audio(f"./../Interface/audio/beep.wav")
          
        else:
          print("\nGame over!")
          print(f"Final score: {currentScore}")
          play_audio(f"./../Interface/audio/lose.wav")
          if currentScore > highScore:
            highScore = currentScore
            save_high_score(highScore, highScoreFile)
          print(f"High score: {highScore}")
          currentScore = 0
          prevRoundCorrect = False

      else:
        data = ultraSocket.recv(2)
        ultraDoneTime = time.time()
        player1Class, player2Class = data[0], data[1]

        print("Expected Action:", DATA_LABELS[expectedClass])
        print(f"Player 1 Action: {DATA_LABELS[player1Class]}")
        print(f"Player 2 Action: {DATA_LABELS[player2Class]}")
        print(f"Time taken from ESP done to Ultra96 result:", ultraDoneTime - espDoneTime)

        if mode == 2:
          if player1Class == expectedClass and player2Class == expectedClass:
            currentScore += 1
            print(f"Correct! Current score: {currentScore}")
            prevRoundCorrect = True
            play_audio(f"./../Interface/audio/beep.wav")

          else:
            print("\nGame over!")
            print(f"Final score: {currentScore}")
            play_audio(f"./../Interface/audio/lose.wav")
            if currentScore > highScore:
              highScore = currentScore
              save_high_score(highScore, highScoreFile)
            print(f"High score: {highScore}")
            currentScore = 0
            prevRoundCorrect = False

        else:
          if player1Class == expectedClass and player2Class == expectedClass:
            currentScore += 1
            print(f"Both correct! Current score: {currentScore}")
            play_audio(f"./../Interface/audio/beep.wav")
            prevRoundCorrect = True
          else: 
            if player1Class != expectedClass and player2Class == expectedClass:
              print("\nPlayer 1 made a mistake! Player 2 wins!")
            elif player1Class == expectedClass and player2Class != expectedClass:
              print("\nPlayer 2 made a mistake! Player 1 wins!")
            else:
              print("\nBoth players made a mistake! No winner this round.")
            currentScore = 0
            prevRoundCorrect = False
            play_audio(f"./../Interface/audio/lose.wav")

    except threading.BrokenBarrierError:
      print("Message timeout occurred, cancelling this round")
      msgEndBarrier.abort()

      msg = "ERROR"
      with ultraLock:
        ultraSocket.send(msg.encode())
      msgEndBarrier.reset()


if __name__ == "__main__":
    start_server()
