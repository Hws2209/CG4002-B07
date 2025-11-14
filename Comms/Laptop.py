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


DEBUG = 0
#colors for CLI text
RESET  = "\033[0m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
CYAN   = "\033[36m"
MAGENTA = "\033[95m"
#encryption data
key = bytes([
    0x00, 0x01, 0x02, 0x03,
    0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B,
    0x0C, 0x0D, 0x0E, 0x0F
])
cipher = AES.new(key, AES.MODE_ECB)

CHARKEY = 0x5A
def xor_encrypt(byte_val, key=CHARKEY):
    return bytes([byte_val ^ key])
def xor_decrypt_int(encrypted_int, key=CHARKEY):
    return encrypted_int ^ key
#threading data
connectedClients = []   # store client connections
clientLock = threading.Lock()
ultraLock = threading.Lock()
gameStarted = False
ultraSocket = None

def flush_recv(socket):
  dataSumLen = 0
  socket.setblocking(False)
  try:
    while True:
        data = socket.recv(1024)
        dataSumLen += len(data)
        if not data:
            break
  except BlockingIOError:
    pass  # no more data available
  socket.setblocking(True)
  debug_print("num of packets flushed: ", dataSumLen/20)

def debug_print(*args, **kwargs):
  if DEBUG:
        print(f"{CYAN}[DEBUG]{RESET}", *args, **kwargs)

def print_game_over(currentScore, highScore, highScoreFile, mode=2, loser=0):
  play_nonblock_audio(f"./../Interface/audio/lose.wav", 0.5)
  print(f"{RED}Game over!{RESET}")
  if mode==1 and loser!=0:
    if loser==1:
      print(f"Player 1 made a mistake! Player 2 wins!")
    elif loser==2:
      print("\nPlayer 2 made a mistake! Player 1 wins!")
    else:
      print("\nBoth players made a mistake! No winner this round.")

  print(f"{MAGENTA}Final score:{RESET} {currentScore}")
  if currentScore > highScore:
    highScore = currentScore
    save_high_score(highScore, highScoreFile)
  print(f"{BLUE}High score:{RESET} {highScore}")

def print_correct(currentScore, printScore=True):
  if printScore:
    print(f"{GREEN}Correct!{RESET} Current score: {currentScore}")
  else: 
    print(f"{GREEN}Correct!{RESET}")
  play_audio(f"./../Interface/audio/beep.wav")

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
  for i in range(1,11):
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
  audioFile = f"./../Interface/audio/game.wav"
  if not os.path.exists(audioFile):
    fileError = True
  if fileError:
    print ("sound files missing") 
    sys.exit(1)
  if not pygame.mixer.get_init():
      pygame.mixer.init()



def ESP_client(conn, addr):
  global gameStarted
  global ultraSocket
  global connectedClients

  #handshake
  message = conn.recv(5)
  if message == b"HELLO":
    msg = "ACK"
    conn.send(msg.encode())
    conn.send(xor_encrypt(mode))
    data = conn.recv(1)  
    deviceID = xor_decrypt_int(data[0])
    print("ESP", deviceID, "Connected")

    if not gameStarted:
      startBarrier.wait()
    # Main receive loop
    while True:
      try:
        
        msgStartBarrier.wait()
        flush_recv(conn)
        msg = 'a'
        conn.sendall(msg.encode())
        packetCount = 0
        buffer = b''
        conn.settimeout(7)
        while packetCount < NUM_OF_PACKETS:
          buffer += conn.recv(PACKET_SIZE)
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
              debug_print(dataPacket.hex())
              packetCount += 1
              debug_print(packetCount)

              with ultraLock:
                ultraSocket.sendall(dataPacket) #send to ultra96
        conn.settimeout(None)

        msgEndBarrier.wait()
        classReceived.wait()

        if deviceID==2:
          conn.send(xor_encrypt(player1Class))
        if deviceID==3:
          conn.send(xor_encrypt(player2Class))

      except threading.BrokenBarrierError:
        #need to flush data?
        continue
      except ConnectionResetError:
        break
      except timeout:
        print("ESP timeouted")
        break
  else: #if handshake is not successful
    print('did not receive HELLO')

  print(f"{RED}[DISCONNECTED]{RESET} ESP ID: {deviceID}")
  with clientLock:
      if conn in connectedClients:
          connectedClients.remove(conn)
  conn.close()

def start_server():
  global gameStarted
  global startBarrier, msgEndBarrier, msgStartBarrier, classReceived
  global ultraSocket
  global connectedClients
  global mode
  global player1Class, player2Class

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
    DATA_LABELS = ["Idle", "Wave left hand", "Wave right hand", "Wave both hands", "Left back arm circle", "Right back arm circle", "Both back arms circles", 
                   "Left front arm circle", "Right front arm circle", "Both front arms circles", "Star jump"]
  else:
    DATA_LABELS = ["Idle", "Shake left hand", "Shake right hand", "Shake both hands", "Left high-five", "Right high-five", "Both high-five"]

  numESPs = numPlayers * 2
  print("no. of ESPs to expect: ", numESPs)
  startBarrier = threading.Barrier(numESPs+1)
  msgEndBarrier = threading.Barrier(numESPs+1)
  msgStartBarrier = threading.Barrier(numESPs+1)
  classReceived = threading.Barrier(numESPs+1)

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
  message = ultraSocket.recv(5) #read up to number of bytes
  if message == b"HELLO":
    debug_print('Received HELLO from ultra')
    msg = "ACK"
    ultraSocket.send(msg.encode())
    ultraSocket.send(xor_encrypt(numESPs)+xor_encrypt(mode))
    #ultraSocket.send(bytes([numESPs, mode]))

    debug_print("Waiting for Ultra96 to load bitstream")
    readyMsg = ultraSocket.recv(5)
    if readyMsg == b"READY":
      print("Ultra96 is ready!")
    else:
       print("Did not receive READY from Ultra")
       sys.exit(1)
  else:
    print("Did not receive HELLO from Ultra")
    sys.exit(1)

  msg = "a"
  startBarrier.wait()
  gameStarted = True

  highScoreFile = f"high_score_{numPlayers}_{mode}.txt"
  highScore = load_high_score(highScoreFile)
  print(f"High score: {highScore}")
  
  currentScore = 0
  prevRoundCorrect = False
  tutorialExpectedClass = 1

  while True:
    try:
      if not prevRoundCorrect:
        currentScore = 0
        displayMsg = "\nPress enter to start game. Enter T for tutorial: "

        cliInput = input(displayMsg)
        if cliInput.upper() == "T":
          tutorialMode = True
          tutorialExpectedClass = 1
          print("Entered Tutorial Mode")
        else:
          tutorialMode = False
        with clientLock:
          if len(connectedClients) < numESPs:
            print(f"Not enough devices connected: Only {len(connectedClients)} devices connected.")
            continue
      
      if mode == 1:
        expectedClass = random.randint(1, 10)
      else:
        expectedClass = random.randint(1, 6) 
      simonSays = 1 if random.random() < 0.8 else 0
      
      if tutorialMode:
        simonSays = 1
        expectedClass = tutorialExpectedClass

      sound_command(simonSays, expectedClass, mode)
      if not simonSays:
        expectedClass = 0

      play_nonblock_audio(f"./../Interface/audio/game.wav")
      startTime = time.time()
      msg = "a"
      ultraSocket.send(msg.encode())
      msgStartBarrier.wait()
      msgEndBarrier.wait(timeout=10)
      espDoneTime = time.time()
      broadcastToSendFinPacketsToUltraTime = espDoneTime - startTime
      debug_print("Time taken from broadcast message to receiving all packets:", broadcastToSendFinPacketsToUltraTime)

      data = ultraSocket.recv(numPlayers)
      ultraDoneTime = time.time()
      player1Class = xor_decrypt_int(data[0])
      player2Class = player1Class if numPlayers == 1 else xor_decrypt_int(data[1])
      classReceived.wait()
      ESPDonetoUltra96Result = ultraDoneTime - espDoneTime
      
      debug_print(f"Time taken from ESP done to Ultra96 result:", ESPDonetoUltra96Result )

      if player1Class == expectedClass and player2Class == expectedClass:
        currentScore += 1
        print_correct(currentScore, not tutorialMode)
        prevRoundCorrect = True
        if tutorialMode:
          if tutorialExpectedClass == len(DATA_LABELS)-1:
            prevRoundCorrect = False
            print("Tutorial completed!")
          elif tutorialExpectedClass < len(DATA_LABELS)-1:
            tutorialExpectedClass += 1
      else: #wrong action occurred
        loser = 0
        if numPlayers == 1:
          print("Expected Action:", DATA_LABELS[expectedClass])
          print("Action Detected:", DATA_LABELS[player1Class])
        else: #numPlayers == 2
          print("Expected Action:", DATA_LABELS[expectedClass])
          print(f"Player 1 Action: {DATA_LABELS[player1Class]}")
          print(f"Player 2 Action: {DATA_LABELS[player2Class]}")
          if mode==1: #vs mode
            if player1Class != expectedClass and player2Class == expectedClass:
              loser = 1
            elif player1Class == expectedClass and player2Class != expectedClass:
              loser = 2
            else:
              loser = 3
        if tutorialMode:
          print("Action not performed correctly. Please try again")
          prevRoundCorrect = True
          continue
        print_game_over(currentScore, highScore, highScoreFile,mode, loser)
        prevRoundCorrect = False
          
        
    except threading.BrokenBarrierError:
      print(f"{RED}Message timeout occurred, cancelling this round{RESET}")
      msgEndBarrier.abort()
      prevRoundCorrect = False

      msg = "ERROR"
      with ultraLock:
        ultraSocket.send(msg.encode())
      msgEndBarrier.reset()


if __name__ == "__main__":
    start_server()
