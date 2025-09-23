#import socket 
from socket import *
import sys
import time


PACKET_SIZE = 22 #bytes
NUM_OF_PACKETS = 20 #expected num of packets per action
HEADER = b'\x55\xAA'   # little-endian of 0xAA55


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
  print("num of packets flushed: ", dataSumLen/22)

def recv_exact(socket, n):
  dataPacket = b''
  while len(dataPacket) < n:
    currPacket = socket.recv(n - len(dataPacket))
    if not currPacket:
      raise ConnectionError("Socket is closed")
    dataPacket += currPacket
  return dataPacket


#Ultra96 connect
ultraPort = 8887
ultraSocket = socket(AF_INET, SOCK_STREAM)
ultraSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
ultraSocket.bind(('127.0.0.1', ultraPort))
ultraSocket.listen()
print('Waiting for Ultra to connect')

ultraSocket, clientAddr = ultraSocket.accept()
print('ultra has connected')
#handshake
message = ultraSocket.recv(10) #read upto number of bytes
print(message)
if message == b"HELLO":
  print('received HELLO from ultra')
  msg = "ACK"
  ultraSocket.send(msg.encode())
else:
  print('did not receive HELLO from Ultra')
  sys.exit(1)
   

#firebeetle connection
serverPort = 2105
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
serverSocket.bind(('0.0.0.0', serverPort))
serverSocket.listen()
print('Waiting for firebeetle to connect')
arduinoSocket, clientAddr = serverSocket.accept()
print('A firebeetle has connected')

#handshake
message = arduinoSocket.recv(18) #read upto number of bytes
print(message)
if message == b"HELLO":
  print('received HELLO from firebeetle')
  msg = "ACK"
  arduinoSocket.send(msg.encode())
else:
  print('did not receive HELLO')


#get ready to receive data
duration = 2.2 #2 seconds
msg = "action"
while True:
  input("press enter to receive msg")
    #print('inside loop')
  arduinoSocket.send(msg.encode())
  ultraSocket.send(msg.encode())
  start_time = time.time()
  packetCount = 0
  buffer = b''
  while packetCount < NUM_OF_PACKETS:
    buffer = arduinoSocket.recv(22) #read upto number of bytes
    #if len(dataPacket) < PACKET_SIZE:
    #  print("incorrect len of packet")
    #  continue
    #buffer += recv_exact(arduinoSocket, PACKET_SIZE)
    
    # look for header inside buffer
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
          ultraSocket.send(dataPacket) #send to ultra96
      else:
        print("not enough packet or header not found")
        continue
    
    #header, device_id, ax, ay, az, gx, gy, gz, mx, my, mz = struct.unpack("<H H hhh hhh hhh", dataPacket)

    #if header != 0xAA55:
    #  print("incorrect header! Resync needed")
    #  continue
    #print(" ".join(hex(n) for n in dataPacket))
    #print(dataPacket.hex())
    #packetCount += 1
    #print(packetCount)
  
  #end of recv for 2s
  print("time taken: ", time.time() - start_time)
  #flush_recv(arduinoSocket)


  

connectionSocket.close()