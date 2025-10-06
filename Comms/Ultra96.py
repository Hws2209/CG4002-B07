from socket import *
import sys
import time

PACKET_SIZE = 16 #bytes
NUM_OF_PACKETS = 20 #expected num of packets per action
HEADER = b'\x55\xAA'   # little-endian of 0xAA55

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

while True:

  receivedMsg = ultraSocket.recv(10)
  if receivedMsg != b"action":
    continue
  startTime= time.time()
  packetCount = 0
  buffer = b''
  while packetCount < (NUM_OF_PACKETS * numESPs):
    buffer = ultraSocket.recv(PACKET_SIZE) #read upto number of bytes
    #if len(dataPacket) < PACKET_SIZE:
    #  print("incorrect len of packet")
    #  continue
    #buffer += recv_exact(ultraSocket, PACKET_SIZE)
    
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
  print("time taken: ", time.time() - startTime)
  #flush_recv(ultraSocket)

clientSocket.close()


