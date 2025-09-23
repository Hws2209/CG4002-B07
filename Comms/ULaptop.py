from socket import *
import sys

ultraPort = 8887
ultraSocket = socket(AF_INET, SOCK_STREAM)
ultraSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
ultraSocket.bind(('127.0.0.1', ultraPort))
#ultraSocket.bind(('', ultraPort))
ultraSocket.listen()
print('Waiting for Ultra to connect')

connectionUltraSocket, clientAddr = ultraSocket.accept()
print('Ultra has connected')
#handshake
message = connectionUltraSocket.recv(10) #read upto number of bytes
print(message)
if message == b"HELLO":
  print('received HELLO from Ultra')
  msg = "ACK"
  connectionUltraSocket.send(msg.encode())
else:
  print('did not receive HELLO from Ultra')
  sys.exit(1)
   

while True:
    message = connectionUltraSocket.recv(2048)
    print("received message: ", message)

    connectionUltraSocket.send(message)

connectionSocket.close()