from socket import *

try:
  serverPort = 2105
  serverSocket = socket(AF_INET, SOCK_STREAM)
  serverSocket.bind(('', serverPort))
  serverSocket.listen()
  print('Server is set up')
  connectionSocket, clientAddr = serverSocket.accept()
  print('Server has connected to a client')
  
  #handshake
  message = connectionSocket.recv(18) #read upto number of bytes
  print(message)
  if message == b"HELLO":
    print('received HELLO from firebeetle')
    msg = "ACK"
    connectionSocket.send(msg.encode())
  else:
    print('did not receive HELLO')




  msg = "action"
  while True:
   input("press enter to receive msg")
    #print('inside loop')
   connectionSocket.send(msg.encode())
   message = connectionSocket.recv(22) #read upto number of bytes
   print(" ".join(hex(n) for n in message))
   print(message.hex())

except KeyboardInterrupt:
  print('end')

  

connectionSocket.close()