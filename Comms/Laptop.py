from socket import *

try:
  serverPort = 2105
  serverSocket = socket(AF_INET, SOCK_STREAM)
  serverSocket.bind(('', serverPort))
  serverSocket.listen()
  print('Server is ready to receive the message')
  print('Server is ready to receive the message')
  connectionSocket, clientAddr = serverSocket.accept()
  print('Server is ready to receive the message')
  while True:
    #print('inside loop')
   message = connectionSocket.recv(2048)
   print(message)

except KeyboardInterrupt:
  print('end')

  
  #connectionSocket.send(message)

connectionSocket.close()