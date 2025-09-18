from socket import *

ultraName = '10.104.169.64'
ultraPort = 8888 #check if needs to be 22

clientSocket = socket(AF_INET, SOCK_STREAM)
print("trying to connect to server")
clientSocket.connect((ultraName, ultraPort))
print("Successfully connected to server")


while True:
  message = input ("enter a message: ")
  if not message:
    continue

  clientSocket.send(message.encode())
  receivedMsg = clientSocket.recv(2048)

  print('from server: ', receivedMsg.decode())

clientSocket.close()


