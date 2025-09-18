from socket import *

ultraPort = 8888
ultraSocket = socket(AF_INET, SOCK_STREAM)
ultraSocket.bind(('', ultraPort))
ultraSocket.listen()
print('Server is set up')
connectionSocket, clientAddr = ultraSocket.accept()
print('Server has connected to a client')

while True:
    message = connectionSocket.recv(2048)

    connectionSocket.send(message)

connectionSocket.close()