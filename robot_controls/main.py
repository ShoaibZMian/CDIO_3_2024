#!/usr/bin/env pybricks-micropython
import socket
import threading
import subprocess
from commandparser import parse_and_execute

bind_ip = "192.168.216.33"
bind_port = 8080

# create and bind a new socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((bind_ip, bind_port))
server.listen(5)
print("Server is listening on %s:%d" % (bind_ip, bind_port))


def clientHandler(client_socket):
    client_socket.send("ready".encode())

    request = client_socket.recv(1024)
    print('Received "' + request.decode() + '" from client')
    command = request.decode()
    parse_and_execute(command)


while True:
    # wait for client to connect
    client, addr = server.accept()
    print("Client connected " + str(addr))
    # create and start a thread to handle the client
    client_handler = threading.Thread(target=clientHandler, args=(client,))
    client_handler.start()