#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:15:58 2019

@author: bene
"""

'''
	Simple socket server using threads
'''

import socket
import sys


def recvall(sock):
    BUFF_SIZE = 4096 # 4 KiB
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break
    return data

HOST = ''	# Symbolic name, meaning all available interfaces
PORT = 1234	# Arbitrary non-privileged port
BUFSIZE = 4096 

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#Bind socket to local host and port
try:
	server.bind((HOST, PORT))
except socket.error as msg:
    print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])

	
print('Socket bind complete')

#Start listening on socket
server.listen(10)
print('Socket now listening')

#now keep talking with the client
while True:
    #wait to accept a connection - blocking call
    conn, addr = server.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))    
    
    myfile = open('testfile.raw', 'w')
    mydata = recvall(conn)
    print(mydata)
            
    myfile.write(str(mydata))
    print('writing file ....')
    myfile.close()
    print('finished writing file')

conn.close()
print('client disconnected')
  
	
server.close()

