#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:15:58 2019

@author: bene
"""

'''
	Simple socket server using threads
'''
import numpy as np
import socket
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

def recvall(sock):
    BUFF_SIZE = 2*4096 # 4 KiB
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or end of data
            break
    return data

def bytes2image(mybytes, startpos = 0, mycropsize=100):
    oneimage = mydata[startpos:startpos+mycropsize*mycropsize*2]
    dt = np.dtype(np.uint16)
    dt = dt.newbyteorder('<')
    myimage_array = np.frombuffer(oneimage , dtype=dt)
    myimage = np.reshape(myimage_array, (mycropsize,mycropsize))
    return myimage

HOST = ''	# Symbolic name, meaning all available interfaces
PORT = 4444	# Arbitrary non-privileged port
BUFFSIZE = 4096 

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
mydata=[]

#open and read the file after the appending:
f = open("test.txt", "a")

#%%
mydata = b''
myimages = []
mycropsize = 200
dataacquisition = False
while True:
    #wait to accept a connection - blocking call
    conn, addr = server.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))    
    while 1:
        
        #myfile = open('testfile.raw', 'w')
        #mymessage = recvall(conn)
        mymessage = conn.recv(BUFF_SIZE)
        #print(mymessage)

#        if dataacquisition==True:
        mydata += mymessage# print(mymessage)
#            
        print(len(mymessage))
#        if len(mymessage)==2:
#            print(mymessage)
#        
#        if len(mymessage)==2 and (int(mymessage)==-1 or dataacquisition==True):
#            dataacquisition = True
#        
#        if len(mymessage)==2 and int(mymessage)==-2:
#            dataacquisition = False
#            myimage = bytes2image(mydata, mycropsize)
#            myimages.append(myimage)
#            plt.imshow(myimage), plt.colorbar(), plt.show()
#            mydata = bytes(0)
#




f.write(str(mydata)) 
f.close()
conn.close()
f.write(str(np.array(mydata)))










print('client disconnected')


server.close()

