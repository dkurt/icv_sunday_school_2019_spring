# C:\Users\[user]\AppData\Local\Android\sdk\platform-tools\adb.exe forward tcp:43656 tcp:43656
import socket
import struct
import cv2 as cv
import numpy as np

s = socket.socket()
s.connect(('', 43656))  # It's OK to use an empty string as IP for Linux but for Windows use '127.0.0.1'

def read(size):
    batchSize = 4096
    data = ''
    while len(data) < size:
        data += s.recv(min(size - len(data), batchSize))
    return data

def receiveImg():
    rows, cols, channels = struct.unpack('iii', read(4*3))
    data = read(rows * cols * channels)
    return np.fromstring(data, dtype=np.uint8).reshape([rows, cols, channels])

cv.namedWindow('Android', cv.WINDOW_NORMAL)
while cv.waitKey(1) < 0:
    img = receiveImg()
    img = img[:,:,[2, 1, 0]]
    cv.imshow('Android', img)
