import cv2 as cv
import os
import numpy as np

net = cv.dnn.readNetFromTensorflow('edges2cats.pb')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

size = 256

canvas = np.ones([size, size, 3], dtype=np.float32)
pressed = False
lastX, lastY = -1, -1
def onMouse(e, x, y, flags, userdata):
    global lastX, lastY, pressed
    if e == cv.EVENT_LBUTTONDOWN or e == cv.EVENT_MOUSEMOVE and pressed:
        if x < 0 or y < 0:
            lastX, lastY = -1, -1
            return
        if lastX > 0 and lastY > 0:
            cv.line(canvas, (lastX, lastY), (x, y), 0)
        else:
            canvas[y, x] = 0
        pressed = True
        lastX, lastY = x, y
    else:
        pressed = False
        lastX, lastY = -1, -1

cv.namedWindow('canvas', cv.WINDOW_NORMAL)
cv.setMouseCallback('canvas', onMouse)
cv.resizeWindow('canvas', 960, 960)

i = 0
if not os.path.exists('samples'):
    os.mkdir('samples')
while True:
    key = cv.waitKey(1)
    if key == 32:
        canvas[:,:,:] = 1
    elif key == 13:
        blob = cv.dnn.blobFromImage(canvas, 2.0, (size, size), (0.5, 0.5, 0.5), False, False)
        net.setInput(blob)
        out = net.forward()
        out += 1
        out /= 2
        res = out.transpose(0, 2, 3, 1).reshape(size, size, 3)[:,:,[2, 1, 0]]
        merged = np.concatenate((canvas, res), axis=1)
        while os.path.exists(os.path.join('samples', '%06d.png' % i)):
            i += 1
        cv.imwrite(os.path.join('samples', '%06d.png' % i), merged * 255)

        cv.imshow('edges2cats using OpenCV', merged)
        canvas[:,:,:] = res
    elif key == 27:
        break
    cv.imshow('canvas', canvas)
