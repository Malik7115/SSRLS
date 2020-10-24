import cv2
import numpy as np
import os
from numpy.linalg import inv
from numpy.linalg import matrix_power


A = np.array([[1 , 1],
              [0 , 1]])

C = np.array([[1 , 0]])

y = np.array([[1,2,3,4,5]])
y = np.transpose(y)



p = 5
for i in range (p):
    if i == 0:
        H = np.matmul(C,matrix_power(A,0))
        
    else:
        H = np.concatenate((np.matmul(C,matrix_power(A,-i)),H), axis = 0)

H_t = np.transpose(H)
F = np.matmul(H_t,H)
F = inv(F)
F = np.matmul(F,H_t)

def drawBB(frame, stats, centroids, num_labels):
    for i in range(num_labels):
        if(stats[i, cv2.CC_STAT_AREA] > 50 and stats[i, cv2.CC_STAT_AREA] < 20000):

            x1 = stats[i, cv2.CC_STAT_LEFT]
            x2 = x1 + stats[i, cv2.CC_STAT_WIDTH]

            y1 = stats[i, cv2.CC_STAT_TOP]
            y2 = y1 + stats[i, cv2.CC_STAT_HEIGHT]

            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,255), 2)

            cx = int(centroids[i, 0])
            cy = int(centroids[i, 1])

            cv2.circle(frame, (cx,cy), 2, (255,255,0), 3)

            return cx, cy

global yx
global yy
yx = np.array([[0]])
yy = np.array([[0]])
np.transpose(yx)
np.transpose(yy)

def SSRLS (frame, cx, cy, frameCount):
    print ("SSRLS")
    global yx
    global yy

    if(frameCount == 1):
        yx[0][0] = cx
        yy[0][0] = cy

    elif(frameCount <= 5):
        yx = np.concatenate((yx,[[cx]]), axis=0) 
        yy = np.concatenate((yy,[[cy]]), axis=0) 

        
    else:
        yx = np.roll(yx, -1)
        yx[4][0] = cx
        yy= np.roll(yy, -1)
        yy[4][0] = cy
    
        est_x  = np.matmul(F,yx)
        pred_x = np.matmul(A,est_x)
        est_y  = np.matmul(F,yy)
        pred_y = np.matmul(A,est_y)

        pred_xPos = int(pred_x[0][0])
        pred_yPos = int(pred_y[0][0])
        cv2.circle(frame, (pred_xPos,pred_yPos), 2, (0,0,255), 3)

  


vid_path = "/home/ibrahim/Projects/SSRLS/"
os.chdir(vid_path)
os.getcwd()

cap = cv2.VideoCapture("20201003_111351.mp4")

frameCount = 0
while(True):
    frameCount += 1 
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1024,512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    __ ,thresh = cv2.threshold(gray, 61, 255, cv2.THRESH_BINARY)

    ## CCL:
    output =  cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)

    num_labels = output[0]
    labels     = output[1]
    stats      = output[2]
    centroids  = output[3]

                                                    
    cx, cy = drawBB(frame, stats, centroids, num_labels)
    SSRLS(frame, cx, cy, frameCount)

    cv2.imshow('thresh',thresh)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



