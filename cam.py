import cv2 
import numpy as np
url = 'http://192.168.2.3:8080/video'
cap = cv2.VideoCapture(url)
while(True):
    ret, frame = cap.read()
    if frame is not None:
        cv2.imshow('frame',frame)
    q = cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()


#import urllib
#import cv2
#import numpy as numpy

#url = ' '
#img_resp = urllib.urlopen(url)
#img_np = np.array(bytearray(img_resp.read()), dtype = np.uint8)
#img = cv2.imdecode(img_np, -1)
#cv2.imshow('cam',img)
#cv2.waitKey(10)
