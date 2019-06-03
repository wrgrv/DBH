import cv2
#from real_time_streaming_filter import *

vc = cv2.VideoCapture(0)
#vc = real_time_streaming_filter

def gen():
    while True:
    	
        rval, frame = vc.read()
        
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')