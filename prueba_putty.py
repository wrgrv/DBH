import cv2

cap = cv2.VideoCapture(0)

while True_
	ret, frame = cap.read()
	cv2.imshow("asd", frame)
	while True:
	    key = cv2.waitKey(1)
	    if key & 0xFF == ord('q'):
	        break