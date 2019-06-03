import cv2

FILENAME = 'a.png'

def mostrar (filename):
    while True:
        imagen = cv2.imread(filename)
        cv2.imshow('filename.jpg', imagen)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

def main():
    #mostrar (FILENAME)
    vc = cv2.VideoCapture(0)
    rval, frame = vc.read()
    cv2.imwrite('a.jpg', frame)
    #cv2.imwrite('a.png', imagen)
    while True:
	    key = cv2.waitKey(1)
	    if key & 0xFF == ord('q'):
	        break

main()