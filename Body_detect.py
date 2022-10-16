import cv2
import numpy as np

body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
vid = cv2.VideoCapture('teste.mp4')

while True:    
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Video', frame)

    if cv2.waitKey(1) == 32:
        break

vid.release()
cv2.destroyAllWindows()