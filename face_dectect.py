import cv2

img=cv2.imread("friends.jpg")
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #OPENCV lê as imagens em cinza
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray, 1.1, 7) #x,y,w,h. A escala é entre 1.1 e 1.9. Quantas características faciais precisam estar presentes 
print(len(faces)) #quantidade de rostos
#print(faces)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('resultado',img)
cv2.waitKey(0)