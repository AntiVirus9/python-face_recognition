import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier ('Casca/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
   # frame = cv2.flip(frame, -1)  # Otoceni kamery do -1 v případě, potřeby odkomentovat
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

    cv2.imshow('Kamera - Detekce celeho obliceje', frame)

    key = cv2.waitKey(30) & 0xff
    if key == ord('q'):  # stisk Q pro ukončení
        break

cap.release()
cv2.destroyAllWindows()