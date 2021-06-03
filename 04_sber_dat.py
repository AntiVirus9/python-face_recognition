import numpy as np
import cv2
import os

faceCascade = cv2.CascadeClassifier('Casca/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

faceID = input('Zadejte ID uzivatele: ')
print('[INFO] Analyza obliceje na kamere, koukejte do kamery a cekejte prosim. . .')
i = 0

while (True):
    ret, frame = cap.read()
    # frame = cv2.flip(frame, -1)  # Otoceni kamery do -1 v případě, potřeby odkomentovat
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
       # roi_gray = gray[y:y + h, x:x + w]
       # roi_color = frame[y:y + h, x:x + w]
        i += 1

        cv2.imwrite("data/User." + str(faceID) + '.' + str(i) + ".jpg", gray[y:y+h,x:x+w])

    cv2.imshow('Kamera - Sber dat', frame)

    key = cv2.waitKey(30) & 0xff
    if key == ord('q'):  # stisk Q pro ukončení
        print('[SHUTDOWN] Program byl prerusen uzivatelem. . .')
        break
    elif i >= 30: # po 30 snimcích ukončit
        print('[SHUTDOWN] Ukoncuji program. . .')
        break

cap.release()
cv2.destroyAllWindows()