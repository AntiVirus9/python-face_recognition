import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('treningData.yml')
faceCascade = cv2.CascadeClassifier('Casca/haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX

# Seznam jmen podle ID, ID 0 = Non, ...
name = ['None', 'Pavel Chludil', 'Elon Musk', 'Vaclav Pithart']
ID = 0

while True:
    ret, frame = cam.read()
    #frame = cv2.flip(frame, -1)   # Otoceni kamery do -1 v případě, potřeby odkomentovat
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray, # Otoceni kamery do -1 v případě, potřeby odkomentovat
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(10, 10),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ID, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # 0 = nejlepší shoda
        if (confidence < 100):
            ID = name[ID]
            confidence = "  {0}%".format(round(confidence))
        else:
            ID = "unknown"
            confidence = "  {0}%".format(round(confidence))

        # jméno
        cv2.putText(
            frame,
            str(ID),
            (x+5, y+h-5),
            font,
            1,
            (255, 255, 255),
            2
        )
        # procentuální schoda
        cv2.putText(
            frame,
            str(confidence),
            (x-30, y+30),
            font,
            1,
            (255, 255, 255),
            1
        )

    cv2.imshow('Kamera - Rozpoznavani obliceje', frame)

    key = cv2.waitKey(30) & 0xff
    if key == ord('q'):  # stisk Q pro ukončení
        print('[SHUTDOWN] Program byl prerusen uzivatelem. . .')
        break

cam.release()
cv2.destroyAllWindows()