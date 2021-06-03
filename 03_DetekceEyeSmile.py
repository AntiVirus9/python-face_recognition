import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('Casca/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Casca/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('Casca/haarcascade_smile.xml')

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

        eyes = eyeCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(2, 2))
        for (e_x, e_y, e_w, e_h) in eyes:
            cv2.rectangle(roi_color, (e_x, e_y), (e_x+e_w, e_y+e_h), (255, 0, 0), 2)

        smiles = smileCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(2, 2))
        for (s_x, s_y, s_w, s_h) in smiles:
            cv2.rectangle(roi_color, (s_x, s_y), (s_x+s_w, s_y+s_h), (255, 0, 0), 2)

    cv2.imshow('Kamera - Detekce celeho obliceje, oci a ust', frame)

    key = cv2.waitKey(30) & 0xff
    if key == ord('q'):  # stisk Q pro ukončení
        break

cap.release()
cv2.destroyAllWindows()