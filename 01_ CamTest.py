import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
   # frame = cv2.flip(frame, -1)  # Otoceni kamery do -1 v případě, potřeby odkomentovat
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Kamera - Barva', frame)
    cv2.imshow('Kamera - stupne sedi', gray)

    key = cv2.waitKey(30) & 0xff
    if key == ord('q'):  # stisk Q pro ukončení
        break
cap.release()
cv2.destroyAllWindows()