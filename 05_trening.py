import cv2
import numpy as np
import os
from PIL import Image

path = 'data' #Složka, kde jsou uložené data z 04_sber_dat.py
rec = cv2.face.LBPHFaceRecognizer_create()
detekce = cv2.CascadeClassifier("Casca/haarcascade_frontalface_default.xml")

def img(path):
    img_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face = []
    IDs = []
    for img_paths in img_paths:
        PIL_img = Image.open(img_paths).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(img_paths)[-1].split(".")[1])
        faces = detekce.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            face.append(img_numpy[y:y+h, x:x+w])
            IDs.append(id)
    return face, IDs

print("[INFO] Trenovani obliceju, cekejte prosim. . .")

faces, IDs = img(path)
rec.train(faces, np.array(IDs))

rec.write('treningData.yml')

print("[INFO] Bylo nauceno {0} obliceju".format(len(np.unique(IDs))))
