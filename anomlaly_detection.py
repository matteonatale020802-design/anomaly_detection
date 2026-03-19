
import cv2
import json
import numpy as np
import random
import os

path = "C:\\Users\\matteonatale\\Desktop\\anomaly_detection\\images\\"

img = cv2.imread(path + "pcb.png")

with open("boxes.json", "r") as f:
    boxes = json.load(f)

os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

N = 200

H, W = img.shape[:2]

fill_color = img[262, 183]  

for i in range(N):
    img_copy = img.copy()
   
    present_boxes = []
    
    for (x, y, w, h) in boxes:
        if random.random() < 0.5:
            # Copre il componente (mancante)
            img_copy[y:y+h, x:x+w] = fill_color
        else:
            # Il componente RESTA visibile
            present_boxes.append((x, y, w, h))
            
    # Se non ha rimosso nulla, salta
    if len(present_boxes) == 0:
        continue

    # salva immagine
    img_name = f"img_{i}.jpg"
    cv2.imwrite(f"dataset/images/{img_name}", img_copy)

    # salva annotazione in formato YOLO
    label_name = f"img_{i}.txt"
    with open(f"dataset/labels/{label_name}", "w") as f:
        for (x, y, w, h) in present_boxes:
            
            # conversione in formato YOLO normalizzato
            x_center = (x + w/2) / W
            y_center = (y + h/2) / H
            w_norm = w / W
            h_norm = h / H
            
            class_id = 0  # unica classe: component
            
            f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")
    
        # --- DOPO IL LOOP DEI 200 CAMPIONI ---

    # 1. Salva l'immagine originale (Golden) nel dataset
    img_golden_name = "img_golden.jpg"
    cv2.imwrite(f"dataset/images/{img_golden_name}", img)

    # 2. Salva il file label per la Golden (TUTTE le box presenti)
    with open(f"dataset/labels/img_golden.txt", "w") as f:
        for (x, y, w, h) in boxes:
            x_center = (x + w/2) / W
            y_center = (y + h/2) / H
            w_norm = w / W
            h_norm = h / H
            f.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")

    print("Dataset generato con successo, inclusa l'immagine Golden!")



