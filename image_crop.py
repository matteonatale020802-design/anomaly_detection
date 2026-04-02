import json
import os
import cv2
import numpy as np
import random

# --- CONFIGURAZIONE ---
ORIGINAL_IMAGE = r"C:\Users\MatteoNatale\OneDrive - GSR TECHNOLOGY\Desktop\job\anomaly_detection-1\images\pcb.jpeg"
LABELS_FILE = r"C:\Users\MatteoNatale\OneDrive - GSR TECHNOLOGY\Desktop\job\anomaly_detection-1\bounding_boxes.labels"
OUTPUT_DIR_TRAIN = r"C:\Users\MatteoNatale\OneDrive - GSR TECHNOLOGY\Desktop\job\anomaly_detection-1\dataset\training"
OUTPUT_DIR_TEST = r"C:\Users\MatteoNatale\OneDrive - GSR TECHNOLOGY\Desktop\job\anomaly_detection-1\dataset\testing"

REPLICAS = 100 
SPLIT_RATIO = 0.8 
CROP_SIZE = (640, 640) 
MIN_VISIBILITY_RATIO = 0.25 # Scarta componenti visibili per meno del 25%

# Creazione cartelle di output
for d in [OUTPUT_DIR_TRAIN, OUTPUT_DIR_TEST]:
    if not os.path.exists(d):
        os.makedirs(d)

# 1. Caricamento Risorse
img = cv2.imread(ORIGINAL_IMAGE)
if img is None:
    print("Errore: Immagine originale non trovata.")
    exit()

h_orig, w_orig, _ = img.shape

try:
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Errore: Il file {LABELS_FILE} non esiste.")
    exit()

# Estraiamo i box originali
original_filename = list(data["boundingBoxes"].keys())[0]
original_boxes = data["boundingBoxes"][original_filename]

# Dizionari per Edge Impulse
train_labels = {"version": 1, "type": "bounding-box-labels", "boundingBoxes": {}}
test_labels = {"version": 1, "type": "bounding-box-labels", "boundingBoxes": {}}

print(f"Inizio generazione di {REPLICAS} ritagli...")

# 2. Loop di Generazione
count_saved = 0
for i in range(1, REPLICAS + 1):
    # Selezione destinazione (Train o Test)
    is_train = i <= int(REPLICAS * SPLIT_RATIO)
    current_dir = OUTPUT_DIR_TRAIN if is_train else OUTPUT_DIR_TEST
    current_dict = train_labels if is_train else test_labels
    
    # Punto di ritaglio casuale
    x_start = random.randint(0, max(0, w_orig - CROP_SIZE[0]))
    y_start = random.randint(0, max(0, h_orig - CROP_SIZE[1]))
    
    crop_boxes = []
    
    for box in original_boxes:
        # Calcolo area originale
        orig_area = box['width'] * box['height']
        
        # Coordinate nel nuovo sistema del ritaglio (clamped ai bordi del crop)
        nx1 = max(0, box['x'] - x_start)
        ny1 = max(0, box['y'] - y_start)
        nx2 = min(CROP_SIZE[0], (box['x'] - x_start) + box['width'])
        ny2 = min(CROP_SIZE[1], (box['y'] - y_start) + box['height'])
        
        # Dimensioni effettive nel ritaglio
        v_w = max(0, nx2 - nx1)
        v_h = max(0, ny2 - ny1)
        v_area = v_w * v_h
        
        # Controllo visibilità (es. se rimane solo il 10% del componente, lo ignoriamo)
        if v_area / orig_area >= MIN_VISIBILITY_RATIO:
            crop_boxes.append({
                "label": box['label'],
                "x": int(nx1),
                "y": int(ny1),
                "width": int(v_w),
                "height": int(v_h)
            })

    # MODIFICA RICHIESTA: Salviamo solo se ci sono etichette valide
    if len(crop_boxes) > 0:
        new_name = f"pcb_crop_{i:03d}.jpeg"
        crop_img = img[y_start:y_start+CROP_SIZE[1], x_start:x_start+CROP_SIZE[0]]
        
        # Salvataggio fisico file
        cv2.imwrite(os.path.join(current_dir, new_name), crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # Aggiunta al JSON
        current_dict["boundingBoxes"][new_name] = crop_boxes
        count_saved += 1

# 3. Scrittura file .labels finali
with open(os.path.join(OUTPUT_DIR_TRAIN, "bounding_boxes.labels"), "w", encoding="utf-8") as f:
    json.dump(train_labels, f, indent=4)

with open(os.path.join(OUTPUT_DIR_TEST, "bounding_boxes.labels"), "w", encoding="utf-8") as f:
    json.dump(test_labels, f, indent=4)

print(f"\nOperazione completata con successo!")
print(f"Ritagli validi generati: {count_saved} su {REPLICAS} tentati.")
print(f" - Training: {len(train_labels['boundingBoxes'])} immagini")
print(f" - Testing: {len(test_labels['boundingBoxes'])} immagini")