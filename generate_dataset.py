import json
import os
import cv2
import numpy as np

# --- CONFIGURAZIONE ---
ORIGINAL_IMAGE = r"C:\Users\MatteoNatale\OneDrive - GSR TECHNOLOGY\Desktop\job\anomaly_detection-1\images\original_ispezione_pcb_scatto_1.jpg"
LABELS_FILE = "bounding_boxes.labels"
OUTPUT_DIR_1 = "dataset/training"
OUTPUT_DIR_2 = "dataset/testing"
REPLICAS = 1500 
SPLIT_RATIO = 0.8  # 80% training, 20% testing

# Creazione cartelle
for d in [OUTPUT_DIR_1, OUTPUT_DIR_2]:
    if not os.path.exists(d):
        os.makedirs(d)

# 1. Carichiamo l'immagine e le etichette
img = cv2.imread(ORIGINAL_IMAGE)
if img is None:
    print("Errore: Immagine non caricata correttamente.")
    exit()

try:
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Errore: Manca il file {LABELS_FILE}")
    exit()

original_filename = list(data["boundingBoxes"].keys())[0]
original_boxes = data["boundingBoxes"][original_filename]

# Strutture per i nuovi file label
train_labels = {"version": 1, "type": "bounding-box-labels", "boundingBoxes": {}}
test_labels = {"version": 1, "type": "bounding-box-labels", "boundingBoxes": {}}

print(f"Generazione di {REPLICAS} repliche (80% Training, 20% Testing)...")

# 2. Generiamo le copie
for i in range(1, REPLICAS + 1):
    new_name = f"pcb_{i:03d}.jpeg"
    
    # Determiniamo la destinazione
    if i <= int(REPLICAS * SPLIT_RATIO):
        dest_dir = OUTPUT_DIR_1
        current_labels = train_labels
    else:
        dest_dir = OUTPUT_DIR_2
        current_labels = test_labels
    
    dest_path = os.path.join(dest_dir, new_name)
    
    # Aggiungiamo un leggerissimo rumore casuale ai pixel
    noise = np.random.randint(0, 2, img.shape, dtype='uint8')
    temp_img = cv2.add(img, noise)
    
    # Salviamo la nuova immagine
    cv2.imwrite(dest_path, temp_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    # Associazioni labels al dizionario corretto
    current_labels["boundingBoxes"][new_name] = original_boxes

# 3. Salviamo i file etichette nelle rispettive cartelle
with open(os.path.join(OUTPUT_DIR_1, "bounding_boxes.labels"), "w", encoding="utf-8") as f:
    json.dump(train_labels, f, indent=4)

with open(os.path.join(OUTPUT_DIR_2, "bounding_boxes.labels"), "w", encoding="utf-8") as f:
    json.dump(test_labels, f, indent=4)

print(f"\nFATTO!")
print(f"Training: {int(REPLICAS * SPLIT_RATIO)} immagini in '{OUTPUT_DIR_1}'")
print(f"Testing: {REPLICAS - int(REPLICAS * SPLIT_RATIO)} immagini in '{OUTPUT_DIR_2}'")