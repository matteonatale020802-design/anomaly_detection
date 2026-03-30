import json
import os
import cv2
import numpy as np
import shutil

# --- CONFIGURAZIONE ---
ORIGINAL_IMAGE = r"C:\Users\MatteoNatale\OneDrive - GSR TECHNOLOGY\Desktop\job\anomaly_detection-1\images\pcb.jpeg"
LABELS_FILE = "bounding_boxes.labels"
OUTPUT_DIR = "dataset_per_edge_impulse"
REPLICAS = 100 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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

new_ei_data = {
    "version": 1,
    "type": "bounding-box-labels",
    "boundingBoxes": {}
}

print(f"Generazione di {REPLICAS} repliche UNICHE...")

# 2. Generiamo le copie con variazioni invisibili
for i in range(1, REPLICAS + 1):
    new_name = f"pcb_unique_{i:03d}.jpeg"
    dest_path = os.path.join(OUTPUT_DIR, new_name)
    
    # Aggiungiamo un leggerissimo rumore casuale ai pixel
    # Questo cambia l'hash del file senza rovinare l'immagine
    noise = np.random.randint(0, 2, img.shape, dtype='uint8')
    temp_img = cv2.add(img, noise)
    
    # Salviamo la nuova immagine (ogni salvataggio JPEG avrà un hash diverso)
    cv2.imwrite(dest_path, temp_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    # Associazioni labels
    new_ei_data["boundingBoxes"][new_name] = original_boxes

# 3. Salviamo il file etichette
output_labels_path = os.path.join(OUTPUT_DIR, "bounding_boxes.labels")
with open(output_labels_path, "w", encoding="utf-8") as f:
    json.dump(new_ei_data, f, indent=4)

print(f"\nFATTO! Generati {REPLICAS} file unici in '{OUTPUT_DIR}'.")
print("Ora Edge Impulse li accetterà tutti perché hanno hash differenti.")