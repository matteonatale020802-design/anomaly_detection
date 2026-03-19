import os
import json
import cv2

# --------------------------
# CONFIGURAZIONE (Controlla questi path!)
# --------------------------
images_dir = "dataset/images"       # dove sono le foto
labels_dir = "dataset/labels"       # dove sono i file .txt di YOLO
output_file = "dataset/bounding_boxes.labels" # il file finale

default_label = "component"  # etichetta per tutte le box (puoi cambiarla se vuoi)
# --------------------------

# Struttura UFFICIALE Edge Impulse
data = {
    "version": 1,
    "type": "bounding-box-labels", # FONDAMENTALE: deve esserci questo campo
    "boundingBoxes": {}
}

# Verifichiamo se le cartelle esistono
if not os.path.exists(images_dir):
    print(f"ERRORE: La cartella immagini '{images_dir}' non esiste!")
    exit()

print("Inizio elaborazione...")

for img_filename in sorted(os.listdir(images_dir)):
    if not img_filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_name = os.path.splitext(img_filename)[0]
    label_path = os.path.join(labels_dir, f"{img_name}.txt")
    
    img_boxes = []

    if os.path.exists(label_path):
        # Carichiamo l'immagine per avere le dimensioni reali
        img = cv2.imread(os.path.join(images_dir, img_filename))
        if img is None:
            print(f"Salto {img_filename}: file corrotto o non leggibile.")
            continue
        
        H, W = img.shape[:2]

        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                # YOLO format: class x_center y_center width height (0-1)
                _, x_c, y_c, w_n, h_n = map(float, parts)

                # Conversione in pixel (Edge Impulse vuole coordinate intere)
                # Formula: x_top_left = (x_center - width/2) * image_width
                x = int((x_c - w_n/2) * W)
                y = int((y_c - h_n/2) * H)
                w = int(w_n * W)
                h = int(h_n * H)

                img_boxes.append({
                    "label": default_label,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                })

    # Aggiungiamo al dizionario (usa il nome file come chiave)
    data["boundingBoxes"][img_filename] = img_boxes

# Scrittura ATOMICA del file
try:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        # Usiamo separatori puliti per evitare "grumi" di testo
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno()) # Forza il sistema operativo a scrivere tutto subito
    print(f"SUCCESSO! Generato file per {len(data['boundingBoxes'])} immagini.")
    print(f"File salvato in: {output_file}")
except Exception as e:
    print(f"ERRORE durante la scrittura del file: {e}")