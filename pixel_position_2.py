import cv2
import json
import os

# --- CONFIGURAZIONE ---
FOTO_PATH = r"C:\Users\MatteoNatale\OneDrive - GSR TECHNOLOGY\Desktop\job\anomaly_detection-1\images\pcb_webcam.jpg"
OUTPUT_FILE = "bounding_boxes.labels"
window_name = "Editor_Etichette" # Nome semplice, senza spazi strani

if not os.path.exists(FOTO_PATH):
    print(f"Errore: File non trovato!")
    exit()

img = cv2.imread(FOTO_PATH)
clone = img.copy()

ei_data = {"version": 1, "type": "bounding-box-labels", "boundingBoxes": {os.path.basename(FOTO_PATH): []}}

drawing = False
ix, iy = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, clone
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = clone.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_min, y_min = min(ix, x), min(iy, y)
        w, h = abs(x - ix), abs(y - iy)
        
        # Mostra il rettangolo temporaneo
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow(window_name, img)
        
        # Chiedi l'etichetta
        print(f"\n[INPUT] Box rilevato. Vai sul terminale...")
        label = input("Inserisci etichetta: ").strip() or "component"
        
        # Salva nel dizionario
        ei_data["boundingBoxes"][os.path.basename(FOTO_PATH)].append({
            "label": label, "x": x_min, "y": y_min, "width": w, "height": h
        })
        
        # Disegna permanentemente sul clone
        cv2.rectangle(clone, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
        cv2.putText(clone, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        img = clone.copy()
        print(f"Salvato: {label}. Torna alla foto.")

# --- INIZIALIZZAZIONE FINESTRA ---
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1000, 800)
cv2.setMouseCallback(window_name, draw_rectangle)

print("Comandi: 's' salva, 'q' esce senza salvare.")

while True:
    cv2.imshow(window_name, img)
    key = cv2.waitKey(30) & 0xFF # 30ms di refresh mantengono la finestra attiva
    
    if key == ord("s") or key == ord("S"):
        with open(OUTPUT_FILE, "w") as f:
            json.dump(ei_data, f, indent=4)
        print("FILE SALVATO CON SUCCESSO!")
        break
    elif key == ord("q"):
        print("Uscita...")
        break

cv2.destroyAllWindows()