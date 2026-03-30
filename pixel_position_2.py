import cv2
import json
import os

# --- CONFIGURAZIONE ---
FOTO_PATH = r"C:\Users\MatteoNatale\OneDrive - GSR TECHNOLOGY\Desktop\job\anomaly_detection-1\images\\pcb.jpeg"
OUTPUT_FILE = "bounding_boxes.labels" 

img_filename = os.path.basename(FOTO_PATH)

if not os.path.exists(FOTO_PATH):
    print(f"Errore: File non trovato in {FOTO_PATH}")
    exit()

img = cv2.imread(FOTO_PATH)
clone = img.copy()

ei_data = {
    "version": 1,
    "type": "bounding-box-labels",
    "boundingBoxes": {
        img_filename: [] 
    }
}

drawing = False
ix, iy = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = clone.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        
        x_min = int(min(ix, x))
        y_min = int(min(iy, y))
        w = int(abs(x - ix))
        h = int(abs(y - iy))
        
        print(f"\nBox creato a x:{x_min}, y:{y_min}")
        label = input("Inserisci etichetta (es. resistor, capacitor): ").strip()
        
        if not label: label = "component"
            
        ei_data["boundingBoxes"][img_filename].append({
            "label": label,
            "x": x_min,
            "y": y_min,
            "width": w,
            "height": h
        })
        
        cv2.putText(clone, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(clone, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
        print(f"Aggiunto '{label}'")

# --- MODIFICA QUI PER LA FINESTRA ---
window_name = "Edge Impulse Labeler"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Rende la finestra ridimensionabile
cv2.resizeWindow(window_name, 1000, 800)       # Imposta una dimensione iniziale ragionevole
cv2.setMouseCallback(window_name, draw_rectangle)

print("--- ISTRUZIONI ---")
print("1. Disegna il box col mouse (puoi ridimensionare la finestra trascinando i bordi)")
print("2. Scrivi l'etichetta nel terminale")
print("3. 's' per salvare e uscire, 'q' per annullare")

while True:
    cv2.imshow(window_name, img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        try:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(ei_data, f, indent=4, ensure_ascii=False)
            print(f"\nSUCCESSO! File salvato.")
            break
        except Exception as e:
            print(f"Errore: {e}")
            break
    elif key == ord("q"):
        break

cv2.destroyAllWindows()