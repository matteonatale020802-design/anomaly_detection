import cv2
import json

# Carica l'immagine originale (Golden)
img_path = "C:\\Users\\matteonatale\\Desktop\\anomaly_detection\\images\\pcb.png"
img = cv2.imread(img_path)

# Carica le coordinate salvate con pixel_position.py
with open("boxes.json", "r") as f:
    boxes = json.load(f)

# Disegna ogni rettangolo e aggiunge un numero identificativo
for i, (x, y, w, h) in enumerate(boxes):
    # Disegna il rettangolo verde (B, G, R)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Aggiunge il testo con l'ID del componente
    label = f"ID:{i}"
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostra l'immagine a video
cv2.imshow("Golden Reference Labeled", img)
print("Premi un tasto qualsiasi sull'immagine per chiudere.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Opzionale: Salva l'immagine con le etichette per riferimento futuro
cv2.imwrite("golden_labeled_reference.jpg", img)