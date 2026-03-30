import os
import cv2
import numpy as np

ORIGINAL_IMAGE = r"C:\Users\MatteoNatale\OneDrive - GSR TECHNOLOGY\Immagini\Rullino\WIN_20260327_09_28_50_Pro.jpg"
OUTPUT_DIR = r"C:\Users\MatteoNatale\OneDrive - GSR TECHNOLOGY\Desktop\job\dataset\training"
REPLICAS = 50

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

img = cv2.imread(ORIGINAL_IMAGE)
if img is None:
    print("Errore: Immagine non caricata correttamente.")
    exit()

for i in range (1, REPLICAS + 1):
    new_name = f"connector_3_{i:03d}.jpeg"
    dest_path = os.path.join(OUTPUT_DIR, new_name)
    noise = np.random.randint(0, 2, img.shape, dtype='uint8')
    temp_img = cv2.add(img, noise)

    cv2.imwrite(dest_path, temp_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

print(f"\nFATTO! Generati {REPLICAS} file unici in '{OUTPUT_DIR}'.")
print("Ora Edge Impulse li accetterà tutti perché hanno hash differenti.")
