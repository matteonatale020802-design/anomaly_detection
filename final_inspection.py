import json
import math

# 1. Carica la tua "Mappa Golden" creata con pixel_position.py
with open("boxes.json", "r") as f:
    golden_boxes = json.load(f)

# Calcola i centri (x, y) dei componenti perfetti per il confronto
golden_centers = []
for (x, y, w, h) in golden_boxes:
    center_x = x + (w / 2)
    center_y = y + (h / 2)
    golden_centers.append((center_x, center_y))

def check_board(predictions, threshold_px=15):
    """
    predictions: lista di oggetti trovati dall'IA di Edge Impulse
    threshold_px: quanti pixel di tolleranza per lo spostamento
    """
    found_count = len(predictions)
    expected_count = len(golden_centers)
    
    print(f"--- Risultato Ispezione ---")
    print(f"Componenti attesi: {expected_count} | Trovati: {found_count}")

    # LOGICA 1: Componenti Mancanti
    if found_count < expected_count:
        print(f"STATO: FALLITO - Mancano {expected_count - found_count} componenti!")
    
    # LOGICA 2: Componenti Spostati
    errors_found = 0
    for pred in predictions:
        # Pred è solitamente un dizionario: {'x': 10, 'y': 20, 'width': 30, 'height': 30}
        p_center_x = pred['x'] + (pred['width'] / 2)
        p_center_y = pred['y'] + (pred['height'] / 2)
        
        # Cerca il componente Golden più vicino
        min_dist = float('inf')
        for g_x, g_y in golden_centers:
            dist = math.sqrt((p_center_x - g_x)**2 + (p_center_y - g_y)**2)
            if dist < min_dist:
                min_dist = dist
        
        # Se la distanza dal "posto giusto" è troppa
        if min_dist > threshold_px:
            print(f"STATO: SPOSTATO - Componente rilevato a {min_dist:.1f}px di distanza!")
            errors_found += 1

    if found_count == expected_count and errors_found == 0:
        print("STATO: SCHEDA OK")

# ESEMPIO DI UTILIZZO (Dati che riceverai dal modello Edge Impulse)
# predictions = model.classify(image)