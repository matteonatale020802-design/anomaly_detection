import cv2

def main():
    # Prova l'indice 0 (camera predefinita). Se hai una camera USB esterna, prova 1.
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Errore: Impossibile aprire la fotocamera.")
        return

    print("Anteprima avviata. Premi 'q' per chiudere la finestra.")

    while True:
        # Cattura il frame frame-per-frame
        ret, frame = cap.read()

        if not ret:
            print("Errore: Impossibile ricevere il frame (fine dello streaming?).")
            break

        # Mostra il frame in una finestra chiamata 'Camera Output'
        cv2.imshow('Camera Output', frame)

        # Aspetta 1ms e controlla se l'utente preme 'q' per uscire
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Quando tutto è fatto, rilascia la risorsa e chiudi le finestre
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()