import cv2
import json

image_path = r"C:\Users\MatteoNatale\OneDrive - GSR TECHNOLOGY\Desktop\job\anomaly_detection-1\images\\"
img = cv2.imread(image_path + "pcb.jpeg")
clone = img.copy()

boxes = []
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
            cv2.rectangle(img, (ix, iy), (x, y), (0,255,0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0,255,0), 2)
        
        x_min = min(ix, x)
        y_min = min(iy, y)
        w = abs(x - ix)
        h = abs(y - iy)
        
        boxes.append((x_min, y_min, w, h))
        print("Box:", boxes[-1])

cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_rectangle)

while True:
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        with open("boxes.json", "w") as f:
            json.dump(boxes, f)
        print("Salvato!")
        break
    elif key == ord("q"):
        break

cv2.destroyAllWindows()