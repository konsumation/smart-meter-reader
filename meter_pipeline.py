import torch
from PIL import Image
import easyocr
import cv2
import numpy as np

# YOLOv5-Modell laden
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Pfad zu deinem trainierten Modell

# EasyOCR-Reader initialisieren
reader = easyocr.Reader(['en'], gpu=True)

def process_image(image_path):
    # Bild laden
    img = Image.open(image_path)
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # YOLOv5 ausführen
    results = model(image_path)

    # Ergebnisse anzeigen
    detections = results.xyxy[0].numpy()  # [x1, y1, x2, y2, confidence, class]
    
    output = {}
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        cls_name = model.names[int(cls)]
        cropped_img = img_cv2[int(y1):int(y2), int(x1):int(x2)]

        # OCR auf den Bereich anwenden
        ocr_result = reader.readtext(cropped_img)
        text = ocr_result[0][1] if ocr_result else "Unleserlich"

        output[cls_name] = text

    return output

if __name__ == "__main__":
    image_path = "test_meter.jpg"  # Pfad zu deinem Testbild
    result = process_image(image_path)
    print("Erkannt:", result)
