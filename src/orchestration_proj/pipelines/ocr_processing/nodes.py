"""
Node OCR simple pour Kedro
"""
import cv2
import numpy as np
import pytesseract
import pandas as pd
from PIL import Image
import re

def process_road_sign_ocr(chauzon_image, montepellier_image, poil_image) -> pd.DataFrame:
    """
    OCR sur les 3 images de panneaux
    """
    # ROI définies manuellement
    # En production: remplacer par les sorties YOLO
    rois = {
        "chauzon.jpg": (18, 59, 262, 72),
        "montepellier.jpg": (50, 65, 170, 45), 
        "poil.jpg": (4, 111, 162, 55)
    }
    
    # Images factices
    images = {
        "chauzon.jpg": np.array(chauzon_image),
        "montepellier.jpg": np.array(montepellier_image),
        "poil.jpg": np.array(poil_image)
    }
    
    results = []
    
    for img_name, img in images.items():
        # Convertir PIL en OpenCV si nécessaire
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Extraire ROI
        x, y, w, h = rois[img_name]
        roi = img[y:y+h, x:x+w]
        
        # Preprocessing simple
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR
        text = pytesseract.image_to_string(binary).strip()
        clean_text = re.sub(r'[^A-Za-z0-9]', '', text.lower())
        
        # Résultat
        expected = img_name.split('.')[0]
        results.append({
            'image': img_name,
            'expected': expected,
            'detected': clean_text,
            'correct': clean_text == expected
        })
    
    return pd.DataFrame(results)