#!/usr/bin/env python3
"""
Test OCR avec ROI définies manuellement pour panneaux routiers
"""
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image
import re
import os

def define_rois():
    """
    Définit les ROI manuellement pour chaque image
    Format: (x, y, width, height) calculé depuis coin haut gauche et coin bas droit
    """
    # ROI définies avec tes coordonnées exactes
    rois = {
        "chauzon.jpg": (18, 55, 260, 70),    # chg(18,59) cbd(280,131) → w=262, h=72
        "montpellier.jpg": (59, 74, 145, 30), # chg(50,65) cbd(220,110) → w=170, h=45
        "poil.jpg": (4, 111, 150, 50)        # chg(4,111) cbd(166,166) → w=162, h=55
    }
    return rois

def load_test_sign_images_with_roi(images_path="data/01_raw/"):
    """Charge les images avec leurs ROI définies"""
    print("\n🖼️ Chargement des images avec ROI")
    print("=" * 50)
    
    extensions = ('.jpg', '.jpeg', '.png')
    rois = define_rois()
    test_images = []

    if not os.path.exists(images_path):
        print(f"❌ Dossier introuvable: {images_path}")
        return []

    fichiers = [f for f in os.listdir(images_path) if f.lower().endswith(extensions)]
    
    for fichier in fichiers:
        chemin_complet = os.path.join(images_path, fichier)
        try:
            img_pil = Image.open(chemin_complet).convert("RGB")
            img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            expected_text = fichier.split('.')[0]
            
            # Récupérer la ROI pour cette image
            roi = rois.get(fichier, None)
            
            test_images.append((fichier, img_np, expected_text, roi))
            
            if roi:
                print(f"✓ {fichier} - Taille: {img_pil.size} - Attendu: '{expected_text}' - ROI: {roi}")
            else:
                print(f"⚠️ {fichier} - Pas de ROI définie, utilisation image complète")
                
        except Exception as e:
            print(f"✗ Erreur avec {fichier}: {e}")

    print(f"\n✅ {len(test_images)} image(s) chargée(s)")
    return test_images

def extract_roi(image, roi):
    """Extrait la ROI de l'image"""
    if roi is None:
        return image
    
    x, y, w, h = roi
    return image[y:y+h, x:x+w]

def show_roi_extraction(img_name, image, roi, expected_text):
    """Affiche l'image originale et la ROI extraite"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{img_name} - ROI Extraction (attendu: {expected_text})')
    
    # Image originale
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title('Image originale')
    axes[0].axis('off')
    
    # ROI sur l'image originale
    img_with_roi = img_rgb.copy()
    if roi:
        x, y, w, h = roi
        cv2.rectangle(img_with_roi, (x, y), (x+w, y+h), (255, 0, 0), 2)
    axes[1].imshow(img_with_roi)
    axes[1].set_title('ROI marquée')
    axes[1].axis('off')
    
    # ROI extraite
    roi_img = extract_roi(image, roi)
    roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    axes[2].imshow(roi_rgb)
    axes[2].set_title('ROI extraite')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def preprocess_roi(roi_image):
    """Preprocessing spécifique pour les ROI de panneaux"""
    # Conversion en niveaux de gris
    if len(roi_image.shape) == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_image.copy()
    
    # Redimensionner si trop petit (important pour l'OCR)
    height, width = gray.shape
    if max(height, width) < 100:
        scale = 100 / max(height, width)
        new_width, new_height = int(width * scale), int(height * scale)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        print(f"    Redimensionnement ROI: {width}x{height} → {new_width}x{new_height}")
    
    # Améliorer le contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    enhanced = clahe.apply(gray)
    
    # Seuillage pour avoir du texte noir sur fond blanc
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Inverser si nécessaire (texte doit être noir sur fond blanc)
    if np.mean(binary) > 127:  # Si plus de blanc que de noir, inverser
        binary = cv2.bitwise_not(binary)
    
    return binary

def test_ocr_on_roi(img_name, image, expected_text, roi, show_debug=True):
    """Test OCR sur la ROI extraite avec configuration par défaut uniquement"""
    print(f"\n🎯 Test OCR sur ROI: {img_name} (attendu: '{expected_text}')")
    
    if show_debug:
        show_roi_extraction(img_name, image, roi, expected_text)
    
    # Extraire la ROI
    roi_img = extract_roi(image, roi)
    print(f"  Taille ROI: {roi_img.shape[:2] if roi else 'Image complète'}")
    
    # Preprocessing de la ROI
    processed_roi = preprocess_roi(roi_img)
    
    # Afficher la ROI preprocessée
    if show_debug:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        plt.title('ROI originale')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(processed_roi, cmap='gray')
        plt.title('ROI preprocessée')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    try:
        # OCR avec configuration par défaut uniquement
        text = pytesseract.image_to_string(processed_roi).strip()
        data = pytesseract.image_to_data(processed_roi, output_type=pytesseract.Output.DICT)
        
        # Calculer confiance
        confidences = [int(c) for c in data['conf'] if int(c) > 0]
        avg_conf = np.mean(confidences) if confidences else 0
        
        # Nettoyer le texte
        clean_text = re.sub(r'[^A-Za-z0-9]', '', text.lower())
        expected_clean = re.sub(r'[^A-Za-z0-9]', '', expected_text.lower())
        
        is_correct = clean_text == expected_clean
        
        result_str = f"    Default → '{clean_text}' (conf: {avg_conf:.1f})"
        if is_correct:
            result_str += " ✅"
        else:
            result_str += " ❌"
        
        print(result_str)
        
        return {
            'image': img_name,
            'expected': expected_text,
            'detected': clean_text,
            'confidence': avg_conf,
            'correct': is_correct
        }
        
    except Exception as e:
        print(f"    ERREUR: {e}")
        return {
            'image': img_name,
            'expected': expected_text,
            'detected': '',
            'confidence': 0,
            'correct': False
        }







def main():
    """Test principal avec ROI manuelles"""
    print("🚀 TEST OCR AVEC ROI MANUELLES")
    print("=" * 50)
    
    # Vérifier Tesseract
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
    except:
        print("❌ Tesseract non trouvé!")
        return
    
    # Charger les images avec ROI
    test_images = load_test_sign_images_with_roi()
    
    if not test_images:
        print("❌ Aucune image à tester")
        return
    
    # Afficher les ROI définies
    print(f"\n📍 ROI DÉFINIES:")
    rois = define_rois()
    for filename, roi in rois.items():
        x, y, w, h = roi
        print(f"  {filename}: chg({x},{y}) cbd({x+w},{y+h}) → taille {w}x{h}")
    
    print(f"\n🧪 TESTS OCR AVEC CONFIG PAR DÉFAUT")
    print("=" * 40)
    
    results = []
    for img_name, img_np, expected_text, roi in test_images:
        result = test_ocr_on_roi(img_name, img_np, expected_text, roi, show_debug=True)
        results.append(result)
    
    # Résumé final
    print(f"\n📈 RÉSUMÉ FINAL")
    print("=" * 30)
    successful = [r for r in results if r['correct']]
    total = len(results)
    
    print(f"Images testées: {total}")
    print(f"Succès: {len(successful)} ({len(successful)/total*100:.1f}%)")
    
    if successful:
        print(f"\n✅ DÉTECTIONS RÉUSSIES:")
        for result in successful:
            print(f"  {result['image']}: '{result['detected']}' (conf: {result['confidence']:.1f})")
    
    failed = [r for r in results if not r['correct']]
    if failed:
        print(f"\n❌ ÉCHECS:")
        for result in failed:
            print(f"  {result['image']}: attendu '{result['expected']}', détecté '{result['detected']}'")
    
    print(f"\n💡 ROI utilisées sont maintenant précises selon tes coordonnées !")

if __name__ == "__main__":
    main()