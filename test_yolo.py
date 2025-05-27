from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

model_path = "data/06_models/model_yolo.pt"
model = YOLO(model_path)

# Ton image de test
test_image = "data/01_raw/gtsrb/test.png"

image = cv2.imread(test_image)

print(f"🖼️ Image: {test_image}")
print(f"📏 Taille: {image.shape}")

# Test avec des seuils TRÈS bas
seuils = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]

print(f"\n🔍 TEST AVEC SEUILS TRÈS BAS:")
print("=" * 50)

best_results = []

for seuil in seuils:
    print(f"\n🎯 Seuil {seuil}:")
    
    # Test avec verbose pour voir ce qui se passe
    results = model(image, conf=seuil, iou=0.1, verbose=False, save=False)
    
    total_detections = 0
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            total_detections = len(result.boxes)
            print(f"  ✅ {total_detections} détection(s) trouvée(s)")
            
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                bbox = box.xyxy[0].cpu().numpy()
                
                print(f"    {i+1}. {cls_name}")
                print(f"       Confiance: {conf:.4f}")
                print(f"       BBox: [{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]")
                
                best_results.append({
                    'seuil': seuil,
                    'classe': cls_name,
                    'confiance': conf,
                    'bbox': bbox
                })
        else:
            print(f"  ❌ Aucune détection")

if not best_results:
    print(f"\n❌ AUCUNE DÉTECTION MÊME AVEC SEUIL 0.001 !")
    print(f"\n🔧 TESTS SUPPLÉMENTAIRES:")
    
    # Test 1: Voir toutes les prédictions brutes (même très faibles)
    print(f"\n1️⃣ Test avec seuil 0.0001 (quasi toutes les prédictions):")
    results = model(image, conf=0.0001, verbose=True, save=False)
    
    # Test 2: Essayer différentes tailles d'images
    print(f"\n2️⃣ Test avec différentes tailles:")
    sizes = [320, 480, 640, 800, 1024]
    
    for size in sizes:
        print(f"  Taille {size}x{size}:")
        results = model(image, conf=0.01, imgsz=size, verbose=False)
        
        count = 0
        for result in results:
            if result.boxes is not None:
                count = len(result.boxes)
        print(f"    → {count} détection(s)")
    
    # Test 3: Préprocessing différent
    print(f"\n3️⃣ Test avec amélioration de l'image:")
    
    # Améliorer le contraste
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    results = model(enhanced, conf=0.01, verbose=False)
    count = 0
    for result in results:
        if result.boxes is not None:
            count = len(result.boxes)
    print(f"    Image améliorée → {count} détection(s)")
    
    # Test 4: Comparer avec YOLOv8n standard
    print(f"\n4️⃣ Test avec YOLOv8n standard (pour comparaison):")
    try:
        standard_model = YOLO('yolov8n.pt')
        results = standard_model(image, conf=0.3, verbose=False)
        
        count = 0
        for result in results:
            if result.boxes is not None:
                count = len(result.boxes)
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = standard_model.names[cls_id]
                    print(f"    YOLOv8n détecte: {cls_name} (conf: {conf:.3f})")
        
        if count == 0:
            print(f"    YOLOv8n standard → Aucune détection non plus")
            
    except Exception as e:
        print(f"    Erreur YOLOv8n: {e}")

else:
    print(f"\n🎉 MEILLEURES DÉTECTIONS:")
    for result in sorted(best_results, key=lambda x: x['confiance'], reverse=True)[:5]:
        print(f"  • {result['classe']} (conf: {result['confiance']:.4f}, seuil: {result['seuil']})")

print(f"\n💡 ANALYSE:")
print(f"1. Ton modèle connaît 'construction (danger)' qui correspond à ton panneau")
print(f"2. L'image a une bonne taille (287x363)")
print(f"3. Si aucune détection même à 0.001, il y a un problème fondamental")
print(f"\n🔧 SOLUTIONS POSSIBLES:")
print(f"1. Le modèle Kaggle est peut-être corrompu ou incompatible")
print(f"2. Essayer un autre modèle YOLOv8 spécialisé panneaux")
print(f"3. Vérifier si le modèle attend un preprocessing spécifique")

