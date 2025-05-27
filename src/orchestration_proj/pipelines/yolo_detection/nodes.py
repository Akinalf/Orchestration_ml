import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from PIL import Image
import torch

logger = logging.getLogger(__name__)

def load_gtsrb_data(
    gtsrb_data_config: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Charge et organise les données GTSRB formatées YOLO
    """
    gtsrb_params = parameters["gtsrb"]
    
    # Chemins des images et labels
    train_images_path = Path("data/01_raw/gtsrb/images/train/")
    test_images_path = Path("data/01_raw/gtsrb/images/test/")
    train_labels_path = Path("data/01_raw/gtsrb/labels/train/")
    test_labels_path = Path("data/01_raw/gtsrb/labels/test/")
    
    # Lister les fichiers d'images
    train_images = list(train_images_path.glob("*.jpg")) + list(train_images_path.glob("*.png"))
    test_images = list(test_images_path.glob("*.jpg")) + list(test_images_path.glob("*.png"))
    
    # Limiter le nombre d'images pour les tests
    max_images = gtsrb_params.get("max_images_per_class", 100)
    train_images = train_images[:max_images]
    test_images = test_images[:max_images]
    
    data_info = {
        "train_images": train_images,
        "test_images": test_images,
        "train_labels_path": train_labels_path,
        "test_labels_path": test_labels_path,
        "num_classes": gtsrb_data_config.get('nc', 43),  # Classes depuis data.yaml
        "class_names": gtsrb_data_config.get('names', []),
        "total_train_images": len(train_images),
        "total_test_images": len(test_images)
    }
    
    logger.info(f"Données GTSRB chargées: {data_info['total_train_images']} train, {data_info['total_test_images']} test")
    return data_info

def load_pretrained_yolo(parameters: Dict[str, Any]) -> YOLO:
    """
    Charge le modèle YOLOv8 pré-entraîné
    """
    
    yolo_params = parameters["yolo"]
    model_path = yolo_params["model_path"]
    
    model = YOLO(model_path)

    logger.info(f"Modèle info: {model.info()}")
    
    return model

def preprocess_images_for_detection(
    gtsrb_data: Dict[str, Any],
    parameters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Préprocesse les images GTSRB pour la détection
    """
    logger.info("Preprocessing des images pour détection")
    
    processed_images = []
    
    # Utiliser les images de test
    test_images = gtsrb_data["test_images"]
    
    for image_path in test_images:
        try:
            # Charger l'image
            image = cv2.imread(str(image_path))
            if image is not None:
                # Charger le label correspondant si disponible
                label_path = gtsrb_data["test_labels_path"] / f"{image_path.stem}.txt"
                class_id = None
                
                if label_path.exists():
                    # Lire le fichier de label YOLO
                    with open(label_path, 'r') as f:
                        label_line = f.readline().strip()
                        if label_line:
                            class_id = int(label_line.split()[0])
                
                processed_img = {
                    "original_path": str(image_path),
                    "image": image,
                    "class_id": class_id,
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
                processed_images.append(processed_img)
                    
        except Exception as e:
            logger.warning(f"Erreur lors du traitement de {image_path}: {e}")
            continue
    
    logger.info(f"Images préprocessées: {len(processed_images)}")
    return processed_images



def detect_road_signs(
    model: YOLO,
    processed_images: List[Dict[str, Any]],
    parameters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Détecte les panneaux routiers avec YOLOv8
    """
    
    yolo_params = parameters["yolo"]
    confidence_threshold = yolo_params["confidence_threshold"]
    iou_threshold = yolo_params["iou_threshold"]
    
    detection_results = []
    
    for img_data in processed_images:
        try:
            image = img_data["image"]
            
            # Effectuer la détection
            results = model(
                image,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Extraire les résultats
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        detection = {
                            "image_path": img_data["original_path"],
                            "bbox": box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                            "confidence": float(box.conf[0].cpu().numpy()),
                            "class_id": int(box.cls[0].cpu().numpy()),
                            "original_class": img_data["class_id"],
                            "image_shape": image.shape
                        }
                        detection_results.append(detection)
                else:
                    # Aucune détection
                    detection = {
                        "image_path": img_data["original_path"],
                        "bbox": None,
                        "confidence": 0.0,
                        "class_id": None,
                        "original_class": img_data["class_id"],
                        "image_shape": image.shape
                    }
                    detection_results.append(detection)
                    
        except Exception as e:
            logger.error(f"Erreur détection pour {img_data['original_path']}: {e}")
            continue
    
    logger.info(f"Détections terminées: {len(detection_results)} résultats")
    return detection_results



def extract_roi_for_ocr(
    detection_results: List[Dict[str, Any]],
    parameters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Extrait les ROI des panneaux détectés pour l'OCR
    """
    logger.info("Extraction des ROI pour OCR")
    
    roi_data = []
    
    for detection in detection_results:
        if detection["bbox"] is not None and detection["confidence"] > 0.3:
            try:
                # Charger l'image originale
                image_path = detection["image_path"]
                image = cv2.imread(image_path)
                
                if image is not None:
                    # Extraire la ROI
                    x1, y1, x2, y2 = [int(coord) for coord in detection["bbox"]]
                    roi = image[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        roi_info = {
                            "image_path": image_path,
                            "roi_image": roi,
                            "bbox": detection["bbox"],
                            "confidence": detection["confidence"],
                            "detected_class": detection["class_id"],
                            "original_class": detection["original_class"]
                        }
                        roi_data.append(roi_info)
                        
            except Exception as e:
                logger.warning(f"Erreur extraction ROI pour {detection['image_path']}: {e}")
                continue
    
    logger.info(f"ROI extraites: {len(roi_data)}")
    return roi_data