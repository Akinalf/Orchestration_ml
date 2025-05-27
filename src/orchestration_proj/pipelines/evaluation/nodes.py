import logging
from typing import Dict, List, Any
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)

def evaluate_detection_performance(
    detection_results: List[Dict[str, Any]],
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Évalue les performances de détection
    """
    logger.info("Évaluation des performances de détection")
    
    total_images = len(detection_results)
    successful_detections = len([d for d in detection_results if d["bbox"] is not None])
    
    # Calcul des métriques de détection
    detection_rate = successful_detections / total_images if total_images > 0 else 0
    
    confidences = [d["confidence"] for d in detection_results if d["confidence"] > 0]
    avg_confidence = np.mean(confidences) if confidences else 0
    
    metrics = {
        "total_images": total_images,
        "successful_detections": successful_detections,
        "detection_rate": float(detection_rate),
        "average_confidence": float(avg_confidence),
        "confidence_threshold_used": parameters["yolo"]["confidence_threshold"]
    }
    
    logger.info(f"Détection: {detection_rate:.2%} de réussite, confiance moyenne {avg_confidence:.3f}")
    return metrics

def create_final_report(
    detection_metrics: Dict[str, Any],
    ocr_evaluation: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Crée le rapport final d'évaluation
    """
    logger.info("Création du rapport final")
    
    # Calcul du pipeline end-to-end
    detection_rate = detection_metrics["detection_rate"]
    ocr_success_rate = ocr_evaluation["success_rate"]
    end_to_end_success = detection_rate * ocr_success_rate
    
    final_report = {
        "pipeline_summary": {
            "total_images_processed": detection_metrics["total_images"],
            "detection_success_rate": detection_rate,
            "ocr_success_rate": ocr_success_rate,
            "end_to_end_success_rate": float(end_to_end_success)
        },
        "detection_performance": detection_metrics,
        "ocr_performance": ocr_evaluation,
        "configuration_used": {
            "yolo_confidence_threshold": parameters["yolo"]["confidence_threshold"],
            "yolo_iou_threshold": parameters["yolo"]["iou_threshold"],
            "ocr_min_confidence": parameters["ocr"]["min_confidence"],
            "ocr_language": parameters["ocr"]["language"]
        },
        "recommendations": []
    }
    
    # Ajouter des recommandations basées sur les performances
    if detection_rate < 0.7:
        final_report["recommendations"].append(
            "Considérer réduire le seuil de confiance YOLO pour améliorer le taux de détection"
        )
    
    if ocr_success_rate < 0.5:
        final_report["recommendations"].append(
            "Améliorer le preprocessing OCR ou ajuster la configuration Tesseract"
        )
    
    if end_to_end_success > 0.8:
        final_report["recommendations"].append(
            "Excellentes performances ! Pipeline prêt pour la production"
        )
    
    logger.info(f"Rapport final: Succès end-to-end {end_to_end_success:.2%}")
    return final_report