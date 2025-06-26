"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline
from .pipelines import yolo_detection, ocr_processing


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""
    
    yolo_pipeline = yolo_detection.create_pipeline()
    ocr_pipeline = ocr_processing.create_pipeline()
    
    return {
        "__default__": yolo_pipeline + ocr_pipeline,
        "yolo": yolo_pipeline,
        "ocr": ocr_pipeline,
        "detection_only": yolo_pipeline,
        "ocr_only": ocr_pipeline,
    }