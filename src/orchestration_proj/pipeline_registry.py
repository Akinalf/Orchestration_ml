
from kedro.pipeline import Pipeline
from orchestration_proj.pipelines import (
    _01_load_data as load_data,
    _02_yolo as yolo,
    _03_OCR as OCR
)

def register_pipelines() -> dict:
    """Register the project's pipelines.

    Returns:
        A dictionary mapping pipeline names to their corresponding Pipeline objects.
    """
    
    load_data_pipeline = load_data.create_pipeline()
    yolo_pipeline = yolo.create_pipeline()
    ocr_pipeline = OCR.create_pipeline()

    
    return {
        "__default__": load_data_pipeline + yolo_pipeline + ocr_pipeline,
        "load_data": load_data_pipeline,
        "yolo": yolo_pipeline,
        "ocr": ocr_pipeline,
    }