from kedro.pipeline import Pipeline, node
from .nodes import (
    load_gtsrb_data,
    load_pretrained_yolo,
    preprocess_images_for_detection,
    detect_road_signs,
    extract_roi_for_ocr
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=load_gtsrb_data,
            inputs=["gtsrb_data_config", "parameters"],
            outputs="gtsrb_data",
            name="load_gtsrb_data_node"
        ),
        node(
            func=load_pretrained_yolo,
            inputs="parameters",
            outputs="yolo_model",
            name="load_pretrained_yolo_node"
        ),
        node(
            func=preprocess_images_for_detection,
            inputs=["gtsrb_data", "parameters"],
            outputs="processed_images",
            name="preprocess_images_node"
        ),
        node(
            func=detect_road_signs,
            inputs=["yolo_model", "processed_images", "parameters"],
            outputs="detection_results",
            name="detect_road_signs_node"
        ),
        node(
            func=extract_roi_for_ocr,
            inputs=["detection_results", "parameters"],
            outputs="extracted_rois",
            name="extract_roi_node"
        )
    ])