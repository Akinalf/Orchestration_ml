from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_yolo_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_yolo_model,
            inputs="yolo_data_yaml_path", 
            outputs="yolo_model",
            name="train_yolo_model_node"
        ),
    ])
