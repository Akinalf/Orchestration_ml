
from kedro.pipeline import Pipeline, node
from .nodes import process_road_sign_ocr

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=process_road_sign_ocr,
            inputs=["chauzon_image", "montepellier_image", "poil_image"],
            outputs="ocr_results",
            name="ocr_node"
        )
    ])