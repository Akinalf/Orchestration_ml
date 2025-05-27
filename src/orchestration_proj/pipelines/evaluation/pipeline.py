from kedro.pipeline import Pipeline, node
from .nodes import (
    evaluate_detection_performance,
    create_final_report
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=evaluate_detection_performance,
            inputs=["detection_results", "parameters"],
            outputs="detection_metrics",
            name="evaluate_detection_node"
        ),
        node(
            func=create_final_report,
            inputs=["detection_metrics", "ocr_evaluation", "parameters"],
            outputs="final_report",
            name="create_report_node"
        )
    ])