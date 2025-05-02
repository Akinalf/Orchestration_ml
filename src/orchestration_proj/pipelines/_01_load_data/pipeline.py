from kedro.pipeline import Pipeline, node
from .nodes import download_data,extract_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=download_data,
            inputs=None,
            outputs="path",
            name="download_data.train_model"
        ),
        node(
            func=extract_data,
            inputs="zip_path",
            outputs="model_path",
            name="model_training.save_model"
        )   
    ])
