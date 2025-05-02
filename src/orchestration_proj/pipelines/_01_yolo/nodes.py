from ultralytics import YOLO
import os
def train_yolo_model(yolo_data_yaml_path: str) -> YOLO:

    print(os.getcwd())
    model = YOLO("yolov8n.pt")  # Charge un modèle pré-entraîné
    model.train(data=yolo_data_yaml_path, epochs=20, imgsz=640)
    return model
