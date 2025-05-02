import kagglehub
import zipfile
import os
from typing import Optional

def download_data() -> str:
    """Download the GTSRB dataset from Kaggle and return path to ZIP file."""
    # Télécharge et retourne le chemin du fichier ZIP
    path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
    
    return path


def extract_data(zip_path: str) -> str:
    """Extracts the ZIP file and returns the path to the extracted directory."""
    extract_dir = os.path.splitext(zip_path)[0]  # same name as ZIP without extension
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir
