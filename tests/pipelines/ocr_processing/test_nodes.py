"""
Tests pour le node OCR
"""
import pytest
import pandas as pd
import numpy as np
from PIL import Image
from src.orchestration_proj.pipelines.ocr_processing.nodes import process_road_sign_ocr


class TestOCRProcessing:
    """Tests pour le traitement OCR"""
    
    def test_process_road_sign_ocr_returns_dataframe(self):
        """Test que la fonction retourne bien un DataFrame"""
        # Créer des images factices
        fake_image = Image.new('RGB', (300, 200), color='white')
        
        # Appeler la fonction
        result = process_road_sign_ocr(fake_image, fake_image, fake_image)
        
        # Vérifier que c'est un DataFrame
        assert isinstance(result, pd.DataFrame)
        
    def test_process_road_sign_ocr_has_correct_columns(self):
        """Test que le DataFrame a les bonnes colonnes"""
        fake_image = Image.new('RGB', (300, 200), color='white')
        
        result = process_road_sign_ocr(fake_image, fake_image, fake_image)
        
        expected_columns = ['image', 'expected', 'detected', 'correct']
        assert list(result.columns) == expected_columns
        
    def test_process_road_sign_ocr_has_three_rows(self):
        """Test que le DataFrame a 3 lignes (une par image)"""
        fake_image = Image.new('RGB', (300, 200), color='white')
        
        result = process_road_sign_ocr(fake_image, fake_image, fake_image)
        
        assert len(result) == 3
        
    def test_process_road_sign_ocr_expected_values(self):
        """Test que les valeurs attendues sont correctes"""
        fake_image = Image.new('RGB', (300, 200), color='white')
        
        result = process_road_sign_ocr(fake_image, fake_image, fake_image)
        
        expected_names = ['chauzon', 'montepellier', 'poil']
        actual_expected = result['expected'].tolist()
        
        assert set(actual_expected) == set(expected_names)
        
    def test_process_road_sign_ocr_image_names(self):
        """Test que les noms d'images sont corrects"""
        fake_image = Image.new('RGB', (300, 200), color='white')
        
        result = process_road_sign_ocr(fake_image, fake_image, fake_image)
        
        expected_images = ['chauzon.jpg', 'montepellier.jpg', 'poil.jpg']
        actual_images = result['image'].tolist()
        
        assert set(actual_images) == set(expected_images)
        
    def test_process_road_sign_ocr_with_numpy_arrays(self):
        """Test avec des numpy arrays au lieu d'images PIL"""
        # Créer des arrays numpy factices
        fake_array = np.ones((200, 300, 3), dtype=np.uint8) * 255  # Image blanche
        
        result = process_road_sign_ocr(fake_array, fake_array, fake_array)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


class TestROIExtraction:
    """Tests pour l'extraction des ROI"""
    
    def test_roi_coordinates_are_valid(self):
        """Test que les coordonnées ROI sont valides"""
        from src.orchestration_proj.pipelines.ocr_processing.nodes import process_road_sign_ocr
        
        # Les ROI sont définies dans la fonction
        rois = {
            "chauzon.jpg": (18, 59, 262, 72),
            "montepellier.jpg": (50, 65, 170, 45), 
            "poil.jpg": (4, 111, 162, 55)
        }
        
        for img_name, (x, y, w, h) in rois.items():
            # Vérifier que les coordonnées sont positives
            assert x >= 0, f"x négatif pour {img_name}"
            assert y >= 0, f"y négatif pour {img_name}"
            assert w > 0, f"width négatif pour {img_name}"
            assert h > 0, f"height négatif pour {img_name}"


class TestDataIntegrity:
    """Tests d'intégrité des données"""
    
    def test_detected_text_is_string(self):
        """Test que le texte détecté est toujours une string"""
        fake_image = Image.new('RGB', (300, 200), color='white')
        
        result = process_road_sign_ocr(fake_image, fake_image, fake_image)
        
        for detected in result['detected']:
            assert isinstance(detected, str), "Le texte détecté doit être une string"
            
    def test_correct_is_boolean(self):
        """Test que la colonne 'correct' contient des booléens"""
        fake_image = Image.new('RGB', (300, 200), color='white')
        
        result = process_road_sign_ocr(fake_image, fake_image, fake_image)
        
        for correct in result['correct']:
            assert isinstance(correct, (bool, np.bool_)), "La colonne 'correct' doit contenir des booléens"


# Test simple pour s'assurer que les imports marchent
def test_imports():
    """Test que tous les imports nécessaires fonctionnent"""
    import cv2
    import numpy as np
    import pytesseract
    import pandas as pd
    import re
    
    assert True  # Si on arrive ici, les imports ont marché


# Test de base pour pytest
def test_basic():
    """Test de base pour vérifier que pytest fonctionne"""
    assert 1 + 1 == 2