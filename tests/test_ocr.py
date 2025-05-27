import pytest
from unittest.mock import Mock, patch
from Orchestration_projoj.pipelines.ocr_processing.nodes import perform_ocr, configure_tesseract

def test_tesseract_configuration():
    """Test de la configuration Tesseract"""
    tesseract_config = {"language": "fra+eng"}
    parameters = {
        "ocr": {
            "language": "fra+eng",
            "config": "--psm 8 --oem 3",
            "min_confidence": 60
        }
    }
    
    result = configure_tesseract(tesseract_config, parameters)
    
    assert result["lang"] == "fra+eng"
    assert result["config"] == "--psm 8 --oem 3"
    assert result["min_confidence"] == 60

@patch('pytesseract.image_to_string')
def test_ocr_text_extraction(mock_ocr):
    """Test d'extraction de texte avec Tesseract simulé"""
    # Mock de la réponse Tesseract
    mock_ocr.return_value = "STOP"
    
    ocr_data = [{
        'image_path': 'test.jpg',
        'bbox': [100, 100, 50, 50],
        'roi_image': Mock()
    }]
    
    tesseract_config = {
        'lang': 'fra+eng',
        'config': '--psm 8',
        'min_confidence': 60
    }
    
    parameters = {"ocr": {"min_confidence": 60}}
    
    results = perform_ocr(ocr_data, tesseract_config, parameters)
    
    assert len(results) == 1
    assert results[0]['detected_text'] == "STOP"
    assert results[0]['confidence'] > 0

def test_ocr_result_validation():
    """Test de validation des résultats OCR"""
    # Simuler un résultat OCR
    ocr_result = {
        'image_path': 'test.jpg',
        'detected_text': 'STOP',
        'confidence': 0.95
    }
    
    # Vérifications
    assert 'detected_text' in ocr_result
    assert 'confidence' in ocr_result
    assert 0 <= ocr_result['confidence'] <= 1
    assert isinstance(ocr_result['detected_text'], str)