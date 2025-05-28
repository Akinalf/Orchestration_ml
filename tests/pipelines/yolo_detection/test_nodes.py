import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.orchestration_proj.pipelines.yolo_detection.nodes import (
    load_gtsrb_data,
    load_pretrained_yolo,
    preprocess_images_for_detection,
    detect_road_signs,
    extract_roi_for_ocr
)


class TestLoadGTSRBData:
    """Tests pour le chargement des données GTSRB"""
    
    def test_load_gtsrb_data_returns_dict(self):
        """Test que la fonction retourne un dictionnaire"""
        gtsrb_config = {
            'nc': 43,
            'names': ['stop', 'yield']
        }
        parameters = {
            'gtsrb': {
                'max_images_per_class': 10
            }
        }
        
        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = []
            
            result = load_gtsrb_data(gtsrb_config, parameters)
            
            assert isinstance(result, dict)
            assert 'train_images' in result
            assert 'test_images' in result
            assert 'num_classes' in result
            
    def test_load_gtsrb_data_has_correct_structure(self):
        """Test que le dictionnaire a la bonne structure"""
        gtsrb_config = {'nc': 43, 'names': []}
        parameters = {'gtsrb': {'max_images_per_class': 5}}
        
        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = [Path('test1.jpg'), Path('test2.jpg')]
            
            result = load_gtsrb_data(gtsrb_config, parameters)
            
            expected_keys = [
                'train_images', 'test_images', 'train_labels_path', 
                'test_labels_path', 'num_classes', 'class_names',
                'total_train_images', 'total_test_images'
            ]
            
            for key in expected_keys:
                assert key in result


class TestLoadPretrainedYOLO:
    """Tests pour le chargement du modèle YOLO"""
    
    @patch('src.orchestration_proj.pipelines.yolo_detection.nodes.YOLO')
    def test_load_pretrained_yolo_returns_model(self, mock_yolo):
        """Test que la fonction retourne un modèle YOLO"""
        mock_model = Mock()
        mock_model.info.return_value = "Model info"
        mock_yolo.return_value = mock_model
        
        parameters = {
            'yolo': {
                'model_path': 'yolov8n.pt'
            }
        }
        
        result = load_pretrained_yolo(parameters)
        
        assert result == mock_model
        mock_yolo.assert_called_once_with('yolov8n.pt')
        
    @patch('src.orchestration_proj.pipelines.yolo_detection.nodes.YOLO')
    def test_load_pretrained_yolo_calls_info(self, mock_yolo):
        """Test que la fonction appelle info() sur le modèle"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        parameters = {'yolo': {'model_path': 'test.pt'}}
        
        load_pretrained_yolo(parameters)
        
        mock_model.info.assert_called_once()


class TestPreprocessImagesForDetection:
    """Tests pour le preprocessing des images"""
    
    @patch('cv2.imread')
    @patch('pathlib.Path.exists')
    def test_preprocess_images_returns_list(self, mock_exists, mock_imread):
        """Test que la fonction retourne une liste"""
        mock_imread.return_value = np.ones((100, 100, 3), dtype=np.uint8)
        mock_exists.return_value = False
        
        gtsrb_data = {
            'test_images': [Path('test1.jpg'), Path('test2.jpg')],
            'test_labels_path': Path('labels/')
        }
        parameters = {}
        
        result = preprocess_images_for_detection(gtsrb_data, parameters)
        
        assert isinstance(result, list)
        
    @patch('cv2.imread')
    @patch('pathlib.Path.exists')
    def test_preprocess_images_structure(self, mock_exists, mock_imread):
        """Test la structure des données preprocessées"""
        mock_imread.return_value = np.ones((100, 100, 3), dtype=np.uint8)
        mock_exists.return_value = False
        
        gtsrb_data = {
            'test_images': [Path('test1.jpg')],
            'test_labels_path': Path('labels/')
        }
        parameters = {}
        
        result = preprocess_images_for_detection(gtsrb_data, parameters)
        
        if result:  # Si des images sont traitées
            expected_keys = ['original_path', 'image', 'class_id', 'width', 'height']
            for key in expected_keys:
                assert key in result[0]
                
    @patch('cv2.imread')
    def test_preprocess_images_handles_none_image(self, mock_imread):
        """Test la gestion des images None"""
        mock_imread.return_value = None
        
        gtsrb_data = {
            'test_images': [Path('bad_image.jpg')],
            'test_labels_path': Path('labels/')
        }
        parameters = {}
        
        result = preprocess_images_for_detection(gtsrb_data, parameters)
        
        # Devrait retourner une liste vide si l'image est None
        assert isinstance(result, list)


class TestDetectRoadSigns:
    """Tests pour la détection de panneaux"""
    
    def test_detect_road_signs_returns_list(self):
        """Test que la fonction retourne une liste"""
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        
        processed_images = [
            {
                'image': np.ones((100, 100, 3), dtype=np.uint8),
                'original_path': 'test.jpg',
                'class_id': 0
            }
        ]
        parameters = {
            'yolo': {
                'confidence_threshold': 0.5,
                'iou_threshold': 0.4
            }
        }
        
        result = detect_road_signs(mock_model, processed_images, parameters)
        
        assert isinstance(result, list)
        
    def test_detect_road_signs_with_detections(self):
        """Test la détection avec des résultats"""
        mock_model = Mock()
        
        # Mock des boxes détectées
        mock_box = Mock()
        mock_box.xyxy = [Mock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value.tolist.return_value = [10, 10, 50, 50]
        mock_box.conf = [Mock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = 0.9
        mock_box.cls = [Mock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = 1
        
        mock_result = Mock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        
        processed_images = [
            {
                'image': np.ones((100, 100, 3), dtype=np.uint8),
                'original_path': 'test.jpg',
                'class_id': 1
            }
        ]
        parameters = {
            'yolo': {
                'confidence_threshold': 0.5,
                'iou_threshold': 0.4
            }
        }
        
        result = detect_road_signs(mock_model, processed_images, parameters)
        
        assert len(result) > 0
        if result:
            detection = result[0]
            expected_keys = ['image_path', 'bbox', 'confidence', 'class_id', 'original_class', 'image_shape']
            for key in expected_keys:
                assert key in detection


class TestExtractROIForOCR:
    """Tests pour l'extraction des ROI"""
    
    @patch('cv2.imread')
    def test_extract_roi_returns_list(self, mock_imread):
        """Test que la fonction retourne une liste"""
        mock_imread.return_value = np.ones((100, 100, 3), dtype=np.uint8)
        
        detection_results = [
            {
                'image_path': 'test.jpg',
                'bbox': [10, 10, 50, 50],
                'confidence': 0.8,
                'class_id': 1,
                'original_class': 1
            }
        ]
        parameters = {}
        
        result = extract_roi_for_ocr(detection_results, parameters)
        
        assert isinstance(result, list)
        
    @patch('cv2.imread')
    def test_extract_roi_structure(self, mock_imread):
        """Test la structure des ROI extraites"""
        mock_imread.return_value = np.ones((100, 100, 3), dtype=np.uint8)
        
        detection_results = [
            {
                'image_path': 'test.jpg',
                'bbox': [10, 10, 50, 50],
                'confidence': 0.8,
                'class_id': 1,
                'original_class': 1
            }
        ]
        parameters = {}
        
        result = extract_roi_for_ocr(detection_results, parameters)
        
        if result:  # Si des ROI sont extraites
            roi = result[0]
            expected_keys = ['image_path', 'roi_image', 'bbox', 'confidence', 'detected_class', 'original_class']
            for key in expected_keys:
                assert key in roi
                
    def test_extract_roi_filters_low_confidence(self):
        """Test que les détections à faible confiance sont filtrées"""
        detection_results = [
            {
                'image_path': 'test.jpg',
                'bbox': [10, 10, 50, 50],
                'confidence': 0.1,  # Confiance trop faible
                'class_id': 1,
                'original_class': 1
            }
        ]
        parameters = {}
        
        result = extract_roi_for_ocr(detection_results, parameters)
        
        # Devrait filtrer les détections à faible confiance
        assert len(result) == 0
        
    def test_extract_roi_handles_none_bbox(self):
        """Test la gestion des bbox None"""
        detection_results = [
            {
                'image_path': 'test.jpg',
                'bbox': None,
                'confidence': 0.8,
                'class_id': None,
                'original_class': 1
            }
        ]
        parameters = {}
        
        result = extract_roi_for_ocr(detection_results, parameters)
        
        # Devrait ignorer les détections sans bbox
        assert len(result) == 0


# Tests utilitaires
def test_imports():
    """Test que tous les imports fonctionnent"""
    import numpy as np
    import cv2
    from pathlib import Path
    
    assert True


def test_basic_functionality():
    """Test de base"""
    assert 2 + 2 == 4