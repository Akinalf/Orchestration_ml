"""Tests pour __main__.py"""
import pytest
from unittest.mock import patch, Mock
from pathlib import Path

def test_main_module_imports():
    """Test que le module main peut être importé"""
    try:
        import src.orchestration_proj.__main__ as main_module
        assert hasattr(main_module, 'main')
    except SystemExit:
        # Normal si le script essaie de s'exécuter
        assert True

def test_main_function_exists():
    """Test que la fonction main existe"""
    import src.orchestration_proj.__main__ as main_module
    assert callable(getattr(main_module, 'main', None))

@patch('src.orchestration_proj.__main__.find_run_command')
@patch('src.orchestration_proj.__main__.configure_project')
def test_main_function_calls(mock_configure, mock_find_run):
    """Test que la fonction main fait les bons appels"""
    # Mock du run command
    mock_run = Mock(return_value="success")
    mock_find_run.return_value = mock_run
    
    import src.orchestration_proj.__main__ as main_module
    
    # Appeler main avec des args
    result = main_module.main("test", "args")
    
    # Vérifier que configure_project a été appelé
    mock_configure.assert_called_once()
    
    # Vérifier que find_run_command a été appelé
    mock_find_run.assert_called_once()
    
    # Vérifier que le run command a été appelé
    mock_run.assert_called_once_with("test", "args", standalone_mode=True)

def test_main_package_name_detection():
    """Test que le nom du package est correctement détecté"""
    import src.orchestration_proj.__main__ as main_module
    
    # Le package name devrait être 'orchestration_proj' 
    # (basé sur Path(__file__).parent.name)
    file_path = Path(main_module.__file__)
    package_name = file_path.parent.name
    assert package_name == "orchestration_proj"

@patch('sys.ps1', create=True)
@patch('src.orchestration_proj.__main__.find_run_command')
@patch('src.orchestration_proj.__main__.configure_project')
def test_main_interactive_mode(mock_configure, mock_find_run, mock_ps1):
    """Test du mode interactif"""
    mock_run = Mock()
    mock_find_run.return_value = mock_run
    
    import src.orchestration_proj.__main__ as main_module
    
    # Appeler main en mode interactif
    main_module.main()
    
    # En mode interactif, standalone_mode devrait être False
    mock_run.assert_called_once_with(standalone_mode=False)

def test_main_file_structure():
    """Test de la structure du fichier main"""
    import src.orchestration_proj.__main__ as main_module
    
    # Vérifier que le module a les imports nécessaires
    assert hasattr(main_module, 'Path')
    assert hasattr(main_module, 'find_run_command')
    assert hasattr(main_module, 'configure_project')

@patch('src.orchestration_proj.__main__.find_run_command')
@patch('src.orchestration_proj.__main__.configure_project')  
def test_main_with_kwargs(mock_configure, mock_find_run):
    """Test de main avec des kwargs"""
    mock_run = Mock()
    mock_find_run.return_value = mock_run
    
    import src.orchestration_proj.__main__ as main_module
    
    # Appeler avec kwargs
    main_module.main(test_arg="value", another_arg=123)
    
    # Vérifier l'appel
    mock_run.assert_called_once_with(test_arg="value", another_arg=123, standalone_mode=True)