"""Tests pour pipeline_registry"""
import pytest
from src.orchestration_proj.pipeline_registry import register_pipelines

def test_register_pipelines():
    """Test que register_pipelines retourne un dict"""
    result = register_pipelines()
    assert isinstance(result, dict)
    
def test_register_pipelines_has_default():
    """Test que la pipeline par défaut existe"""
    result = register_pipelines()
    assert "__default__" in result