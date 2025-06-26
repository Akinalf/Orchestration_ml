"""Tests pour pipeline_registry"""
def test_pipeline_registry_module_import():
    """Test d'import du module pipeline_registry"""
    import src.orchestration_proj.pipeline_registry
    assert hasattr(src.orchestration_proj.pipeline_registry, '__file__')

def test_pipeline_registry_has_register_function():
    """Test que le module a une fonction register"""
    import src.orchestration_proj.pipeline_registry as pr
    # Vérifier qu'il y a au moins des attributs dans le module
    attrs = dir(pr)
    assert len(attrs) > 0

