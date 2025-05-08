#!/usr/bin/env python3
"""
Unit tests for the main application entry point (app/main.py)
"""
import sys
import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Mocking required environment variables
@pytest.fixture(scope="function")
def mock_environment():
    """Setup and teardown environment variables for testing"""
    orig_env = os.environ.copy()
    # Set minimal test environment
    os.environ["CASALINGUA_ENV"] = "test"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/models"
    os.environ["TORCH_HOME"] = "/tmp/.cache/torch"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(orig_env)

# Mock for FastAPI
class MockFastAPI:
    def __init__(self, **kwargs):
        self.title = kwargs.get('title', '')
        self.description = kwargs.get('description', '')
        self.version = kwargs.get('version', '')
        self.docs_url = kwargs.get('docs_url', '')
        self.redoc_url = kwargs.get('redoc_url', '')
        self.openapi_url = kwargs.get('openapi_url', '')
        self.lifespan = kwargs.get('lifespan', None)
        self.state = MagicMock()
        self.middleware_stack = []
        self.routers = []
    
    def add_middleware(self, middleware, **kwargs):
        self.middleware_stack.append((middleware, kwargs))
        return self
    
    def include_router(self, router, **kwargs):
        self.routers.append((router, kwargs))
        return self
    
    def mount(self, *args, **kwargs):
        return self
    
    def middleware(self, *args):
        def decorator(func):
            return func
        return decorator

# Mock for hardware detection
@pytest.fixture
def mock_hardware_detector():
    detector = MagicMock()
    hardware_info = {
        "memory": {"total_gb": 16, "available_gb": 8},
        "system": {"processor_type": "intel"},
        "cpu": {"count_physical": 4, "count_logical": 8, "brand": "Intel Core i7"},
        "gpu": {"has_gpu": False}
    }
    detector.detect_all.return_value = hardware_info
    detector.recommend_config.return_value = {"device": "cpu", "memory": {"model_size": "small", "batch_size": 4}}
    return detector

# Enhanced mocks for TokenizerPipeline
@pytest.fixture
def mock_tokenizer_pipeline():
    tokenizer = MagicMock()
    tokenizer.tokenize = MagicMock(return_value={"tokens": ["test", "tokens"]})
    return tokenizer

# Mocks for various dependencies
@pytest.fixture
def mock_dependencies():
    """Create mocks for all external dependencies"""
    with patch("app.main.print_startup_banner") as mock_banner, \
         patch("app.main.configure_logging") as mock_logging, \
         patch("app.main.load_config") as mock_config, \
         patch("app.main.EnhancedModelManager") as mock_manager, \
         patch("app.main.AuditLogger") as mock_audit, \
         patch("app.main.MetricsCollector") as mock_metrics, \
         patch("app.main.initialize_persistence") as mock_persistence, \
         patch("app.main.UnifiedProcessor") as mock_processor, \
         patch("app.main.ModelLoader") as mock_loader, \
         patch("app.main.TokenizerPipeline") as mock_tokenizer_pipeline, \
         patch("app.services.models.loader.ModelRegistry") as mock_registry, \
         patch("app.main.setup_console_logging") as mock_console_logging, \
         patch("app.main.console") as mock_console, \
         patch("app.main.init_terminal_colors") as mock_init_terminal_colors, \
         patch("app.main.Table") as mock_table, \
         patch("app.main.Panel") as mock_panel, \
         patch("app.main.Path") as mock_path, \
         patch("app.services.storage.session_manager.SessionManager") as mock_session_manager:
        
        # Configure mock returns
        mock_config.return_value = {
            "environment": "test",
            "log_level": "info",
            "enable_route_cache": False,
            "server_host": "localhost",
            "server_port": 8000,
            "models": {
                "models_dir": "models",
                "cache_dir": "cache/models"
            }
        }
        
        # Configure mock logger
        mock_app_logger = MagicMock()
        mock_logging.return_value = mock_app_logger
        
        # Configure mock session manager
        mock_session_instance = MagicMock()
        mock_session_manager.return_value = mock_session_instance
        mock_session_instance.start_cleanup_task = AsyncMock()
        mock_session_instance.cleanup = AsyncMock()
        
        # Configure mock table and panel
        mock_table.return_value = MagicMock()
        mock_panel.fit.return_value = MagicMock()
        
        # Configure mock path
        mock_path.return_value = MagicMock()
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = mock_path.return_value
        
        # Configure model loader and registry
        mock_loader_instance = MagicMock()
        mock_loader.return_value = mock_loader_instance
        mock_loader_instance.bootstrap_models = AsyncMock()
        mock_loader_instance.model_config = {"language_detection": {}, "translation": {}}
        mock_loader_instance.registry = mock_registry.return_value
        
        # Configure model manager
        mock_manager_instance = MagicMock()
        mock_manager.return_value = mock_manager_instance
        mock_manager_instance.load_model = AsyncMock()
        mock_manager_instance.unload_all_models = AsyncMock()
        mock_manager_instance.get_model_info = MagicMock(return_value={})
        
        # Configure tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_pipeline.return_value = mock_tokenizer_instance
        
        # Configure processor
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        mock_processor_instance.initialize = AsyncMock()
        
        # Configure audit logger
        mock_audit_instance = MagicMock()
        mock_audit.return_value = mock_audit_instance
        mock_audit_instance.initialize = AsyncMock()
        mock_audit_instance.flush = AsyncMock()
        
        # Configure metrics collector
        mock_metrics_instance = MagicMock()
        mock_metrics.return_value = mock_metrics_instance
        mock_metrics_instance.save_metrics = MagicMock()
        
        # Configure persistence
        mock_persistence_instance = MagicMock()
        mock_persistence.return_value = mock_persistence_instance
        
        # Configure console mocks
        mock_console_logger = MagicMock()
        mock_console_logging.return_value = mock_console_logger
        mock_console_logger.handlers = [MagicMock()]
        
        yield {
            "banner": mock_banner,
            "logging": mock_logging,
            "config": mock_config,
            "model_manager": mock_manager,
            "model_loader": mock_loader,
            "audit_logger": mock_audit,
            "metrics": mock_metrics,
            "persistence": mock_persistence,
            "processor": mock_processor,
            "tokenizer_pipeline": mock_tokenizer_pipeline,
            "app_logger": mock_app_logger,
            "session_manager": mock_session_manager,
            "console": mock_console,
            "registry": mock_registry,
        }

# Mock for download_models module
@pytest.fixture
def mock_download_models():
    with patch("scripts.download_models.download_model") as mock_download:
        mock_download.return_value = True
        
        # Mock the model definitions
        mock_default_models = {
            "translation_model": {
                "name": "Multilingual Translation",
                "model_name": "facebook/mbart-large-50-many-to-many-mmt"
            },
            "language_detection": {
                "name": "Language Detection",
                "model_name": "papluca/xlm-roberta-base-language-detection"
            }
        }
        
        mock_advanced_models = {
            "translation_small": {
                "name": "Translation Small",
                "model_name": "facebook/mbart-large-50-one-to-many-mmt"
            }
        }
        
        with patch("scripts.download_models.DEFAULT_MODELS", mock_default_models), \
             patch("scripts.download_models.ADVANCED_MODELS", mock_advanced_models):
            yield mock_download

# No global pytest marker to avoid marking non-async tests

def test_lifespan_function_existence():
    """Test the existence of the lifespan function in app.main"""
    from app.main import lifespan
    from contextlib import asynccontextmanager
    
    # Check that the lifespan function is a context manager
    assert lifespan.__name__ == 'lifespan'
    # All asynccontextmanager-decorated functions have __wrapped__ attribute
    assert hasattr(lifespan, '__wrapped__')

def test_app_properties():
    """Test the app properties without importing the actual app"""
    # Test properties directly from the main module
    from app.main import ModelSize, ModelType, ProcessorType, ModelSizeConfig
    
    # Verify the app constants
    assert isinstance(ModelSize.LARGE, str)
    assert isinstance(ModelType.TRANSLATION, str)
    assert isinstance(ProcessorType.INTEL, str)
    
    # Verify model size config mappings
    assert ModelSizeConfig.MEMORY_REQUIREMENTS[ModelType.TRANSLATION][ModelSize.LARGE] > 0
    assert "models/translation" in ModelSizeConfig.MODEL_PATHS[ModelType.TRANSLATION][ModelSize.LARGE]

@pytest.mark.parametrize("model_type,model_size", [
    ("TRANSLATION", "LARGE"),
    ("TRANSLATION", "MEDIUM"),
    ("TRANSLATION", "SMALL"),
    ("MULTIPURPOSE", "LARGE"),
    ("MULTIPURPOSE", "MEDIUM"),
    ("MULTIPURPOSE", "SMALL"),
    ("VERIFICATION", "LARGE"),
    ("VERIFICATION", "MEDIUM"),
    ("VERIFICATION", "SMALL"),
])
def test_model_size_config_comprehensive(model_type, model_size):
    """Test the ModelSizeConfig class thoroughly with parameterized tests"""
    from app.main import ModelSizeConfig, ModelType, ModelSize
    
    # Test memory requirements
    assert ModelSizeConfig.MEMORY_REQUIREMENTS[getattr(ModelType, model_type)][getattr(ModelSize, model_size)] > 0
    
    # Test model paths
    path = ModelSizeConfig.MODEL_PATHS[getattr(ModelType, model_type)][getattr(ModelSize, model_size)]
    assert f"models/{model_type.lower()}" in path
    assert model_size.lower() in path

def test_model_size_config_relative_sizing():
    """Test relative sizing logic in ModelSizeConfig"""
    from app.main import ModelSizeConfig, ModelType, ModelSize
    
    # For each model type, ensure the memory requirements follow LARGE > MEDIUM > SMALL
    for model_type in ModelType:
        # Test relative sizing
        assert (
            ModelSizeConfig.MEMORY_REQUIREMENTS[model_type][ModelSize.LARGE] >
            ModelSizeConfig.MEMORY_REQUIREMENTS[model_type][ModelSize.MEDIUM] >
            ModelSizeConfig.MEMORY_REQUIREMENTS[model_type][ModelSize.SMALL]
        )

def test_gpuinfo_class():
    """Test the GPUInfo class"""
    from app.main import GPUInfo
    
    # Create a GPU info object
    gpu = GPUInfo(
        device_id=0,
        name="Test GPU",
        memory_total=8 * 1024 * 1024 * 1024,  # 8GB
        memory_available=6 * 1024 * 1024 * 1024,  # 6GB
        compute_capability="7.5",
        vendor="nvidia"
    )
    
    # Test properties
    assert gpu.device_id == 0
    assert gpu.name == "Test GPU"
    assert gpu.memory_total == 8 * 1024 * 1024 * 1024
    assert gpu.memory_available == 6 * 1024 * 1024 * 1024
    assert gpu.compute_capability == "7.5"
    assert gpu.vendor == "nvidia"
    
    # Test string representation
    assert "Test GPU" in str(gpu)
    assert "8.0 GB" in str(gpu)

def test_enhanced_hardware_info_class():
    """Test the EnhancedHardwareInfo class"""
    from app.main import EnhancedHardwareInfo, ProcessorType, GPUInfo
    
    # Create test GPU info
    test_gpu = GPUInfo(
        device_id=0,
        name="Test GPU",
        memory_total=8 * 1024 * 1024 * 1024,  # 8GB
        memory_available=6 * 1024 * 1024 * 1024,  # 6GB
        vendor="nvidia"
    )
    
    # Create enhanced hardware info
    hardware_info = EnhancedHardwareInfo(
        total_memory=32 * 1024 * 1024 * 1024,  # 32GB
        available_memory=16 * 1024 * 1024 * 1024,  # 16GB
        processor_type=ProcessorType.NVIDIA,
        has_gpu=True,
        cpu_cores=8,
        cpu_threads=16,
        system_name="Linux",
        gpu_count=1,
        gpu_memory=8 * 1024 * 1024 * 1024,  # 8GB
        gpu_name="Test GPU",
        gpus=[test_gpu]
    )
    
    # Test properties
    assert hardware_info.total_memory == 32 * 1024 * 1024 * 1024
    assert hardware_info.available_memory == 16 * 1024 * 1024 * 1024
    assert hardware_info.processor_type == ProcessorType.NVIDIA
    assert hardware_info.has_gpu is True
    assert hardware_info.cpu_cores == 8
    assert hardware_info.cpu_threads == 16
    assert hardware_info.system_name == "Linux"
    assert hardware_info.gpu_count == 1
    assert hardware_info.gpu_memory == 8 * 1024 * 1024 * 1024
    assert hardware_info.gpu_name == "Test GPU"
    assert len(hardware_info.gpus) == 1
    
    # Test GPU methods
    assert hardware_info.get_best_gpu() == test_gpu
    assert hardware_info.get_gpu_by_id(0) == test_gpu
    assert hardware_info.get_gpu_by_id(1) is None
    assert hardware_info.get_total_gpu_memory() == 8 * 1024 * 1024 * 1024
    assert hardware_info.get_available_gpu_memory() == 6 * 1024 * 1024 * 1024

if __name__ == "__main__":
    pytest.main(["-v", __file__])