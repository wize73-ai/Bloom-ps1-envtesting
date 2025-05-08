"""
ModelManager Class for CasaLingua

This is a compatibility wrapper to allow code to use ModelManager 
while using the EnhancedModelManager underneath.
"""

import logging
from typing import Dict, Any, List, Optional, Union

# Import EnhancedModelManager
from app.services.models.manager import EnhancedModelManager

# Configure logging
logger = logging.getLogger(__name__)

class ModelManager:
    """
    ModelManager compatibility wrapper for EnhancedModelManager
    """
    
    def __init__(self, registry_config=None):
        """
        Initialize the model manager with the given registry configuration
        
        Args:
            registry_config: Configuration for the model registry
        """
        self.registry_config = registry_config
        self.enhanced_manager = None
        
    async def initialize(self):
        """Initialize the enhanced model manager"""
        # Create hardware info for the enhanced manager
        hardware_info = {
            "gpu": {
                "has_gpu": False,
                "cuda_available": False,
                "mps_available": False,
                "gpu_memory": 0
            },
            "cpu": {
                "supports_avx2": False,
                "cores": 8
            }
        }
        
        # Import the ModelLoader
        from app.services.models.loader import ModelLoader
        
        # Create loader
        loader = ModelLoader(registry_config=self.registry_config)
        
        # Create enhanced manager
        self.enhanced_manager = EnhancedModelManager(
            loader=loader,
            hardware_info=hardware_info,
            config={"device": "cpu", "precision": "float32"}
        )
        
        logger.info("ModelManager initialized with EnhancedModelManager")
    
    async def list_loaded_models(self) -> List[str]:
        """
        Get a list of loaded model names
        
        Returns:
            List of model names
        """
        if not self.enhanced_manager:
            return []
            
        info = self.enhanced_manager.get_model_info()
        return [model_type for model_type, model_info in info.items() 
                if model_info.get("loaded", False) and model_type != "_system"]