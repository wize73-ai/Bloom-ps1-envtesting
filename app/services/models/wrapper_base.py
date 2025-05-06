"""
Base Model Wrapper for CasaLingua
This module contains the base wrapper class for models to avoid circular imports
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

class ModelInput(BaseModel):
    """Input model for standardized wrapper inputs"""
    text: Union[str, List[str]]
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    context: Optional[List[str]] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ModelOutput(BaseModel):
    """Output model for standardized wrapper outputs"""
    result: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class BaseModelWrapper:
    """Base class for all model wrappers in CasaLingua"""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any] = None, **kwargs):
        """Initialize the model wrapper"""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        
        # Always use CPU for now - MPS needs more work to properly support
        self.device = "cpu"
        logger.info("Using CPU device for language models")
        
        # Move model to CPU - we'll address MPS support in a separate PR
        if hasattr(self.model, "to") and callable(self.model.to):
            try:
                self.model = self.model.to("cpu")
                logger.info("Model moved to CPU")
            except Exception as e:
                logger.warning(f"Could not move model: {str(e)}")
        
        # Set model to evaluation mode if it's a PyTorch model
        if hasattr(self.model, "eval") and callable(self.model.eval):
            self.model.eval()
            
    async def process_async(self, input_data: Union[Dict[str, Any], ModelInput]) -> Dict[str, Any]:
        """Process input asynchronously"""
        # Convert to ModelInput if needed
        if isinstance(input_data, ModelInput):
            model_input = input_data
        else:
            model_input = ModelInput(**input_data)
        
        # Run processing pipeline
        preprocessed = self._preprocess(model_input)
        raw_output = self._run_inference(preprocessed)
        result = self._postprocess(raw_output, model_input)
        
        # Convert to dictionary
        if isinstance(result, ModelOutput):
            output_dict = {
                "result": result.result,
                "metadata": result.metadata
            }
        else:
            # Legacy model wrapper support
            output_dict = {
                "result": result
            }
        
        return output_dict
    
    def process(self, input_data: Union[Dict[str, Any], ModelInput]) -> Dict[str, Any]:
        """Synchronous version of process_async"""
        # Convert to ModelInput if needed
        if isinstance(input_data, ModelInput):
            model_input = input_data
        else:
            model_input = ModelInput(**input_data)
        
        # Run processing pipeline
        preprocessed = self._preprocess(model_input)
        raw_output = self._run_inference(preprocessed)
        result = self._postprocess(raw_output, model_input)
        
        # Convert to dictionary
        if isinstance(result, ModelOutput):
            output_dict = {
                "result": result.result,
                "metadata": result.metadata
            }
        else:
            # Legacy model wrapper support
            output_dict = {
                "result": result
            }
        
        return output_dict
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess input data - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _preprocess")
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run model inference - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _run_inference")
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess model output - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _postprocess")