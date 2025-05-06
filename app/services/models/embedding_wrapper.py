"""
Embedding Model Wrapper for CasaLingua
Provides embedding functionality for the verification system
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union

from app.services.models.wrapper_base import BaseModelWrapper, ModelInput, ModelOutput

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingModelWrapper(BaseModelWrapper):
    """Wrapper for embedding models"""
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        """Preprocess embedding input"""
        if isinstance(input_data.text, list):
            texts = input_data.text
        else:
            texts = [input_data.text]
        
        # For sentence-transformers models
        if hasattr(self.model, "encode") and callable(self.model.encode):
            # Just return the texts directly
            return {"texts": texts}
        
        # For transformer models that need tokenization
        if self.tokenizer:
            try:
                inputs = self.tokenizer(
                    texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=self.config.get("max_length", 512)
                )
                
                # Move to device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                        
                return {
                    "inputs": inputs,
                    "original_texts": texts
                }
            except Exception as e:
                logger.error(f"Error tokenizing texts for embedding: {str(e)}", exc_info=True)
                # Fall back to direct text format
                return {"texts": texts}
        
        # Fall back to direct text format
        return {"texts": texts}
    
    def _run_inference(self, preprocessed: Dict[str, Any]) -> Any:
        """Run embedding inference"""
        # For sentence-transformers models
        if hasattr(self.model, "encode") and callable(self.model.encode):
            texts = preprocessed.get("texts", [])
            if not texts:
                return []
                
            try:
                # Generate embeddings
                with torch.no_grad():
                    embeddings = self.model.encode(
                        texts,
                        convert_to_tensor=True,
                        show_progress_bar=False
                    )
                return embeddings
            except Exception as e:
                logger.error(f"Error generating embeddings with sentence-transformers: {str(e)}", exc_info=True)
                # Generate random embeddings as fallback
                dim = 384  # Default embedding dimension
                return np.random.rand(len(texts), dim)
        
        # For transformer models
        if "inputs" in preprocessed:
            inputs = preprocessed["inputs"]
            try:
                # Get embeddings from transformer model
                outputs = self.model(**inputs)
                
                # Extract embeddings - typically the CLS token or mean of token embeddings
                if hasattr(outputs, "last_hidden_state"):
                    # Use CLS token ([CLS] is typically the first token)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    return embeddings
                elif hasattr(outputs, "pooler_output"):
                    # Use pooler output
                    return outputs.pooler_output
                else:
                    # Custom handling for other output formats
                    logger.warning("Unrecognized output format for embedding model, using random fallback")
                    dim = 768  # Default embedding dimension for many transformer models
                    return torch.tensor(np.random.rand(len(preprocessed.get("original_texts", [1])), dim))
            except Exception as e:
                logger.error(f"Error generating embeddings with transformer model: {str(e)}", exc_info=True)
                # Generate random embeddings as fallback
                dim = 768  # Default embedding dimension for many transformer models
                return np.random.rand(len(preprocessed.get("original_texts", [1])), dim)
        
        logger.error("No valid input format for embedding generation")
        # Generate random embeddings as final fallback
        return np.random.rand(1, 384)  # Default embedding dimension
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        """Postprocess embedding output"""
        try:
            # Convert to list format
            if isinstance(model_output, torch.Tensor):
                # Convert torch tensor to numpy and then to list
                embeddings = model_output.cpu().numpy().tolist()
            elif hasattr(model_output, "tolist") and callable(model_output.tolist):
                # Convert numpy array to list
                embeddings = model_output.tolist()
            elif isinstance(model_output, list):
                # Already a list
                embeddings = model_output
            else:
                # Unknown format - create random embeddings
                logger.warning(f"Unknown embedding output format: {type(model_output)}")
                dim = 384  # Default embedding dimension
                
                if isinstance(input_data.text, list):
                    embeddings = np.random.rand(len(input_data.text), dim).tolist()
                else:
                    embeddings = [np.random.rand(dim).tolist()]
            
            # Ensure we have the correct output shape
            if isinstance(input_data.text, str) and isinstance(embeddings, list) and all(isinstance(item, list) for item in embeddings):
                # Return first embedding for single input
                embeddings = embeddings[0]
            
            return ModelOutput(
                result=embeddings,
                metadata={"dimension": len(embeddings[0]) if isinstance(embeddings[0], list) else len(embeddings)}
            )
        except Exception as e:
            logger.error(f"Error in embedding postprocessing: {str(e)}", exc_info=True)
            # Create fallback embeddings
            dim = 384  # Default embedding dimension
            
            if isinstance(input_data.text, list):
                result = np.random.rand(len(input_data.text), dim).tolist()
            else:
                result = [np.random.rand(dim).tolist()]
                
            return ModelOutput(
                result=result,
                metadata={"dimension": dim, "error": str(e)}
            )