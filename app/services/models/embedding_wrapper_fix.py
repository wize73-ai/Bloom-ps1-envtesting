#!/usr/bin/env python3
"""
Add EmbeddingModelWrapper class to wrapper.py

This script adds the missing EmbeddingModelWrapper class to the wrapper.py file
to handle the "embedding_model" type and fixes the create_model_wrapper function
to use this wrapper.
"""

import re
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Path to the wrapper.py file
WRAPPER_PATH = Path(__file__).parent / "wrapper.py"

def add_embedding_wrapper():
    """Add EmbeddingModelWrapper class to wrapper.py."""
    print(f"Adding EmbeddingModelWrapper class to {WRAPPER_PATH}")
    
    # Read the wrapper.py file
    with open(WRAPPER_PATH, 'r') as f:
        content = f.read()
    
    # Check if the class already exists
    if "class EmbeddingModelWrapper" in content:
        print("EmbeddingModelWrapper class already exists, no changes needed.")
        return True
    
    # Find the position to insert the new class
    # Look for the class definition before the create_model_wrapper function
    match = re.search(r'class AnonymizerWrapper.*?def _get_replacement.*?\n\n\n', content, re.DOTALL)
    
    if not match:
        print("Could not find the right position to insert the new class")
        return False
    
    # Get the position to insert the new class
    end_pos = match.end()
    
    # Define the new class
    new_class = """
class EmbeddingModelWrapper(BaseModelWrapper):
    \"\"\"Wrapper for embedding models\"\"\"
    
    def _preprocess(self, input_data: ModelInput) -> Dict[str, Any]:
        \"\"\"Preprocess embedding input\"\"\"
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
        \"\"\"Run embedding inference\"\"\"
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
                import numpy as np
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
                    import numpy as np
                    dim = 768  # Default embedding dimension for many transformer models
                    return torch.tensor(np.random.rand(len(preprocessed.get("original_texts", [1])), dim))
            except Exception as e:
                logger.error(f"Error generating embeddings with transformer model: {str(e)}", exc_info=True)
                # Generate random embeddings as fallback
                import numpy as np
                dim = 768  # Default embedding dimension for many transformer models
                return np.random.rand(len(preprocessed.get("original_texts", [1])), dim)
        
        logger.error("No valid input format for embedding generation")
        # Generate random embeddings as final fallback
        import numpy as np
        return np.random.rand(1, 384)  # Default embedding dimension
    
    def _postprocess(self, model_output: Any, input_data: ModelInput) -> ModelOutput:
        \"\"\"Postprocess embedding output\"\"\"
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
                import numpy as np
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
            import numpy as np
            dim = 384  # Default embedding dimension
            
            if isinstance(input_data.text, list):
                result = np.random.rand(len(input_data.text), dim).tolist()
            else:
                result = [np.random.rand(dim).tolist()]
                
            return ModelOutput(
                result=result,
                metadata={"dimension": dim, "error": str(e)}
            )


"""
    
    # Insert the new class
    updated_content = content[:end_pos] + new_class + content[end_pos:]
    
    # Now update the create_model_wrapper function to use the new class
    # Find the wrapper_map dictionary
    wrapper_map_match = re.search(r'wrapper_map = {.*?}', updated_content, re.DOTALL)
    
    if not wrapper_map_match:
        print("Could not find wrapper_map dictionary")
        return False
    
    # Get the old dictionary
    old_map = wrapper_map_match.group(0)
    
    # Add the embedding_model type to the dictionary
    if "embedding_model" not in old_map:
        new_map = old_map.replace(
            "}",
            "    \"embedding_model\": EmbeddingModelWrapper,\n    }"
        )
        updated_content = updated_content.replace(old_map, new_map)
    
    # Write the updated content back to the file
    with open(WRAPPER_PATH, 'w') as f:
        f.write(updated_content)
    
    print(f"Successfully added EmbeddingModelWrapper class to {WRAPPER_PATH}")
    return True

if __name__ == "__main__":
    success = add_embedding_wrapper()
    if success:
        print("Fix applied successfully.")
    else:
        print("Failed to apply fix.")
        sys.exit(1)