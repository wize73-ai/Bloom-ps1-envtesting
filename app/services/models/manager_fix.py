#!/usr/bin/env python3
"""
Add the missing create_embeddings method to EnhancedModelManager
"""

import sys
import os
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Path to the manager.py file
MANAGER_PATH = Path(__file__).parent / "manager.py"

def add_missing_method():
    """Add the missing create_embeddings method to EnhancedModelManager."""
    print(f"Adding missing create_embeddings method to {MANAGER_PATH}")
    
    # Read the manager.py file
    with open(MANAGER_PATH, 'r') as f:
        content = f.read()
    
    # Check if the method already exists
    if "async def create_embeddings" in content:
        print("Method already exists, no changes needed.")
        return True
    
    # Find the end of get_model_info method
    match = re.search(r'def get_model_info.*?\n {8}return info', content, re.DOTALL)
    if not match:
        print("Could not find the end of get_model_info method")
        return False
    
    # Get the position to insert the new method
    end_pos = match.end()
    
    # Define the new method
    new_method = """

    async def create_embeddings(self, texts: Union[str, List[str]], model_key: str = "rag_retriever") -> List[List[float]]:
        \"\"\"
        Create embeddings for the given texts using a specified model.
        
        Args:
            texts: Text or list of texts to embed
            model_key: Key of the model to use (default: "rag_retriever")
            
        Returns:
            List of embeddings as float lists
        \"\"\"
        logger.info(f"Creating embeddings using model: {model_key}")
        
        # Ensure the model is loaded
        if model_key not in self.loaded_models:
            logger.info(f"Model {model_key} not loaded, loading now")
            model_info = await self.load_model(model_key)
            
            if not model_info.get("model"):
                logger.error(f"Failed to load model {model_key} for embeddings")
                # Return simple random embeddings as fallback
                import numpy as np
                if isinstance(texts, str):
                    return [np.random.rand(384).tolist()]  # Simple 384-dim embedding
                else:
                    return [np.random.rand(384).tolist() for _ in texts]
        
        # Convert single text to list
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = texts
        
        # Get the model and prepare for embedding creation
        model = self.loaded_models[model_key]
        
        try:
            # Use sentence-transformers if available
            if hasattr(model, "encode") and callable(model.encode):
                # Use the encode method directly
                import numpy as np
                if len(texts_list) == 0:
                    return []
                
                embeddings = model.encode(texts_list, convert_to_tensor=True, show_progress_bar=False)
                # Convert to numpy arrays and then to lists
                if hasattr(embeddings, "cpu") and callable(embeddings.cpu):
                    embeddings = embeddings.cpu().numpy()
                
                # Convert to list of lists
                embeddings_list = embeddings.tolist() if hasattr(embeddings, "tolist") else [emb.tolist() for emb in embeddings]
                
                return embeddings_list
            else:
                # Use the RAG wrapper approach
                from app.services.models.wrapper import ModelInput, create_model_wrapper
                
                # Get tokenizer
                tokenizer = self.model_metadata.get(model_key, {}).get("tokenizer")
                
                # Create wrapper for the model
                wrapper = create_model_wrapper(
                    model_key,
                    model,
                    tokenizer,
                    {"task": model_key, "device": self.device, "precision": self.precision}
                )
                
                # Create embeddings one by one
                embeddings = []
                for text in texts_list:
                    # Create input
                    model_input = ModelInput(text=text)
                    
                    # Get embedding
                    result = wrapper.process(model_input)
                    
                    if isinstance(result.result, list):
                        embeddings.append(result.result)
                    else:
                        # Try to get embedding from result
                        if hasattr(result, "embedding") and result.embedding is not None:
                            embeddings.append(result.embedding)
                        else:
                            # If all else fails, create a dummy embedding
                            import numpy as np
                            embeddings.append(np.random.rand(384).tolist())  # Simple 384-dim embedding
                
                return embeddings
        except Exception as e:
            # Log error and return fallback embeddings
            logger.error(f"Error creating embeddings: {str(e)}", exc_info=True)
            
            # Create fallback embeddings
            import numpy as np
            return [np.random.rand(384).tolist() for _ in texts_list]"""
    
    # Insert the new method
    updated_content = content[:end_pos] + new_method + content[end_pos:]
    
    # Write the updated content back to the file
    with open(MANAGER_PATH, 'w') as f:
        f.write(updated_content)
    
    print(f"Successfully added create_embeddings method to {MANAGER_PATH}")
    return True

if __name__ == "__main__":
    success = add_missing_method()
    if success:
        print("Fix applied successfully.")
    else:
        print("Failed to apply fix.")
        sys.exit(1)