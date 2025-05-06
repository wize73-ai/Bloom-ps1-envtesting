#!/usr/bin/env python3
"""
Update the model registry to include embedding_model mapping
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Path to the model registry file
REGISTRY_PATH = Path(__file__).parent.parent.parent.parent / "config" / "model_registry.json"

def update_registry():
    """Update the model registry to include embedding_model mapping"""
    print(f"Updating model registry at {REGISTRY_PATH}")
    
    # Check if file exists
    if not REGISTRY_PATH.exists():
        print(f"Registry file {REGISTRY_PATH} not found")
        return False
    
    # Read the registry file
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)
    
    # Check if embedding_model is already defined
    if "embedding_model" in registry:
        print("embedding_model is already defined in the registry")
        return True
    
    # Add embedding_model mapping (if rag_retriever exists, use its definition)
    if "rag_retriever" in registry:
        print("Using rag_retriever definition for embedding_model")
        registry["embedding_model"] = registry["rag_retriever"].copy()
        registry["embedding_model"]["task"] = "embedding"
    else:
        # Create a new definition
        print("Creating new definition for embedding_model")
        registry["embedding_model"] = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "tokenizer_name": "sentence-transformers/all-MiniLM-L6-v2",
            "task": "embedding",
            "type": "sentence-transformers",
            "framework": "sentence-transformers"
        }
    
    # Write the updated registry
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"Successfully updated registry with embedding_model")
    return True

if __name__ == "__main__":
    success = update_registry()
    if success:
        print("Update completed successfully.")
    else:
        print("Failed to update registry")
        sys.exit(1)