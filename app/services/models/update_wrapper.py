#!/usr/bin/env python3
"""
Update the wrapper.py file to use the EmbeddingModelWrapper class
"""

import sys
import os
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Path to the wrapper.py file
WRAPPER_PATH = Path(__file__).parent / "wrapper.py"

def update_wrapper():
    """
    Update the wrapper.py file to import and use the EmbeddingModelWrapper class
    """
    print(f"Updating {WRAPPER_PATH} to use EmbeddingModelWrapper")
    
    # Read the wrapper.py file
    with open(WRAPPER_PATH, 'r') as f:
        content = f.read()
        
    # Add import for EmbeddingModelWrapper
    import_line = "# Import model loader\nfrom app.services.models.embedding_wrapper import EmbeddingModelWrapper"
    
    if "from app.services.models.embedding_wrapper import" not in content:
        # Add import after other imports
        import_section_end = content.find("\n# Configure logging")
        if import_section_end == -1:
            import_section_end = content.find("\nlogger = logging.getLogger")
            
        if import_section_end != -1:
            content = content[:import_section_end] + "\n" + import_line + content[import_section_end:]
    
    # Update the wrapper_map to include embedding_model
    wrapper_map_pattern = r'wrapper_map = {\s*"translation":.+?}'
    wrapper_map_match = re.search(wrapper_map_pattern, content, re.DOTALL)
    
    if wrapper_map_match:
        wrapper_map = wrapper_map_match.group(0)
        # Check if embedding_model is already in the map
        if '"embedding_model"' not in wrapper_map:
            # Add embedding_model to the map
            updated_map = wrapper_map.replace(
                "}",
                '    "embedding_model": EmbeddingModelWrapper,\n}'
            )
            content = content.replace(wrapper_map, updated_map)
    else:
        print("Could not find wrapper_map in the file")
        return False
    
    # Write the updated content back to the file
    with open(WRAPPER_PATH, 'w') as f:
        f.write(content)
    
    print(f"Successfully updated {WRAPPER_PATH}")
    return True

if __name__ == "__main__":
    success = update_wrapper()
    if success:
        print("Update completed successfully.")
    else:
        print("Failed to update wrapper.py")
        sys.exit(1)