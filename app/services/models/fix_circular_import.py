"""
Fix for circular import issues between wrapper.py and embedding_wrapper.py
This module is imported at startup to ensure both modules are fully loaded
"""

import sys
import logging

logger = logging.getLogger(__name__)

def fix_circular_imports():
    """Fix circular imports by ensuring modules are fully loaded"""
    try:
        # First ensure wrapper_base is loaded
        if 'app.services.models.wrapper_base' not in sys.modules:
            import app.services.models.wrapper_base
            logger.info("Loaded wrapper_base module")
        
        # Then ensure wrapper is loaded
        if 'app.services.models.wrapper' not in sys.modules:
            import app.services.models.wrapper
            logger.info("Loaded wrapper module")
            
        # Finally ensure embedding_wrapper is loaded
        if 'app.services.models.embedding_wrapper' not in sys.modules:
            import app.services.models.embedding_wrapper
            logger.info("Loaded embedding_wrapper module")
            
        # Load all necessary modules
        from app.services.models.wrapper_base import BaseModelWrapper, ModelInput, ModelOutput
        from app.services.models.embedding_wrapper import EmbeddingModelWrapper
        
        # Patch wrapper's get_wrapper_for_model function
        from app.services.models.wrapper import get_wrapper_for_model
        
        # Ensure EmbeddingModelWrapper is in the wrapper_map
        from app.services.models.wrapper import wrapper_map
        if 'embedding_model' not in wrapper_map and hasattr(wrapper_map, '__setitem__'):
            wrapper_map['embedding_model'] = EmbeddingModelWrapper
            logger.info("Added EmbeddingModelWrapper to wrapper_map")
            
        logger.info("Successfully fixed circular imports")
        return True
    except Exception as e:
        logger.error(f"Error fixing circular imports: {str(e)}", exc_info=True)
        return False

# Run the fix when this module is imported
fix_result = fix_circular_imports()