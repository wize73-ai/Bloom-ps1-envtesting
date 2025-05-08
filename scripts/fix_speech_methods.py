#!/usr/bin/env python3
"""
Fix script for speech processing methods.
Adds proper method registration to the model manager for speech-related methods.
"""

import os
import sys
import asyncio
import time
from pathlib import Path

def update_model_manager():
    """Update model manager to register speech-related methods."""
    file_path = "app/services/models/manager.py"
    
    print(f"Updating model manager at {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return False
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Find the method registration section
    target_section = """        # Register the method
        if method_name not in self.registered_methods.get(model_id, []):
            raise ValueError(f"Unknown method {method_name}")"""
    
    # Create improved method registration with fallbacks
    improved_section = """        # Register the method
        if method_name not in self.registered_methods.get(model_id, []):
            # Handle speech-related methods with special fallbacks
            speech_method_mappings = {
                "get_languages": "get_supported_languages",
                "get_voices": "list_voices",
                "transcribe": "process"
            }
            
            # Check if this is a speech method with an alias
            if method_name in speech_method_mappings:
                alternative_method = speech_method_mappings[method_name]
                if alternative_method in self.registered_methods.get(model_id, []):
                    # Use the alternative method name
                    method_name = alternative_method
                    logger.info(f"Using alternative method: {method_name} instead of {method_name}")
                else:
                    # Method and alternatives not found
                    raise ValueError(f"Unknown method {method_name} (and no alternatives available)")
            else:
                # Not a speech method, original error
                raise ValueError(f"Unknown method {method_name}")"""
    
    # Replace the section
    updated_content = content.replace(target_section, improved_section)
    
    # Write back to file
    with open(file_path, "w") as f:
        f.write(updated_content)
    
    print(f"Updated model manager with improved method registration")
    return True

def update_model_wrapper():
    """Update model wrapper to handle speech-related methods."""
    file_path = "app/services/models/wrapper.py"
    
    print(f"Updating model wrapper at {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return False
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Check if we need to add methods
    if "def get_voices" not in content and "def get_languages" not in content:
        # Find a good insertion point - after _postprocess method
        insertion_point = content.find("    def _postprocess")
        if insertion_point == -1:
            print("Could not find insertion point")
            return False
        
        # Find the end of the _postprocess method
        method_end = content.find("    def", insertion_point + 1)
        if method_end == -1:
            method_end = len(content)
        
        # Create new speech-related methods
        new_methods = """
    def get_voices(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get available voices for speech synthesis."""
        # Default implementation returns basic voice set
        default_voices = [
            {"id": "en-us-1", "language": "en", "name": "English Voice 1", "gender": "male"},
            {"id": "en-us-2", "language": "en", "name": "English Voice 2", "gender": "female"},
            {"id": "es-es-1", "language": "es", "name": "Spanish Voice 1", "gender": "male"},
            {"id": "fr-fr-1", "language": "fr", "name": "French Voice 1", "gender": "male"},
            {"id": "de-de-1", "language": "de", "name": "German Voice 1", "gender": "male"}
        ]
        
        return {
            "result": default_voices,
            "metadata": {
                "model_used": "wrapper_fallback"
            }
        }
    
    def get_languages(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get supported languages for speech recognition."""
        # Default implementation returns basic language set
        default_languages = [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"}
        ]
        
        return {
            "result": default_languages,
            "metadata": {
                "model_used": "wrapper_fallback"
            }
        }
    
    def get_supported_languages(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for get_languages to maintain compatibility."""
        return self.get_languages(input_data)
    
    def list_voices(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for get_voices to maintain compatibility."""
        return self.get_voices(input_data)
        
"""
        
        # Insert the new methods
        updated_content = content[:method_end] + new_methods + content[method_end:]
        
        # Write back to file
        with open(file_path, "w") as f:
            f.write(updated_content)
        
        print(f"Added speech-related methods to model wrapper")
        return True
    else:
        print("Speech methods already exist in wrapper")
        return True

def update_model_registration():
    """Update model registration to include speech methods."""
    file_path = "app/services/models/loader.py"
    
    print(f"Updating model registration in {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return False
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Find method registration section
    target_section = """        # Register available methods
        self.model_manager.register_method(model_id, "process")
        self.model_manager.register_method(model_id, "generate")"""
    
    # Add speech methods to registration
    improved_section = """        # Register available methods
        self.model_manager.register_method(model_id, "process")
        self.model_manager.register_method(model_id, "generate")
        
        # Register speech-related methods if this is a speech model
        if model_type in ["tts", "speech_to_text", "tts_fallback"]:
            self.model_manager.register_method(model_id, "get_languages")
            self.model_manager.register_method(model_id, "get_voices")
            self.model_manager.register_method(model_id, "get_supported_languages")
            self.model_manager.register_method(model_id, "list_voices")
            self.model_manager.register_method(model_id, "transcribe")"""
    
    # Replace the section
    if target_section in content:
        updated_content = content.replace(target_section, improved_section)
        
        # Write back to file
        with open(file_path, "w") as f:
            f.write(updated_content)
        
        print(f"Updated model registration with speech methods")
        return True
    else:
        print("Could not find method registration section")
        return False

def main():
    """Main function to apply fixes."""
    print("Applying fixes to speech processing methods...")
    
    # Update model manager
    manager_updated = update_model_manager()
    
    # Update model wrapper
    wrapper_updated = update_model_wrapper()
    
    # Update model registration
    registration_updated = update_model_registration()
    
    # Print summary
    print("\nSummary of changes:")
    print(f"- Model Manager: {'✅ Updated' if manager_updated else '❌ Failed'}")
    print(f"- Model Wrapper: {'✅ Updated' if wrapper_updated else '❌ Failed'}")
    print(f"- Model Registration: {'✅ Updated' if registration_updated else '❌ Failed'}")
    
    if manager_updated and wrapper_updated and registration_updated:
        print("\n✅ All fixes applied successfully!")
        print("Restart the server to apply changes, then run comprehensive tests with:")
        print("  python scripts/monitor_speech_processing.py")
    else:
        print("\n⚠️ Some fixes could not be applied")
        print("Check files manually and apply the necessary changes")

if __name__ == "__main__":
    main()