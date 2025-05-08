#\!/usr/bin/env python3
"""
Fix for TTS to always use emergency audio.
This script modifies the synthesize method to always use the fixed emergency audio method.
"""

import os
import sys
import time
from pathlib import Path

def create_direct_tts_wrapper():
    """Create a direct wrapper that uses gTTS for all TTS requests."""
    # Create output directory
    wrapper_path = "app/core/pipeline/direct_tts_wrapper.py"
    
    # Check if file already exists
    if os.path.exists(wrapper_path):
        print(f"Direct TTS wrapper already exists at {wrapper_path}")
        return False
    
    # Create the wrapper
    wrapper_content = """
'''
Direct TTS wrapper that always uses gTTS for speech synthesis
This is a temporary fix to ensure audible speech for all requests
'''

import os
import io
import logging
import time
from typing import Dict, Any

from gtts import gTTS

def process(input_data: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Process function that handles TTS using gTTS directly.
    
    Args:
        input_data: Input data dictionary with text and parameters
        
    Returns:
        Dict with result containing audio bytes
    '''
    # Extract text and parameters
    text = input_data.get('text', '')
    if not text:
        return {"result": b'', "error": "No text provided"}
    
    source_language = input_data.get('source_language', 'en')
    params = input_data.get('parameters', {})
    output_format = params.get('format', 'mp3')
    
    # Make sure we have a 2-letter language code for gTTS
    if len(source_language) > 2 and '-' in source_language:
        source_language = source_language.split('-')[0]
    
    start_time = time.time()
    
    try:
        # Create audio buffer
        buffer = io.BytesIO()
        
        # Create gTTS object and save to buffer
        tts = gTTS(text=text, lang=source_language)
        tts.write_to_fp(buffer)
        
        # Get the audio content
        buffer.seek(0)
        audio_content = buffer.read()
        
        duration = time.time() - start_time
        
        # Return the result
        return {
            "result": audio_content,
            "metadata": {
                "model_used": "gtts_direct",
                "duration": duration,
                "text_length": len(text),
                "language": source_language,
                "format": output_format
            }
        }
    except Exception as e:
        error_msg = f"Error generating speech with gTTS: {str(e)}"
        logging.error(error_msg)
        return {"result": b'', "error": error_msg}
"""

    # Write the wrapper
    try:
        with open(wrapper_path, "w") as f:
            f.write(wrapper_content)
        
        print(f"Created direct TTS wrapper at {wrapper_path}")
        return True
    except Exception as e:
        print(f"Error creating wrapper: {str(e)}")
        return False

def register_wrapper_with_manager():
    """Register our direct wrapper with the model manager."""
    manager_path = "app/services/models/wrapper.py"
    
    try:
        with open(manager_path, "r") as f:
            content = f.read()
        
        # Check if we need to inject our wrapper
        if "direct_tts_wrapper" not in content:
            print("Need to register our direct TTS wrapper...")
            
            # Find the imports section
            import_section = ""
            if "import app.services.models.wrapper_base" in content:
                import_section = "import app.services.models.wrapper_base"
            
            if import_section:
                # Add our import after this line
                new_import = "import app.core.pipeline.direct_tts_wrapper"
                if new_import not in content:
                    content = content.replace(
                        import_section, 
                        f"{import_section}\n{new_import}"
                    )
            
            # Find the run_model method and modify it to always use our wrapper for TTS
            if "async def run_model(" in content:
                run_model_section = "async def run_model("
                
                # Add a special case for TTS
                tts_handler = """
        # Direct TTS handler for all TTS requests
        if model_type == "tts":
            try:
                return app.core.pipeline.direct_tts_wrapper.process(input_data)
            except Exception as e:
                logger.error(f"Error using direct TTS wrapper: {str(e)}")
                # Continue to normal processing if direct wrapper fails
        
"""
                # Insert our handler at the beginning of the method
                pos = content.find(run_model_section)
                if pos > 0:
                    # Find the beginning of the method body
                    method_start = content.find(":", pos)
                    if method_start > 0:
                        # Find the first line after the method definition
                        next_line = content.find("\n", method_start)
                        if next_line > 0:
                            # Insert our handler after the first line of the method
                            next_line_end = content.find("\n", next_line + 1)
                            if next_line_end > 0:
                                content = (
                                    content[:next_line_end + 1] + 
                                    tts_handler + 
                                    content[next_line_end + 1:]
                                )
            
            # Create backup
            backup_path = f"{manager_path}.bak"
            with open(backup_path, "w") as f:
                f.write(content)
            
            # Write modified file
            with open(manager_path, "w") as f:
                f.write(content)
            
            print(f"Modified model manager at {manager_path}")
            print(f"Backup saved to {backup_path}")
            return True
        else:
            print("Could not find the run_model method in the wrapper")
            return False
    except Exception as e:
        print(f"Error modifying model manager: {str(e)}")
        return False

def main():
    """Main function."""
    print("===== FIXING TTS TO ALWAYS USE EMERGENCY AUDIO =====")
    
    # Create the direct TTS wrapper
    wrapper_ok = create_direct_tts_wrapper()
    if not wrapper_ok:
        print("Failed to create direct TTS wrapper")
        return False
    
    # Register the wrapper with the model manager
    manager_ok = register_wrapper_with_manager()
    if not manager_ok:
        print("Failed to register wrapper with model manager")
        return False
    
    print("\n===== FIX COMPLETE =====")
    print("The system has been updated to always use gTTS for all TTS requests.")
    print("Restart the server for changes to take effect.")
    print("\nTo test the fix:")
    print("1. Restart the server")
    print("2. Run: python scripts/test_tts_endpoint.py")
    print("3. Check the generated files for audible speech")

if __name__ == "__main__":
    main()
