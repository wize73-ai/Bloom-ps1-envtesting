#\!/usr/bin/env python3
"""
Fix for TTS emergency audio to ensure real, audible files are generated.
This script modifies the _create_emergency_audio method in the TTS pipeline.
"""

import os
import sys
import time
from pathlib import Path

def create_fixed_tts_module():
    """Create a fixed version of the TTS module that uses gTTS for emergency audio."""
    tts_module_path = "app/core/pipeline/tts.py"
    
    # Read the original file
    try:
        with open(tts_module_path, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading TTS module: {str(e)}")
        return False
    
    # Find the _create_emergency_audio method
    if "_create_emergency_audio" not in content:
        print("Could not find _create_emergency_audio method in TTS module")
        return False
    
    # Create the improved implementation
    improved_method = """    def _create_emergency_audio(self, output_format: str) -> bytes:
        \"\"\"
        Create emergency audio content when model fails.
        Uses gTTS to generate a real, audible fallback.
        
        Args:
            output_format: Output format (mp3, wav, etc.)
            
        Returns:
            Binary audio content
        \"\"\"
        logger.info("Creating emergency audio content with gTTS")
        
        # Emergency message
        emergency_text = "This is an emergency fallback audio file generated by the text to speech system."
        
        # Try to use gTTS for a real audio file
        try:
            # Import gTTS only when needed
            from gtts import gTTS
            import io
            
            # Create a buffer to hold the audio
            buffer = io.BytesIO()
            
            # Create gTTS object and save to buffer
            tts = gTTS(text=emergency_text, lang="en")
            tts.write_to_fp(buffer)
            
            # Get the buffer content
            buffer.seek(0)
            audio_content = buffer.read()
            
            # If we got content, return it
            if audio_content and len(audio_content) > 1000:
                logger.info(f"Created emergency audio with gTTS: {len(audio_content)} bytes")
                return audio_content
                
        except ImportError:
            logger.warning("gTTS not available for emergency audio, trying fallback")
        except Exception as e:
            logger.error(f"Error creating emergency audio with gTTS: {str(e)}")
        
        # If gTTS fails, check for pre-generated emergency files
        try:
            # Possible locations for emergency files
            emergency_locations = [
                f"temp/emergency_fallback.{output_format}",
                f"temp/tts_emergency_0.{output_format}",
                f"audio/tts_emergency_0.{output_format}",
                f"temp/tts/tts_emergency_0.{output_format}"
            ]
            
            # Try each location
            for location in emergency_locations:
                if os.path.exists(location):
                    with open(location, "rb") as f:
                        audio_content = f.read()
                    
                    logger.info(f"Using pre-generated emergency audio: {location}")
                    return audio_content
                    
        except Exception as e:
            logger.error(f"Error reading pre-generated emergency audio: {str(e)}")
        
        # If all else fails, create a silent audio file as absolute last resort
        logger.warning("Using silent audio as last resort fallback")
        
        # Create audio content based on format
        if output_format == "mp3":
            # Simple MP3 file header + minimal data
            silence_mp3 = b'\\xFF\\xFB\\x10\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00'
            return silence_mp3 * 100  # Repeat to make it longer
        elif output_format == "wav":
            # Create a WAV file with silence
            buffer = io.BytesIO()
            
            # WAV header parameters
            sample_rate = 44100
            bits_per_sample = 16
            channels = 1
            
            # Write WAV header
            buffer.write(b'RIFF')
            buffer.write((36).to_bytes(4, byteorder='little'))  # Chunk size
            buffer.write(b'WAVE')
            buffer.write(b'fmt ')
            buffer.write((16).to_bytes(4, byteorder='little'))  # Subchunk1 size
            buffer.write((1).to_bytes(2, byteorder='little'))   # Audio format (PCM)
            buffer.write(channels.to_bytes(2, byteorder='little'))
            buffer.write(sample_rate.to_bytes(4, byteorder='little'))
            byte_rate = sample_rate * channels * bits_per_sample // 8
            buffer.write(byte_rate.to_bytes(4, byteorder='little'))
            block_align = channels * bits_per_sample // 8
            buffer.write(block_align.to_bytes(2, byteorder='little'))
            buffer.write(bits_per_sample.to_bytes(2, byteorder='little'))
            buffer.write(b'data')
            buffer.write((0).to_bytes(4, byteorder='little'))  # Subchunk2 size (0 = empty)
            
            return buffer.getvalue()
        else:
            # For other formats, just return null bytes as placeholder
            return b'\\x00' * 1024"""
    
    # Fix escape sequences in the improved method string
    improved_method = improved_method.replace("\\\\xFF", "\\xFF").replace("\\\\x00", "\\x00")
    
    # Replace the old method with the improved one
    if "_create_emergency_audio" in content:
        # Split the content to find the method
        lines = content.split("\n")
        method_start = -1
        method_end = -1
        
        # Find the method boundaries
        for i, line in enumerate(lines):
            if "def _create_emergency_audio" in line:
                method_start = i
            elif method_start >= 0 and i > method_start:
                # Check if this line is the start of a new method or class (indentation level)
                if line.strip() and not line.startswith(" " * 4):
                    method_end = i
                    break
                # Also check for indented methods at the same level
                elif line.strip().startswith("def ") and line.startswith(" " * 4):
                    method_end = i
                    break
        
        # If we reached the end of the file
        if method_end == -1:
            method_end = len(lines)
        
        # Replace the method
        if method_start >= 0 and method_end > method_start:
            new_content = "\n".join(lines[:method_start]) + "\n" + improved_method + "\n" + "\n".join(lines[method_end:])
            
            # Write the fixed file
            backup_path = f"{tts_module_path}.bak"
            print(f"Creating backup of original file: {backup_path}")
            with open(backup_path, "w") as f:
                f.write(content)
            
            print(f"Writing fixed TTS module")
            with open(tts_module_path, "w") as f:
                f.write(new_content)
            
            print("TTS module updated successfully")
            return True
        else:
            print("Could not locate method boundaries")
            return False
    
    print("Could not find _create_emergency_audio method")
    return False

def ensure_gtts_installed():
    """Ensure gTTS is installed in the environment."""
    try:
        import gtts
        print("gTTS is already installed")
        return True
    except ImportError:
        print("Installing gTTS...")
        result = os.system("pip install gtts")
        
        if result == 0:
            print("gTTS installed successfully")
            return True
        else:
            print("Failed to install gTTS")
            return False

def main():
    """Main function."""
    print("===== FIXING TTS EMERGENCY AUDIO =====")
    
    # Make sure gTTS is installed
    if not ensure_gtts_installed():
        print("Error: gTTS must be installed for this fix to work")
        sys.exit(1)
    
    # Create the fixed TTS module
    if create_fixed_tts_module():
        print("\n===== FIX COMPLETE =====")
        print("The TTS module has been updated to use gTTS for emergency audio.")
        print("Restart the server for changes to take effect.")
        print("\nTo test the fix:")
        print("1. Restart the server")
        print("2. Run: python scripts/test_tts_endpoint.py")
        print("3. Check the generated files for audible speech")
    else:
        print("\n===== FIX FAILED =====")
        print("There was an error updating the TTS module.")
        print("Please check the logs for details.")

if __name__ == "__main__":
    main()
