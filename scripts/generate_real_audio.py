#!/usr/bin/env python3
"""
Generate a real audio file using gTTS for testing.
"""

import os
import sys
import tempfile
from pathlib import Path

def generate_audio():
    """Generate a test audio file with gTTS."""
    try:
        # Try to import gTTS
        from gtts import gTTS
        
        # Create a real speech file
        text = "This is a test of the text to speech system. Can you hear this audio?"
        print(f"Generating audio for text: '{text}'")
        
        # Create output file
        output_file = "temp/real_speech_test.mp3"
        
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)
        
        # Generate the audio
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_file)
        
        print(f"Generated audio file: {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
        
        # Provide command to play the file
        print("\nTo play the file:")
        print(f"  open {output_file}  # On macOS")
        print(f"  xdg-open {output_file}  # On Linux")
        print(f"  start {output_file}  # On Windows")
        
        return True
    except ImportError:
        print("gTTS not installed. Let's create a fallback script to install it:")
        print("\nTo install gTTS, run:")
        print("  pip install gtts")
        
        # Create a simple audio file
        try:
            print("\nCreating a simple audio file...")
            output_file = "temp/emergency_audio_test.mp3"
            os.makedirs("temp", exist_ok=True)
            
            # Create a minimal MP3 file with repeated tones
            with open(output_file, "wb") as f:
                # This creates a basic MP3 silence pattern
                silence_mp3 = b'\xFF\xFB\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                # Repeat pattern to make a longer file
                for _ in range(500):
                    f.write(silence_mp3)
            
            print(f"Created emergency audio file: {output_file}")
            print(f"File size: {os.path.getsize(output_file)} bytes")
            
            print("\nTo play the file:")
            print(f"  open {output_file}  # On macOS")
            print(f"  xdg-open {output_file}  # On Linux")
            print(f"  start {output_file}  # On Windows")
            
            return True
        except Exception as e:
            print(f"Error creating emergency audio: {str(e)}")
            return False

if __name__ == "__main__":
    generate_audio()