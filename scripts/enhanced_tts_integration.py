#!/usr/bin/env python3
"""
Enhanced TTS integration script to create proper audio files for the TTS endpoint.
This script adds gTTS integration to the TTS fallback mechanism.
"""

import os
import sys
import time
import uuid
import requests
import json
from pathlib import Path

# Base URL for API
BASE_URL = "http://localhost:8000"

def ensure_directory(path):
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)
    return path

def test_tts_with_gtts(text="Hello, this is a test of the enhanced text to speech system with gTTS integration.", language="en"):
    """Test TTS with improved gTTS integration."""
    print(f"Converting text to speech: '{text}'")
    
    # First, create a local gTTS file
    try:
        from gtts import gTTS
        
        # Generate a local file with gTTS
        temp_dir = ensure_directory("temp")
        local_file = Path(temp_dir) / f"gtts_test_{uuid.uuid4()}.mp3"
        
        print(f"Generating local gTTS file...")
        tts = gTTS(text=text, lang=language)
        tts.save(str(local_file))
        
        print(f"Created local audio file: {local_file}")
        print(f"File size: {os.path.getsize(local_file)} bytes")
        
        # Now test the TTS endpoint
        print("\nTesting TTS endpoint...")
        url = f"{BASE_URL}/pipeline/tts"
        
        # Prepare request data
        request_data = {
            "text": text,
            "language": language,
            "output_format": "mp3"
        }
        
        # Send request
        response = requests.post(url, json=request_data)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Get audio URL
            audio_url = data.get("data", {}).get("audio_url")
            if audio_url:
                print(f"Audio URL: {audio_url}")
                
                # Download the audio
                full_audio_url = f"{BASE_URL}{audio_url}"
                print(f"Downloading audio from: {full_audio_url}")
                
                audio_response = requests.get(full_audio_url)
                if audio_response.status_code == 200:
                    # Save to a temp file
                    api_file = Path(temp_dir) / f"api_tts_{uuid.uuid4()}.mp3"
                    
                    with open(api_file, "wb") as f:
                        f.write(audio_response.content)
                    
                    print(f"API audio file saved to: {api_file}")
                    print(f"API audio file size: {os.path.getsize(api_file)} bytes")
                    
                    # Compare file sizes
                    local_size = os.path.getsize(local_file)
                    api_size = os.path.getsize(api_file)
                    
                    print(f"\nLocal gTTS file size: {local_size} bytes")
                    print(f"API file size: {api_size} bytes")
                    
                    if api_size < 2000:
                        print("❌ API file is too small, likely using fallback audio")
                    else:
                        print("✅ API file size looks reasonable")
                    
                    # Play instructions
                    print("\nTo compare the files:")
                    print(f"Local gTTS file: open {local_file}")
                    print(f"API TTS file: open {api_file}")
                    
                    return local_file, api_file
                else:
                    print(f"Error downloading audio: {audio_response.status_code}")
            else:
                print("No audio URL in response")
        else:
            print(f"Error with TTS request: {response.text}")
        
        return local_file, None
        
    except ImportError:
        print("gTTS not installed. Install with: pip install gtts")
        return None, None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None

def integrate_gtts_with_tts_pipeline():
    """Integrate gTTS with the TTS pipeline for better fallbacks."""
    print("Integrating gTTS with TTS pipeline...")
    
    # Check if gTTS is installed
    try:
        import gtts
        print("✅ gTTS is installed")
    except ImportError:
        print("❌ gTTS is not installed. Install with: pip install gtts")
        return False
    
    # Create an enhanced fallback implementation in the TTS pipeline
    tts_file = "app/core/pipeline/tts.py"
    
    if not os.path.exists(tts_file):
        print(f"❌ TTS pipeline file not found: {tts_file}")
        return False
    
    # Read the file
    with open(tts_file, "r") as f:
        content = f.read()
    
    # Check if we already integrated gTTS
    if "except ImportError" in content and "tts = gTTS" in content:
        print("✅ gTTS already integrated with TTS pipeline")
        return True
    
    # Find the place to add gTTS integration (in the fallback synthesis method)
    fallback_section = """            # Try gTTS as a last resort
            try:
                # Import gTTS only if needed
                from gtts import gTTS
                
                # Create output file
                if not output_path:
                    audio_id = str(uuid.uuid4())
                    output_path = str(self.temp_dir / f"tts_gtts_{audio_id}.mp3")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Map language code to gTTS format if needed
                gtts_lang = language
                if len(language) > 2 and '-' in language:
                    gtts_lang = language.split('-')[0]
                
                # Create gTTS object
                tts = gTTS(text=text, lang=gtts_lang, slow=(speed < 0.8))
                
                # Save to file
                tts.save(output_path)"""
    
    if fallback_section in content:
        print("✅ gTTS implementation already exists in fallback synthesis")
        
        # Now let's verify if it's properly triggered
        try_section = """            # Try to use alternative TTS model
            fallback_model_type = "tts_fallback"
            """
    
        # Make sure fallback immediately uses gTTS if available
        improved_section = """            # Try to use alternative TTS model or gTTS directly
            try:
                # Try gTTS first since it's more reliable
                from gtts import gTTS
                
                # Create output file
                if not output_path:
                    audio_id = str(uuid.uuid4())
                    output_path = str(self.temp_dir / f"tts_gtts_{audio_id}.{output_format}")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Map language code to gTTS format if needed
                gtts_lang = language
                if len(language) > 2 and '-' in language:
                    gtts_lang = language.split('-')[0]
                
                logger.info(f"Using gTTS for text-to-speech with language '{gtts_lang}'")
                
                # Create gTTS object
                tts = gTTS(text=text, lang=gtts_lang, slow=(speed < 0.8))
                
                # Save to file
                tts.save(output_path)
                
                # Read audio content
                with open(output_path, "rb") as f:
                    audio_content = f.read()
                
                # Get audio info
                audio_info = self._get_audio_info(output_path)
                
                return {
                    "audio_file": output_path,
                    "audio_content": audio_content,
                    "format": output_format,
                    "language": language,
                    "voice": voice,
                    "duration": audio_info.get("duration", 3.0),
                    "model_used": "gtts",
                    "fallback": True
                }
            except ImportError:
                logger.info("gTTS not available, trying fallback model")
                pass
            except Exception as gtts_e:
                logger.error(f"Error using gTTS: {str(gtts_e)}", exc_info=True)
                
            # Try fallback model as backup
            fallback_model_type = "tts_fallback"
            """
        
        # Replace the section only if it's not already enhanced
        if try_section in content and "Try gTTS first since it's more reliable" not in content:
            updated_content = content.replace(try_section, improved_section)
            
            # Write back the updated file
            with open(tts_file, "w") as f:
                f.write(updated_content)
            
            print("✅ Enhanced gTTS integration to prioritize it in fallbacks")
            return True
        else:
            print("✅ gTTS already prioritized in fallbacks")
            return True
    else:
        print("❌ Could not find fallback synthesis section to enhance")
        return False

def main():
    """Main function to test enhanced TTS integration."""
    print("===== ENHANCED TTS INTEGRATION =====")
    
    # Step 1: Integrate gTTS with TTS pipeline
    integrate_gtts_with_tts_pipeline()
    
    # Step 2: Test TTS with gTTS
    local_file, api_file = test_tts_with_gtts()
    
    # Step 3: Provide final instructions
    if local_file and api_file:
        print("\n===== TESTING COMPLETE =====")
        print("Two audio files have been generated:")
        print(f"1. Local gTTS file: {local_file}")
        print(f"2. API TTS file: {api_file}")
        print("\nTo play these files:")
        print(f"  open {local_file}  # Local gTTS file")
        print(f"  open {api_file}    # API TTS file")
    else:
        print("\n===== TESTING INCOMPLETE =====")
        print("Some errors occurred during testing.")
        print("Please check the logs for details.")

if __name__ == "__main__":
    main()