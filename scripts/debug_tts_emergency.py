#\!/usr/bin/env python3
"""
Debug script for emergency TTS audio generation.
This script directly tests the emergency audio generation methods.
"""

import os
import sys
import json
import time
import io
import requests
from pathlib import Path

# Add the app directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gtts_direct():
    """Test gtts directly to verify it works."""
    print("===== TESTING GTTS DIRECTLY =====")
    try:
        from gtts import gTTS
        
        # Create output directory
        os.makedirs("temp/debug", exist_ok=True)
        
        # Test text
        test_text = "This is a test of the emergency audio system, generated directly by gTTS."
        output_file = "temp/debug/direct_gtts_test.mp3"
        
        print(f"Creating gTTS audio for text: '{test_text}'")
        tts = gTTS(text=test_text, lang="en")
        tts.save(output_file)
        
        filesize = os.path.getsize(output_file)
        print(f"✅ Direct gTTS test successful\!")
        print(f"   Created file: {output_file}")
        print(f"   File size: {filesize} bytes")
        print(f"   To play: open {output_file}")
        
        return True
    except Exception as e:
        print(f"❌ Direct gTTS test failed: {str(e)}")
        return False

def test_emergency_audio_function():
    """Attempt to test the emergency audio function directly."""
    print("\n===== TESTING EMERGENCY AUDIO FUNCTION DIRECTLY =====")
    try:
        # Import our TTS module
        from app.core.pipeline.tts import TTSPipeline
        
        print("Creating minimal TTSPipeline instance...")
        
        # We need to mock the model_manager
        class MockModelManager:
            async def load_model(self, *args, **kwargs):
                return None
                
            async def run_model(self, *args, **kwargs):
                raise Exception("Mock error to trigger emergency audio")
        
        # Create a minimal TTSPipeline instance
        tts_pipeline = TTSPipeline(MockModelManager())
        
        # Call _create_emergency_audio directly
        print("Calling _create_emergency_audio...")
        audio_content = tts_pipeline._create_emergency_audio("mp3")
        
        # Save the output
        output_file = "temp/debug/emergency_direct_call.mp3"
        with open(output_file, "wb") as f:
            f.write(audio_content)
        
        filesize = os.path.getsize(output_file)
        print(f"✅ Direct emergency audio call successful\!")
        print(f"   Created file: {output_file}")
        print(f"   File size: {filesize} bytes")
        print(f"   Audio content length: {len(audio_content)} bytes")
        print(f"   To play: open {output_file}")
        
        if filesize < 2000:
            print(f"⚠️  WARNING: File size is very small ({filesize} bytes)")
            print("   This might be a silent or corrupted audio file.")
            
            # Check if it's the silent MP3 from the original implementation
            silent_mp3_header = b'\xFF\xFB\x10\x00'
            if audio_content.startswith(silent_mp3_header):
                print("   File appears to be using the original silent MP3 format.")
                print("   Fix may not have been applied correctly to the _create_emergency_audio method.")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("   This test needs to be run in the same environment as the server.")
        return False
    except Exception as e:
        print(f"❌ Direct emergency audio test failed: {str(e)}")
        return False

def force_emergency_audio_via_endpoint():
    """Force emergency audio generation via the endpoint by causing errors."""
    print("\n===== FORCING EMERGENCY AUDIO VIA ENDPOINT =====")
    
    # Create output directory
    os.makedirs("temp/debug", exist_ok=True)
    
    # Base URL
    base_url = "http://localhost:8000"
    
    # Create a request that will likely cause the model to fail
    # Use an extremely long text that will cause issues
    request_data = {
        "text": "This is an extremely long text " * 1000, # Make it very long
        "language": "xx",  # Invalid language to force fallback
        "output_format": "mp3",
        "voice": "invalid-voice"  # Invalid voice to force more fallbacks
    }
    
    # Output file
    output_file = "temp/debug/forced_emergency.mp3"
    
    # Make request
    print("Making request with invalid parameters to force emergency audio...")
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/pipeline/tts",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        request_time = time.time() - start_time
        
        # Check response
        if response.status_code == 200:
            # Save audio file
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            # Get file size
            file_size = os.path.getsize(output_file)
            
            print(f"✅ Success\! Output saved to {output_file}")
            print(f"   File size: {file_size} bytes")
            print(f"   Request time: {request_time:.2f} seconds")
            
            # Check if file is too small (potential silent file)
            if file_size < 2000:
                print(f"⚠️  WARNING: File size is very small ({file_size} bytes)")
                print("   This might be a silent or corrupted audio file.")
                
                # Check if it's the silent MP3 from the original implementation
                with open(output_file, "rb") as f:
                    content = f.read(20)  # Read first 20 bytes
                    
                silent_mp3_header = b'\xFF\xFB\x10\x00'
                if content.startswith(silent_mp3_header):
                    print("   File appears to be using the original silent MP3 format.")
                    print("   Fix may not have been applied correctly or server not restarted.")
            else:
                print(f"   File size looks good ({file_size} bytes)")
            
            # Suggest command to play the file
            print(f"   To play: open {output_file}")
        else:
            print(f"❌ Error: Status code {response.status_code}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Error making request: {str(e)}")

def main():
    """Main function."""
    print("===== EMERGENCY TTS AUDIO DEBUG =====\n")
    
    # Test gTTS directly
    gtts_ok = test_gtts_direct()
    
    # Test emergency audio function directly
    emergency_ok = test_emergency_audio_function()
    
    # Force emergency audio via endpoint
    force_emergency_audio_via_endpoint()
    
    print("\n===== DEBUG COMPLETE =====")
    if not gtts_ok:
        print("❌ Direct gTTS test failed - gTTS may not be installed correctly")
    else:
        print("✅ Direct gTTS test succeeded - gTTS is working correctly")
        
    if not emergency_ok:
        print("❌ Direct emergency audio function test failed - may not be accessible")
    
    print("\nCheck the generated files in temp/debug directory to see which ones have audible audio.")
    print("Compare the file sizes and content:")
    print("- If direct_gtts_test.mp3 is large (>20KB) but others are small (<2KB): Fix not applied")
    print("- If all files are small: gTTS may not be generating audio correctly")
    print("- If emergency_direct_call.mp3 is large but forced_emergency.mp3 is small: Endpoint issue")

if __name__ == "__main__":
    main()
