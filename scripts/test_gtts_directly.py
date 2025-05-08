#\!/usr/bin/env python3
"""
Direct test of gTTS functionality.
This script tests gTTS directly to ensure it can generate audible audio.
"""

import os
import sys
import time
from gtts import gTTS

def test_gtts_directly():
    """Test gTTS directly."""
    # Create output directory
    os.makedirs("temp/gtts_test", exist_ok=True)
    
    # Test cases
    test_cases = [
        {
            "name": "english_simple",
            "text": "This is a test message created directly with gTTS.",
            "language": "en"
        },
        {
            "name": "spanish_simple",
            "text": "Este es un mensaje de prueba creado directamente con gTTS.",
            "language": "es"
        }
    ]
    
    print("===== DIRECT GTTS TEST =====")
    
    # Test each case
    for case in test_cases:
        print(f"\nTesting case: {case['name']}")
        
        # Output file
        output_file = f"temp/gtts_test/direct_gtts_{case['name']}.mp3"
        
        # Generate audio
        print(f"Generating audio for: '{case['text'][:30]}...'")
        try:
            # Create gTTS object
            tts = gTTS(text=case['text'], lang=case['language'])
            
            # Save to file
            tts.save(output_file)
            
            # Get file size
            file_size = os.path.getsize(output_file)
            
            print(f"✅ Success\! Output saved to {output_file}")
            print(f"   File size: {file_size} bytes")
            
            # Suggest command to play the file
            print(f"   To play: open {output_file}")
        except Exception as e:
            print(f"❌ Error generating audio: {str(e)}")
    
    print("\n===== TEST COMPLETE =====")
    print("Test files have been saved to temp/gtts_test directory.")
    print("You can play these files to verify they contain audible speech.")

if __name__ == "__main__":
    test_gtts_directly()
