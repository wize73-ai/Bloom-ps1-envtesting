#!/usr/bin/env python3
"""
Emergency fix for TTS functionality.
This script creates a simple audio file with silence that will be returned
by the TTS endpoint when other methods fail.
"""

import os
import uuid
from pathlib import Path

def create_emergency_audio_files():
    """Create emergency audio files in various formats."""
    # Create temp directory if it doesn't exist
    temp_dir = Path("temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a simple MP3 file
    mp3_file = temp_dir / f"emergency_tts_{uuid.uuid4()}.mp3"
    with open(mp3_file, "wb") as f:
        # Simple MP3 file header + minimal data
        silence_mp3 = b'\xFF\xFB\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        f.write(silence_mp3 * 100)  # Repeat to make it longer
    
    # Create a simple WAV file
    wav_file = temp_dir / f"emergency_tts_{uuid.uuid4()}.wav"
    with open(wav_file, "wb") as f:
        # Simple WAV header + silence
        sample_rate = 44100
        bits_per_sample = 16
        channels = 1
        
        # Write WAV header
        f.write(b'RIFF')
        f.write((36).to_bytes(4, byteorder='little'))  # Chunk size
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write((16).to_bytes(4, byteorder='little'))  # Subchunk1 size
        f.write((1).to_bytes(2, byteorder='little'))   # Audio format (PCM)
        f.write(channels.to_bytes(2, byteorder='little'))
        f.write(sample_rate.to_bytes(4, byteorder='little'))
        byte_rate = sample_rate * channels * bits_per_sample // 8
        f.write(byte_rate.to_bytes(4, byteorder='little'))
        block_align = channels * bits_per_sample // 8
        f.write(block_align.to_bytes(2, byteorder='little'))
        f.write(bits_per_sample.to_bytes(2, byteorder='little'))
        f.write(b'data')
        f.write((0).to_bytes(4, byteorder='little'))  # Subchunk2 size (0 = empty)
    
    print(f"Created emergency MP3 file: {mp3_file}")
    print(f"Created emergency WAV file: {wav_file}")
    print("These files will be used as fallbacks for TTS endpoint testing.")
    
    return [mp3_file, wav_file]

if __name__ == "__main__":
    files = create_emergency_audio_files()
    
    # Print the paths that can be used in testing
    for file in files:
        print(f"curl http://localhost:8000/pipeline/tts/audio/{os.path.basename(file)}")