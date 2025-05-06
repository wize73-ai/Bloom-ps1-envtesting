"""
Test script for the text-to-speech functionality of the CasaLingua API.

This script tests the TTS endpoints of the API, including text-to-speech conversion
and voice listing.
"""

import aiohttp
import asyncio
import json
import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create a test configuration with authentication token
API_URL = "http://localhost:8000"
AUTH_HEADER = {"Authorization": "Bearer test_token"}  # Update with a valid token if needed


@pytest.mark.asyncio
async def test_tts_endpoint():
    """Test the TTS endpoint with a simple text input."""
    async with aiohttp.ClientSession() as session:
        # Test text-to-speech endpoint
        tts_data = {
            "text": "This is a test of the text to speech API.",
            "language": "en",
            "voice": None,
            "speed": 1.0,
            "pitch": 1.0,
            "output_format": "mp3"
        }
        
        async with session.post(
            f"{API_URL}/api/pipeline/tts",
            json=tts_data,
            headers=AUTH_HEADER
        ) as response:
            assert response.status == 200
            response_data = await response.json()
            
            # Check response structure
            assert response_data["status"] == "success"
            assert "data" in response_data
            assert "audio_url" in response_data["data"]
            assert "format" in response_data["data"]
            assert response_data["data"]["format"] == "mp3"
            
            # Get the audio URL for further testing
            audio_url = response_data["data"]["audio_url"]
            
            # Check if we can access the audio file
            async with session.get(
                f"{API_URL}{audio_url}",
                headers=AUTH_HEADER
            ) as audio_response:
                assert audio_response.status == 200
                assert audio_response.headers["Content-Type"] == "audio/mpeg"
                
                # We should be able to download the audio content
                audio_content = await audio_response.read()
                assert len(audio_content) > 0


@pytest.mark.asyncio
async def test_tts_voices_endpoint():
    """Test the endpoint for listing available TTS voices."""
    async with aiohttp.ClientSession() as session:
        # Test voice listing endpoint
        async with session.get(
            f"{API_URL}/api/pipeline/tts/voices",
            headers=AUTH_HEADER
        ) as response:
            assert response.status == 200
            response_data = await response.json()
            
            # Check response structure
            assert response_data["status"] == "success"
            assert "data" in response_data
            assert "voices" in response_data["data"]
            
            # Should return a list of voices
            voices = response_data["data"]["voices"]
            assert isinstance(voices, list)
            
            # Each voice should have some basic info
            if voices:
                assert "id" in voices[0]
                assert "language" in voices[0]


@pytest.mark.asyncio
async def test_tts_with_language_parameter():
    """Test TTS with different language parameters."""
    async with aiohttp.ClientSession() as session:
        # Test Spanish text-to-speech
        tts_data = {
            "text": "Hola, esto es una prueba de texto a voz.",
            "language": "es",
            "voice": None,
            "output_format": "mp3"
        }
        
        async with session.post(
            f"{API_URL}/api/pipeline/tts",
            json=tts_data,
            headers=AUTH_HEADER
        ) as response:
            assert response.status == 200
            response_data = await response.json()
            
            # Should return success with Spanish language
            assert response_data["status"] == "success"
            assert response_data["data"]["language"] == "es"


@pytest.mark.asyncio
async def test_tts_with_voice_parameter():
    """Test TTS with specific voice parameter."""
    async with aiohttp.ClientSession() as session:
        # First get available voices
        async with session.get(
            f"{API_URL}/api/pipeline/tts/voices",
            headers=AUTH_HEADER
        ) as response:
            voices_data = await response.json()
            
            # If we have voices, test with a specific voice
            if voices_data["data"]["voices"]:
                test_voice = voices_data["data"]["voices"][0]["id"]
                
                tts_data = {
                    "text": "This is a test with a specific voice.",
                    "language": "en",
                    "voice": test_voice,
                    "output_format": "mp3"
                }
                
                async with session.post(
                    f"{API_URL}/api/pipeline/tts",
                    json=tts_data,
                    headers=AUTH_HEADER
                ) as tts_response:
                    assert tts_response.status == 200
                    tts_result = await tts_response.json()
                    
                    # Should return success with the correct voice
                    assert tts_result["status"] == "success"
                    assert tts_result["data"]["voice"] == test_voice


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_tts_endpoint())
    asyncio.run(test_tts_voices_endpoint())
    asyncio.run(test_tts_with_language_parameter())
    asyncio.run(test_tts_with_voice_parameter())
    
    print("All TTS tests completed successfully!")