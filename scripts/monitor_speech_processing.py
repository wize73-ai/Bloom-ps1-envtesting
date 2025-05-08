#!/usr/bin/env python3
"""
Comprehensive monitoring and testing script for speech processing features.
Monitors logs while testing all speech endpoints with detailed verification.
"""

import os
import sys
import time
import json
import uuid
import tempfile
import subprocess
import threading
import requests
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Define base URL for API requests
BASE_URL = "http://localhost:8000"

# Set up logging
LOG_FILE = "logs/speech_testing.log"
SERVER_LOG = "logs/server.log"

def setup_logging():
    """Set up logging directory."""
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write(f"Speech testing log started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def log_message(message):
    """Log a message to the log file and print it."""
    print(message)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{time.strftime('%H:%M:%S')} - {message}\n")

def print_separator(title):
    """Print a separator with a title."""
    separator = f"\n{'='*10} {title} {'='*10}\n"
    log_message(separator)

def run_command(command, background=False):
    """Run a command and return its output."""
    if background:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        return process
    else:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        return result.stdout

def monitor_logs(log_file, stop_event):
    """Monitor a log file and print new content."""
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as f:
            pass  # Create empty file

    # Get the current size of the log file
    try:
        current_size = os.path.getsize(log_file)
    except FileNotFoundError:
        with open(log_file, 'w') as f:
            pass
        current_size = 0

    log_message(f"Monitoring log file: {log_file}")
    
    while not stop_event.is_set():
        try:
            if os.path.exists(log_file):
                new_size = os.path.getsize(log_file)
                if new_size > current_size:
                    # Read and print only the new content
                    with open(log_file, 'r') as f:
                        f.seek(current_size)
                        new_content = f.read()
                        if new_content:
                            for line in new_content.splitlines():
                                # Filter for important log entries related to speech processing
                                if any(keyword in line for keyword in ['tts', 'stt', 'speech', 'audio', 'voice', 'transc']):
                                    log_message(f"LOG: {line}")
                    current_size = new_size
        except Exception as e:
            log_message(f"Error monitoring log: {str(e)}")
        
        # Sleep briefly before checking again
        time.sleep(0.1)

def check_server_status():
    """Check if the server is running."""
    try:
        response = requests.get(f"{BASE_URL}/healthz", timeout=2)
        return response.status_code == 200
    except:
        return False

def create_test_audio_file(text="This is a test audio file for speech transcription", language="en"):
    """Create a test audio file using gTTS or fallback to basic audio."""
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    audio_file = temp_dir / f"test_audio_{uuid.uuid4()}.mp3"
    
    try:
        # Try using gTTS
        try:
            from gtts import gTTS
            log_message(f"Creating audio file with gTTS: '{text}'")
            tts = gTTS(text=text, lang=language)
            tts.save(str(audio_file))
            log_message(f"Created audio file: {audio_file}")
            return audio_file
        except ImportError:
            log_message("gTTS not available, creating fallback audio")
        
        # Fallback to creating a simple MP3
        log_message("Creating basic audio file")
        with open(audio_file, "wb") as f:
            # Simple MP3 header + silence
            silence_mp3 = b'\xFF\xFB\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            f.write(silence_mp3 * 100)
        
        log_message(f"Created fallback audio file: {audio_file}")
        return audio_file
    except Exception as e:
        log_message(f"Error creating test audio file: {str(e)}")
        return None

def test_stt_languages():
    """Test the STT languages endpoint with enhanced verification."""
    print_separator("TESTING STT LANGUAGES ENDPOINT")
    
    url = f"{BASE_URL}/pipeline/stt/languages"
    log_message(f"GET {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url)
        response_time = time.time() - start_time
        
        log_message(f"Response time: {response_time:.3f}s")
        log_message(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Deep verification
            success = True
            issues = []
            
            # Verify response structure
            if "data" not in data:
                success = False
                issues.append("Missing 'data' field in response")
            
            # Verify languages array
            languages = data.get("data", {}).get("languages", [])
            if not languages:
                success = False
                issues.append("No languages found in response")
            
            # Verify language entries
            for lang in languages:
                if "code" not in lang or "name" not in lang:
                    success = False
                    issues.append(f"Language entry missing required fields: {lang}")
            
            # Verify default language
            if "default_language" not in data.get("data", {}):
                success = False
                issues.append("Missing default language")
            
            # Log results
            if success:
                log_message("✅ STT languages endpoint verification PASSED")
                log_message(f"Found {len(languages)} languages")
                # List first few languages
                for lang in languages[:5]:
                    log_message(f"  - {lang.get('code')}: {lang.get('name')}")
                if len(languages) > 5:
                    log_message(f"  ... and {len(languages) - 5} more")
            else:
                log_message("❌ STT languages endpoint verification FAILED")
                for issue in issues:
                    log_message(f"  - {issue}")
            
            return success
        else:
            log_message(f"❌ Error response: {response.text}")
            return False
    except Exception as e:
        log_message(f"❌ Exception testing STT languages: {str(e)}")
        return False

def test_tts_voices():
    """Test the TTS voices endpoint with enhanced verification."""
    print_separator("TESTING TTS VOICES ENDPOINT")
    
    url = f"{BASE_URL}/pipeline/tts/voices"
    log_message(f"GET {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url)
        response_time = time.time() - start_time
        
        log_message(f"Response time: {response_time:.3f}s")
        log_message(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Deep verification
            success = True
            issues = []
            
            # Verify response structure
            if "data" not in data:
                success = False
                issues.append("Missing 'data' field in response")
            
            # Verify voices
            voices = data.get("data", {}).get("voices", [])
            if not voices:
                success = False
                issues.append("No voices found in response")
            
            # Verify voice entries
            for voice in voices:
                required_fields = ["id", "language", "name"]
                for field in required_fields:
                    if field not in voice:
                        success = False
                        issues.append(f"Voice entry missing required field '{field}': {voice}")
            
            # Log results
            if success:
                log_message("✅ TTS voices endpoint verification PASSED")
                log_message(f"Found {len(voices)} voices")
                # List first few voices
                for voice in voices[:5]:
                    log_message(f"  - {voice.get('id')}: {voice.get('name')} ({voice.get('language')})")
                if len(voices) > 5:
                    log_message(f"  ... and {len(voices) - 5} more")
            else:
                log_message("❌ TTS voices endpoint verification FAILED")
                for issue in issues:
                    log_message(f"  - {issue}")
            
            return success
        else:
            log_message(f"❌ Error response: {response.text}")
            return False
    except Exception as e:
        log_message(f"❌ Exception testing TTS voices: {str(e)}")
        return False

def test_tts_endpoint(text="Hello, this is a test of text-to-speech functionality.", language="en"):
    """Test the TTS endpoint with enhanced verification."""
    print_separator("TESTING TTS ENDPOINT")
    
    url = f"{BASE_URL}/pipeline/tts"
    log_message(f"POST {url}")
    log_message(f"Text: '{text}'")
    log_message(f"Language: {language}")
    
    request_data = {
        "text": text,
        "language": language,
        "output_format": "mp3"
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=request_data)
        response_time = time.time() - start_time
        
        log_message(f"Response time: {response_time:.3f}s")
        log_message(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Deep verification
            success = True
            issues = []
            audio_file = None
            
            # Verify response structure
            if "data" not in data:
                success = False
                issues.append("Missing 'data' field in response")
            
            # Verify audio URL
            audio_url = data.get("data", {}).get("audio_url")
            if not audio_url:
                success = False
                issues.append("No audio URL in response")
            else:
                log_message(f"Audio URL: {audio_url}")
                
                # Try to download the audio
                full_audio_url = f"{BASE_URL}{audio_url}"
                log_message(f"Downloading audio from: {full_audio_url}")
                
                audio_response = requests.get(full_audio_url)
                if audio_response.status_code == 200:
                    # Verify audio content
                    audio_content = audio_response.content
                    if len(audio_content) < 10:
                        success = False
                        issues.append(f"Audio file too small: {len(audio_content)} bytes")
                    
                    # Save to a temp file
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    audio_file = temp_dir / f"tts_test_{uuid.uuid4()}.mp3"
                    
                    with open(audio_file, "wb") as f:
                        f.write(audio_content)
                    
                    log_message(f"Downloaded audio to: {audio_file} ({len(audio_content)} bytes)")
                else:
                    success = False
                    issues.append(f"Failed to download audio: {audio_response.status_code}")
            
            # Verify other fields
            required_fields = ["format", "language", "voice", "duration", "text", "model_used"]
            for field in required_fields:
                if field not in data.get("data", {}):
                    success = False
                    issues.append(f"Missing required field '{field}' in response")
            
            # Check if fallback was used
            fallback = data.get("data", {}).get("fallback", False)
            if fallback:
                log_message("⚠️ Fallback mechanism was used for TTS")
            
            # Log results
            if success:
                log_message("✅ TTS endpoint verification PASSED")
                log_message(f"Voice: {data.get('data', {}).get('voice')}")
                log_message(f"Model: {data.get('data', {}).get('model_used')}")
                log_message(f"Duration: {data.get('data', {}).get('duration')} seconds")
            else:
                log_message("❌ TTS endpoint verification FAILED")
                for issue in issues:
                    log_message(f"  - {issue}")
            
            return success, audio_file
        else:
            log_message(f"❌ Error response: {response.text}")
            return False, None
    except Exception as e:
        log_message(f"❌ Exception testing TTS endpoint: {str(e)}")
        return False, None

def test_stt_endpoint(audio_file=None, language=None, expected_text=None):
    """Test the STT endpoint with enhanced verification."""
    print_separator("TESTING STT ENDPOINT")
    
    # Create audio file if not provided
    if audio_file is None:
        text = expected_text or "This is a test audio file for speech recognition"
        audio_file = create_test_audio_file(text, language or "en")
        if audio_file is None:
            log_message("❌ Failed to create test audio file")
            return False
    
    url = f"{BASE_URL}/pipeline/stt"
    log_message(f"POST {url} (multipart/form-data)")
    log_message(f"Audio file: {audio_file}")
    log_message(f"Language: {language or 'auto-detect'}")
    
    # Prepare form data
    form_data = {}
    if language:
        form_data["language"] = language
        form_data["detect_language"] = "false"
    else:
        form_data["detect_language"] = "true"
    
    try:
        with open(audio_file, "rb") as f:
            files = {
                "audio_file": (os.path.basename(audio_file), f, "audio/mpeg")
            }
            
            # Send request
            start_time = time.time()
            response = requests.post(url, data=form_data, files=files)
            response_time = time.time() - start_time
            
            log_message(f"Response time: {response_time:.3f}s")
            log_message(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Deep verification
                success = True
                issues = []
                
                # Verify response structure
                if "data" not in data:
                    success = False
                    issues.append("Missing 'data' field in response")
                
                # Verify transcription text
                transcribed_text = data.get("data", {}).get("text", "")
                if not transcribed_text and not data.get("data", {}).get("fallback", False):
                    success = False
                    issues.append("No transcription text in response")
                
                # Verify other fields
                required_fields = ["language", "confidence", "model_used"]
                for field in required_fields:
                    if field not in data.get("data", {}):
                        success = False
                        issues.append(f"Missing required field '{field}' in response")
                
                # Verify language detection
                detected_language = data.get("data", {}).get("language")
                if not detected_language:
                    success = False
                    issues.append("No language detected")
                
                # Check if fallback was used
                fallback = data.get("data", {}).get("fallback", False)
                if fallback:
                    log_message("⚠️ Fallback mechanism was used for STT")
                
                # Compare with expected text if provided
                if expected_text and transcribed_text:
                    # Simple similarity check
                    similarity = calculate_text_similarity(transcribed_text, expected_text)
                    log_message(f"Text similarity: {similarity:.2f}")
                    
                    if similarity < 0.3 and not fallback:
                        log_message("⚠️ Low similarity to expected text")
                
                # Log results
                if success:
                    log_message("✅ STT endpoint verification PASSED")
                    log_message(f"Transcribed text: '{transcribed_text}'")
                    log_message(f"Language: {detected_language}")
                    log_message(f"Model: {data.get('data', {}).get('model_used')}")
                    log_message(f"Confidence: {data.get('data', {}).get('confidence')}")
                else:
                    log_message("❌ STT endpoint verification FAILED")
                    for issue in issues:
                        log_message(f"  - {issue}")
                
                return success
            else:
                log_message(f"❌ Error response: {response.text}")
                return False
    except Exception as e:
        log_message(f"❌ Exception testing STT endpoint: {str(e)}")
        return False

def test_end_to_end_speech_workflow():
    """Test the complete speech workflow: TTS → Audio File → STT."""
    print_separator("TESTING END-TO-END SPEECH WORKFLOW")
    
    # Step 1: Text to generate
    original_text = "This is a comprehensive test of the speech processing workflow."
    language = "en"
    log_message(f"Original text: '{original_text}'")
    log_message(f"Language: {language}")
    
    # Step 2: Convert text to speech
    log_message("\n[STEP 1] Converting text to speech...")
    tts_success, audio_file = test_tts_endpoint(original_text, language)
    
    if not tts_success or audio_file is None:
        log_message("❌ END-TO-END TEST FAILED: Could not generate audio")
        return False
    
    # Step 3: Convert speech back to text
    log_message("\n[STEP 2] Converting speech back to text...")
    stt_success = test_stt_endpoint(audio_file, language, original_text)
    
    # Log final result
    if tts_success and stt_success:
        log_message("✅ END-TO-END SPEECH WORKFLOW TEST PASSED")
        return True
    else:
        log_message("❌ END-TO-END SPEECH WORKFLOW TEST FAILED")
        log_message(f"  - TTS Success: {tts_success}")
        log_message(f"  - STT Success: {stt_success}")
        return False

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two texts."""
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    if not words1 and not words2:
        return 1.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)

def print_test_summary(results):
    """Print a summary of test results."""
    print_separator("TEST SUMMARY")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        log_message(f"{name}: {status}")
    
    log_message(f"\nPassed {passed}/{total} tests ({passed/total*100:.1f}%)")
    
    if passed < total:
        log_message("\nSuggestions for failed tests:")
        log_message("1. Check server logs for detailed error messages")
        log_message("2. Run 'python scripts/install_tts_requirements.py' to install dependencies")
        log_message("3. Check network connectivity for external model access")
        log_message("4. Verify configuration settings for speech processing")

def check_system_health():
    """Perform system health checks before running tests."""
    print_separator("SYSTEM HEALTH CHECK")
    
    health_issues = []
    
    # Check server status
    log_message("Checking server status...")
    if not check_server_status():
        health_issues.append("Server is not running")
    
    # Check required directories
    for directory in ["temp", "logs"]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            log_message(f"Created missing directory: {directory}")
    
    # Check for gtts installation
    try:
        import gtts
        log_message("✅ gTTS is installed")
    except ImportError:
        log_message("⚠️ gTTS is not installed - audio generation will use fallbacks")
        health_issues.append("gTTS not installed")
    
    # Check for pydub installation
    try:
        import pydub
        log_message("✅ pydub is installed")
    except ImportError:
        log_message("⚠️ pydub is not installed - audio processing will be limited")
        health_issues.append("pydub not installed")
    
    # Log results
    if health_issues:
        log_message("\n⚠️ System health check found issues:")
        for issue in health_issues:
            log_message(f"  - {issue}")
        return False
    else:
        log_message("✅ System health check passed")
        return True

def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Monitor and test speech processing endpoints")
    parser.add_argument("--no-monitor", action="store_true", help="Don't monitor server logs")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    print_separator("SPEECH PROCESSING TEST SUITE")
    
    # Start log monitoring in a separate thread
    stop_event = threading.Event()
    if not args.no_monitor:
        log_message("Starting log monitoring...")
        monitor_thread = threading.Thread(
            target=monitor_logs,
            args=(SERVER_LOG, stop_event)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
    
    try:
        # Perform system health check
        health_ok = check_system_health()
        if not health_ok:
            log_message("⚠️ Continuing tests despite health check issues")
        
        # Run all tests
        results = {}
        
        # Test STT languages endpoint
        results["STT Languages"] = test_stt_languages()
        
        # Test TTS voices endpoint
        results["TTS Voices"] = test_tts_voices()
        
        # Test TTS endpoint
        tts_success, audio_file = test_tts_endpoint()
        results["TTS Endpoint"] = tts_success
        
        # Test STT endpoint
        results["STT Endpoint"] = test_stt_endpoint()
        
        # Only run end-to-end test if not in quick mode
        if not args.quick:
            # Test end-to-end workflow
            results["End-to-End Workflow"] = test_end_to_end_speech_workflow()
        
        # Print test summary
        print_test_summary(results)
        
    except KeyboardInterrupt:
        log_message("Testing interrupted by user")
    except Exception as e:
        log_message(f"Error during testing: {str(e)}")
    finally:
        # Stop log monitoring
        stop_event.set()
        log_message("Testing complete")

if __name__ == "__main__":
    main()