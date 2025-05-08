#!/bin/bash
# Test TTS with curl to see detailed error output

echo "Testing TTS endpoint with curl..."
curl -v -X POST \
  http://localhost:8000/pipeline/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test message", "language": "en", "output_format": "mp3"}' \
  2>&1