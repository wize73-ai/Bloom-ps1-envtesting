# Speech Processing API

This document outlines the architecture, endpoints, and implementation details for the Speech Processing features in CasaLingua, including Text-to-Speech (TTS) and Speech-to-Text (STT).

## Overview

CasaLingua provides a comprehensive speech processing API with the following capabilities:

1. **Text-to-Speech (TTS)**: Convert text to synthesized speech in multiple languages and voices
2. **Speech-to-Text (STT)**: Transcribe audio recordings to text with language detection

Both systems include multi-level fallback mechanisms to ensure service continuity even when underlying models fail, making them suitable for production environments.

## API Endpoints

### 1. Text-to-Speech (TTS)

#### 1.1 Convert Text to Speech

```
POST /pipeline/tts
```

Converts text to speech in the specified language and voice.

**Request Body:**

```json
{
  "text": "Text to be converted to speech",
  "language": "en",
  "voice": "en-us-1",
  "speed": 1.0,
  "pitch": 1.0,
  "output_format": "mp3"
}
```

**Parameters:**

- `text`: (required) Text to convert to speech
- `language`: (optional) Language code (e.g., "en", "es"), defaults to "en"
- `voice`: (optional) Voice identifier, defaults to language-specific default
- `speed`: (optional) Speech rate multiplier (0.5-2.0), defaults to 1.0
- `pitch`: (optional) Voice pitch adjustment (0.5-2.0), defaults to 1.0
- `output_format`: (optional) Output audio format ("mp3", "wav", "ogg"), defaults to "mp3"

**Response:**

```json
{
  "status": "success",
  "message": "Text-to-speech synthesis completed successfully",
  "data": {
    "audio_url": "/pipeline/tts/audio/abc123.mp3",
    "format": "mp3",
    "language": "en",
    "voice": "en-us-1",
    "duration": 2.5,
    "text": "Text to be converted to speech",
    "model_used": "tts",
    "processing_time": 0.523,
    "fallback": false
  },
  "metadata": {
    "request_id": "uuid",
    "timestamp": "ISO date",
    "process_time": 0.523
  }
}
```

#### 1.2 Get Audio File

```
GET /pipeline/tts/audio/{file_name}
```

Retrieves a generated audio file by its file name.

**Parameters:**

- `file_name`: (required) The file name from the audio_url in the TTS response

**Response:**

The audio file content with appropriate content type header.

#### 1.3 List Available Voices

```
GET /pipeline/tts/voices
```

Returns a list of available voices for TTS.

**Query Parameters:**

- `language`: (optional) Filter voices by language code

**Response:**

```json
{
  "status": "success",
  "message": "Available TTS voices retrieved successfully",
  "data": {
    "voices": [
      {
        "id": "en-us-1",
        "language": "en",
        "name": "English Voice 1",
        "gender": "male"
      },
      {
        "id": "es-es-1",
        "language": "es",
        "name": "Spanish Voice 1",
        "gender": "male"
      },
      ...
    ],
    "default_voice": "en-us-1"
  },
  "metadata": {
    "request_id": "uuid",
    "timestamp": "ISO date"
  }
}
```

### 2. Speech-to-Text (STT)

#### 2.1 Transcribe Audio

```
POST /pipeline/stt
```

Converts speech in an audio file to text.

**Form Data:**

- `audio_file`: (required) Audio file to transcribe
- `language`: (optional) Language code (e.g., "en", "es")
- `detect_language`: (optional) Boolean to auto-detect language
- `model_id`: (optional) Specific model to use
- `enhanced_results`: (optional) Include enhanced results

**Response:**

```json
{
  "status": "success",
  "message": "Speech transcription completed successfully",
  "data": {
    "text": "Transcribed text from audio",
    "language": "en",
    "confidence": 0.87,
    "segments": [...],
    "duration": 3.5,
    "model_used": "speech_to_text",
    "processing_time": 0.63,
    "audio_format": "mp3",
    "fallback": false
  },
  "metadata": {
    "request_id": "uuid",
    "timestamp": "ISO date",
    "process_time": 0.63
  }
}
```

#### 2.2 List Supported Languages

```
GET /pipeline/stt/languages
```

Returns a list of supported languages for STT.

**Response:**

```json
{
  "status": "success",
  "message": "Supported STT languages retrieved successfully",
  "data": {
    "languages": [
      {
        "code": "en",
        "name": "English"
      },
      {
        "code": "es",
        "name": "Spanish"
      },
      ...
    ],
    "default_language": "en"
  },
  "metadata": {
    "request_id": "uuid",
    "timestamp": "ISO date"
  }
}
```

## Implementation Architecture

### Components

The speech processing system consists of the following components:

1. **API Routes**: FastAPI routes defined in `app/api/routes/pipeline.py`
2. **Pipeline Modules**:
   - Text-to-Speech: `app/core/pipeline/tts.py`
   - Speech-to-Text: `app/core/pipeline/stt.py`
3. **Schema Definitions**:
   - TTS schemas: `app/api/schemas/tts.py`
   - STT schemas: `app/api/schemas/speech.py`
4. **Model Management**:
   - Model Manager: `app/services/models/manager.py`
   - Model Wrapper: `app/services/models/wrapper.py`
   - Model Loader: `app/services/models/loader.py`

### Fallback Mechanisms

The system implements a multi-level fallback architecture to ensure high availability:

1. **Level 1**: Primary ML models for TTS/STT
2. **Level 2**: Alternative ML models (e.g., smaller or less complex)
3. **Level 3**: Basic implementations (e.g., gTTS for TTS)
4. **Level 4**: Emergency fallbacks (e.g., generating silent audio files)

This ensures that API requests always receive a valid response, even when underlying components fail.

### Error Handling

The system implements robust error handling:

1. **Exception Catching**: All operations are wrapped in try/except blocks
2. **Graceful Degradation**: Falls back to simpler models when primary models fail
3. **Emergency Responses**: Provides basic responses even when all fallbacks fail
4. **Detailed Logging**: All errors are logged with context information
5. **Metrics Collection**: Failures are tracked for monitoring

### Caching

Audio files are cached for improved performance:

1. **File Caching**: Generated audio files are stored in the filesystem
2. **HTTP Caching**: HTTP headers are set for client-side caching
3. **Cache Cleaning**: Automatic cleanup of old cache files to manage disk space

## Security Considerations

### Input Validation

1. **Text Validation**: Limits on text length, format validation
2. **Audio Validation**: File size limits, format checking
3. **Parameter Validation**: Type checking, range validation for numeric parameters

### File Safety

1. **Path Traversal Prevention**: Sanitization of file names
2. **Content Type Verification**: Validation of audio file formats
3. **Secure File Handling**: Safe creation and deletion of temporary files

## Testing

### Test Scripts

Comprehensive test scripts are provided to verify functionality:

1. `scripts/monitor_speech_processing.py`: Monitors logs while testing all endpoints
2. `scripts/test_tts_curl.sh`: Simple curl-based test for TTS endpoint
3. `scripts/test_speech_workflow.py`: End-to-end test of speech workflow

### Test Cases

1. **Basic Functionality**: Verifies all endpoints return expected responses
2. **Error Handling**: Tests system responses to invalid inputs
3. **Robustness**: Verifies fallback mechanisms trigger appropriately
4. **End-to-End**: Confirms complete workflow from text to speech to text
5. **Performance**: Checks response times and resource usage

## Recent Fixes and Improvements

1. **Fixed UnboundLocalError**: Resolved an issue where `audio_content` was referenced before assignment
2. **Enhanced Method Registration**: Added proper method registration for speech-related methods
3. **Improved Fallback System**: Created more robust emergency fallbacks
4. **Fixed Validation Errors**: Ensured voice parameters are never None
5. **Enhanced Error Handling**: Improved detection and handling of model errors
6. **Added Security Validation**: Implemented file name validation and input sanitization
7. **Optimized Caching**: Added HTTP caching headers for better performance
8. **Added Comprehensive Testing**: Created thorough test scripts for verification

## Example Usage

### Text-to-Speech

```bash
curl -X POST http://localhost:8000/pipeline/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the text to speech system.",
    "language": "en",
    "output_format": "mp3"
  }'
```

### Speech-to-Text

```bash
curl -X POST http://localhost:8000/pipeline/stt \
  -F "audio_file=@speech.mp3" \
  -F "language=en" \
  -F "detect_language=false"
```

## Conclusion

The speech processing system provides robust, fault-tolerant APIs for text-to-speech and speech-to-text functionality. With multiple fallback levels and comprehensive error handling, it ensures high availability even when underlying components fail.

Future improvements could include adding more sophisticated TTS/STT models, expanding language support, and implementing streaming capabilities for real-time speech processing.