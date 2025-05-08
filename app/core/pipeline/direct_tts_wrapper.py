
'''
Direct TTS wrapper that always uses gTTS for speech synthesis
This is a temporary fix to ensure audible speech for all requests
'''

import os
import io
import logging
import time
from typing import Dict, Any

from gtts import gTTS

def process(input_data: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Process function that handles TTS using gTTS directly.
    
    Args:
        input_data: Input data dictionary with text and parameters
        
    Returns:
        Dict with result containing audio bytes
    '''
    # Extract text and parameters
    text = input_data.get('text', '')
    if not text:
        return {"result": b'', "error": "No text provided"}
    
    source_language = input_data.get('source_language', 'en')
    params = input_data.get('parameters', {})
    output_format = params.get('format', 'mp3')
    
    # Make sure we have a 2-letter language code for gTTS
    if len(source_language) > 2 and '-' in source_language:
        source_language = source_language.split('-')[0]
    
    start_time = time.time()
    
    try:
        # Create audio buffer
        buffer = io.BytesIO()
        
        # Create gTTS object and save to buffer
        tts = gTTS(text=text, lang=source_language)
        tts.write_to_fp(buffer)
        
        # Get the audio content
        buffer.seek(0)
        audio_content = buffer.read()
        
        duration = time.time() - start_time
        
        # Return the result
        return {
            "result": audio_content,
            "metadata": {
                "model_used": "gtts_direct",
                "duration": duration,
                "text_length": len(text),
                "language": source_language,
                "format": output_format
            }
        }
    except Exception as e:
        error_msg = f"Error generating speech with gTTS: {str(e)}"
        logging.error(error_msg)
        return {"result": b'', "error": error_msg}
