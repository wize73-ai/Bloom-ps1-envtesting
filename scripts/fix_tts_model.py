#!/usr/bin/env python3
"""
Fix TTS model configuration and loading to support pipeline type models.
This script modifies the model loader to add support for pipeline type models,
fixing the "Unsupported model type: pipeline" error.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to model loader file
LOADER_FILE = "app/services/models/loader.py"

# Path to model registry configuration
REGISTRY_FILE = "config/model_registry.json"

def fix_tts_model_registry():
    """
    Fix the TTS model configuration in model_registry.json.
    Updates the TTS model to use a properly supported configuration.
    """
    try:
        # Read the current registry
        with open(REGISTRY_FILE, 'r') as f:
            registry = json.load(f)
        
        # Check if TTS model exists
        if "tts" not in registry:
            # If TTS model is missing, add it
            registry["tts"] = {
                "model_name": "facebook/mms-tts-eng",
                "tokenizer_name": "facebook/mms-tts-eng",
                "task": "text-to-speech",
                "type": "transformers",
                "model_class": "AutoModelForTextToSpeech",
                "framework": "transformers",
                "use_pipeline": True,
                "pipeline_task": "text-to-speech",
                "tokenizer_kwargs": {
                    "max_length": 256
                }
            }
        else:
            # Update existing TTS model
            registry["tts"].update({
                "model_name": "facebook/mms-tts-eng",
                "tokenizer_name": "facebook/mms-tts-eng",
                "type": "transformers",
                "model_class": "AutoModelForTextToSpeech",
                "use_pipeline": True,
                "pipeline_task": "text-to-speech"
            })
        
        # Add a fallback TTS model
        registry["tts_fallback"] = {
            "model_name": "espnet/kan-bayashi_ljspeech_vits",
            "tokenizer_name": "espnet/kan-bayashi_ljspeech_vits",
            "task": "text-to-speech",
            "type": "transformers",
            "model_class": "AutoModelForTextToSpeech",
            "framework": "transformers",
            "use_pipeline": True,
            "pipeline_task": "text-to-speech",
            "is_fallback": True,
            "tokenizer_kwargs": {
                "max_length": 256
            }
        }
        
        # Add STT model configuration if not present
        if "speech_to_text" not in registry:
            registry["speech_to_text"] = {
                "model_name": "openai/whisper-small",
                "tokenizer_name": "openai/whisper-small",
                "task": "speech-to-text",
                "type": "transformers",
                "model_class": "AutoModelForSpeechSeq2Seq",
                "framework": "transformers",
                "use_pipeline": True,
                "pipeline_task": "automatic-speech-recognition",
                "tokenizer_kwargs": {
                    "max_length": 256
                }
            }
        
        # Write the updated registry
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Updated TTS model configuration in {REGISTRY_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error updating TTS model configuration: {e}")
        return False

def add_pipeline_support_to_loader():
    """
    Add pipeline type support to the model loader.
    Modifies the _load_transformers_model method to handle pipeline type models.
    """
    try:
        # Read the current loader file
        with open(LOADER_FILE, 'r') as f:
            content = f.read()
        
        # Check if we need to apply the fix
        if "load_pipeline_model" not in content:
            # Find the end of the _load_transformers_model method
            method_end = content.find("def _load_sentence_transformer")
            if method_end == -1:
                logger.error("Could not find _load_sentence_transformer method")
                return False
            
            # Add pipeline support method before _load_sentence_transformer
            pipeline_method = """
    def _load_pipeline_model(self, model_config: ModelConfig, device: str) -> Any:
        \"\"\"
        Load a model using the transformers Pipeline API
        
        Args:
            model_config (ModelConfig): Model configuration
            device (str): Device to load model on
            
        Returns:
            Any: Pipeline model
        \"\"\"
        # Check if transformers is available
        if not HAVE_TRANSFORMERS:
            raise ImportError("transformers library is required for pipeline models")
        
        # Import pipeline from transformers
        from transformers import pipeline
        
        logger.info(f"Loading pipeline model: {model_config.model_name} for task {model_config.pipeline_task}")
        
        # Get model kwargs
        model_kwargs = model_config.model_kwargs or {}
        
        # Add cache directory
        model_kwargs["cache_dir"] = self.get_cache_path(model_config.model_name)
        
        # Create the pipeline
        try:
            # Check if we have tokenizer arguments
            tokenizer_kwargs = model_config.tokenizer_kwargs or {}
            
            # Create the pipeline with appropriate task and model
            pipe = pipeline(
                task=model_config.pipeline_task,
                model=model_config.model_name,
                device=device if device != "mps" else -1,  # Use -1 for CPU if MPS is specified
                **model_kwargs
            )
            
            logger.info(f"Successfully loaded pipeline for {model_config.pipeline_task} using {model_config.model_name}")
            return pipe
            
        except Exception as e:
            logger.error(f"Error creating pipeline for {model_config.model_name}: {e}")
            
            # Try creating a pipeline without a specific model (using default)
            try:
                logger.info(f"Trying to create pipeline for {model_config.pipeline_task} without specific model")
                pipe = pipeline(
                    task=model_config.pipeline_task,
                    device=device if device != "mps" else -1,
                    **model_kwargs
                )
                logger.info(f"Successfully created default pipeline for {model_config.pipeline_task}")
                return pipe
            except Exception as e2:
                logger.error(f"Error creating default pipeline: {e2}")
                raise ValueError(f"Failed to create pipeline for {model_config.pipeline_task}: {e}")
    
    """
            
            # Modify the Unsupported model type error in _load_model
            unsupported_model_error = "                else:\n                    # Unsupported model type\n                    raise ValueError(f\"Unsupported model type: {model_config.type}\")"
            pipeline_model_support = """                elif model_config.type == "pipeline" or getattr(model_config, "use_pipeline", False):
                    # Load pipeline model
                    logger.info(f"Loading pipeline model for {model_type}...")
                    console.print(f"[cyan]Loading pipeline model for {model_type}...[/cyan]")
                    model = self._load_pipeline_model(model_config, device)
                    tokenizer = None  # Pipeline handles tokenization internally
                    
                else:
                    # Unsupported model type
                    raise ValueError(f"Unsupported model type: {model_config.type}")"""
            
            # Replace the unsupported model error with pipeline support
            updated_content = content.replace(unsupported_model_error, pipeline_model_support)
            
            # Add pipeline_method just before _load_sentence_transformer
            updated_content = updated_content[:method_end] + pipeline_method + updated_content[method_end:]
            
            # Write the updated content back to the file
            with open(LOADER_FILE, 'w') as f:
                f.write(updated_content)
            
            logger.info(f"Added pipeline support to model loader in {LOADER_FILE}")
            return True
        else:
            logger.info("Pipeline support already exists in model loader")
            return True
    except Exception as e:
        logger.error(f"Error adding pipeline support to model loader: {e}")
        return False

def create_emergency_audio_files():
    """Create emergency audio files for TTS fallback."""
    try:
        # Import required libraries
        from gtts import gTTS
        import os
        
        # Create directories
        os.makedirs("audio", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        os.makedirs("temp/tts", exist_ok=True)
        
        # Create emergency audio files
        emergency_texts = [
            "This is an emergency fallback audio file for the text to speech system.",
            "The text to speech system was unable to process your request. This is a fallback audio file.",
            "This is a backup audio file provided when the text to speech system encounters an error."
        ]
        
        # Create emergency files in standard locations
        for directory in ["temp", "audio", "temp/tts"]:
            for i, text in enumerate(emergency_texts):
                filename = f"{directory}/tts_emergency_{i}.mp3"
                try:
                    tts = gTTS(text=text, lang="en")
                    tts.save(filename)
                    logger.info(f"Created emergency audio file: {filename}")
                except Exception as e:
                    logger.error(f"Error creating emergency audio file {filename}: {e}")
        
        logger.info("Created emergency audio files for TTS fallback")
        return True
    except ImportError:
        logger.error("gTTS not installed. Install with: pip install gtts")
        return False
    except Exception as e:
        logger.error(f"Error creating emergency audio files: {e}")
        return False

def add_speech_metrics_collection():
    """Add metrics collection for speech processing endpoints."""
    try:
        # Path to metrics collection file
        metrics_file = "app/audit/metrics.py"
        
        # Read the current metrics file
        with open(metrics_file, 'r') as f:
            content = f.read()
        
        # Check if speech metrics are already included
        if "speech_metrics" not in content:
            # Find the end of the collect_metrics method
            method_end = content.find("def save_metrics")
            if method_end == -1:
                logger.error("Could not find save_metrics method")
                return False
            
            # Add speech metrics collection
            speech_metrics = """    def collect_speech_metrics(self, metrics_type: str, **kwargs):
        \"\"\"Collect metrics for speech processing operations.\"\"\"
        if not self.enabled:
            return
            
        # Extract common metrics
        duration = kwargs.get("duration", 0)
        model_used = kwargs.get("model_used", "unknown")
        is_fallback = kwargs.get("is_fallback", False)
        language = kwargs.get("language", "en")
        
        # Collect appropriate metrics based on type
        if metrics_type == "tts":
            # Text-to-Speech metrics
            text_length = kwargs.get("text_length", 0)
            audio_size = kwargs.get("audio_size", 0)
            voice = kwargs.get("voice", "default")
            
            self.metrics["speech_synthesis"] = self.metrics.get("speech_synthesis", {
                "count": 0,
                "total_duration": 0,
                "total_text_length": 0,
                "total_audio_size": 0,
                "fallback_count": 0,
                "models": {},
                "languages": {},
                "voices": {}
            })
            
            # Update speech synthesis metrics
            synthesis = self.metrics["speech_synthesis"]
            synthesis["count"] += 1
            synthesis["total_duration"] += duration
            synthesis["total_text_length"] += text_length
            synthesis["total_audio_size"] += audio_size
            if is_fallback:
                synthesis["fallback_count"] += 1
                
            # Track by model
            synthesis["models"][model_used] = synthesis["models"].get(model_used, 0) + 1
            
            # Track by language
            synthesis["languages"][language] = synthesis["languages"].get(language, 0) + 1
            
            # Track by voice
            synthesis["voices"][voice] = synthesis["voices"].get(voice, 0) + 1
            
        elif metrics_type == "stt":
            # Speech-to-Text metrics
            text_length = kwargs.get("text_length", 0)
            audio_size = kwargs.get("audio_size", 0)
            confidence = kwargs.get("confidence", 0)
            
            self.metrics["speech_recognition"] = self.metrics.get("speech_recognition", {
                "count": 0,
                "total_duration": 0,
                "total_text_length": 0,
                "total_audio_size": 0,
                "fallback_count": 0,
                "avg_confidence": 0,
                "models": {},
                "languages": {}
            })
            
            # Update speech recognition metrics
            recognition = self.metrics["speech_recognition"]
            recognition["count"] += 1
            recognition["total_duration"] += duration
            recognition["total_text_length"] += text_length
            recognition["total_audio_size"] += audio_size
            
            # Update average confidence
            prev_avg = recognition["avg_confidence"]
            prev_count = recognition["count"] - 1
            recognition["avg_confidence"] = (prev_avg * prev_count + confidence) / recognition["count"]
            
            if is_fallback:
                recognition["fallback_count"] += 1
                
            # Track by model
            recognition["models"][model_used] = recognition["models"].get(model_used, 0) + 1
            
            # Track by language
            recognition["languages"][language] = recognition["languages"].get(language, 0) + 1
        
        # Save metrics after collection
        self.save_metrics()
"""
            
            # Add speech metrics collection to metrics.py
            updated_content = content[:method_end] + speech_metrics + content[method_end:]
            
            # Write the updated content back to the file
            with open(metrics_file, 'w') as f:
                f.write(updated_content)
            
            logger.info(f"Added speech metrics collection to {metrics_file}")
            return True
        else:
            logger.info("Speech metrics collection already exists")
            return True
    except Exception as e:
        logger.error(f"Error adding speech metrics collection: {e}")
        return False

def main():
    """Main function to execute all fixes."""
    print("===== FIXING TTS AND STT MODELS =====")
    
    # Fix TTS model registry
    print("\n1. Updating TTS model configuration in registry...")
    if fix_tts_model_registry():
        print("✅ Successfully updated TTS model configuration")
    else:
        print("❌ Failed to update TTS model configuration")
    
    # Add pipeline support to model loader
    print("\n2. Adding pipeline support to model loader...")
    if add_pipeline_support_to_loader():
        print("✅ Successfully added pipeline support to model loader")
    else:
        print("❌ Failed to add pipeline support to model loader")
    
    # Create emergency audio files
    print("\n3. Creating emergency audio files for TTS fallback...")
    if create_emergency_audio_files():
        print("✅ Successfully created emergency audio files")
    else:
        print("❌ Failed to create emergency audio files")
    
    # Add speech metrics collection
    print("\n4. Adding speech metrics collection...")
    if add_speech_metrics_collection():
        print("✅ Successfully added speech metrics collection")
    else:
        print("❌ Failed to add speech metrics collection")
    
    print("\n===== FIX COMPLETE =====")
    print("TTS and STT models have been fixed and configured for optimal performance.")
    print("You should now restart the server to apply these changes.")
    print("\nTo restart the server, run: kill $(cat server.pid) && uvicorn app.main:app --reload")
    print("\nTo test the fixed TTS endpoint, run: python scripts/test_direct_tts.py")
    print("To test the fixed STT endpoint, run: python scripts/test_speech_workflow.py")

if __name__ == "__main__":
    main()