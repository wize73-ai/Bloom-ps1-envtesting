#!/usr/bin/env python3
"""
CasaLingua Model Verification Script

This script checks if models are already downloaded and loaded properly
before attempting any operations. It will ensure system stability with
larger models loaded.
"""

import os
import sys
import json
import logging
import torch
import time
import importlib.util
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_verifier")

def check_model_files(model_name, cache_dir=None):
    """Check if a model's files are already downloaded"""
    if not cache_dir:
        # Check typical Hugging Face cache locations
        home = os.path.expanduser("~")
        possible_locations = [
            os.path.join(home, ".cache", "huggingface", "transformers"),
            os.path.join(home, ".cache", "huggingface", "hub"),
            os.path.join("cache", "models"),
            os.path.join("models"),
            os.path.join("cache", "transformers")
        ]
        
        # Add location from environment variable if set
        if "TRANSFORMERS_CACHE" in os.environ:
            possible_locations.insert(0, os.environ["TRANSFORMERS_CACHE"])
        
        # Check each location
        model_dirs = []
        for location in possible_locations:
            if os.path.exists(location):
                # Look for model directory
                for subdir in os.listdir(location):
                    if model_name.replace("/", "--") in subdir:
                        model_dirs.append(os.path.join(location, subdir))
        
        if model_dirs:
            logger.info(f"Found model files for {model_name} at: {model_dirs[0]}")
            return True
    
    # If cache_dir is provided, check there specifically
    if cache_dir:
        model_dir = os.path.join(cache_dir, model_name.replace("/", "--"))
        if os.path.exists(model_dir):
            logger.info(f"Found model files for {model_name} at: {model_dir}")
            return True
    
    logger.warning(f"Model files for {model_name} not found locally. Download would be required.")
    return False

def read_registry(registry_path="config/model_registry.json"):
    """Read the model registry file"""
    try:
        with open(registry_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read registry at {registry_path}: {str(e)}")
        return {}

def check_transformers_installed():
    """Check if transformers library is installed"""
    if importlib.util.find_spec("transformers") is None:
        logger.error("Transformers library is not installed")
        return False
    return True

def check_sentence_transformers_installed():
    """Check if sentence-transformers library is installed"""
    if importlib.util.find_spec("sentence_transformers") is None:
        logger.error("Sentence-transformers library is not installed")
        return False
    return True

def check_device_availability():
    """Check device availability for models"""
    device_info = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    }
    
    # Log device information
    logger.info(f"CUDA available: {device_info['cuda']}")
    if device_info['cuda']:
        device_count = torch.cuda.device_count()
        logger.info(f"CUDA device count: {device_count}")
        for i in range(device_count):
            logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"CUDA device properties: {torch.cuda.get_device_properties(0)}")
    
    logger.info(f"MPS (Apple Silicon) available: {device_info['mps']}")
    
    # Determine best device
    if device_info['cuda']:
        best_device = "cuda"
    elif device_info['mps']:
        best_device = "mps"
    else:
        best_device = "cpu"
        
    logger.info(f"Best available device: {best_device}")
    return device_info, best_device

def check_memory_availability():
    """Check system memory availability"""
    import psutil
    
    # System memory
    virtual_memory = psutil.virtual_memory()
    memory_available_gb = virtual_memory.available / (1024**3)
    memory_total_gb = virtual_memory.total / (1024**3)
    
    logger.info(f"System memory: {memory_total_gb:.1f} GB total / {memory_available_gb:.1f} GB available")
    
    # GPU memory if available
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_memory_available = gpu_memory_total - gpu_memory_allocated
        
        logger.info(f"GPU memory: {gpu_memory_total:.1f} GB total / {gpu_memory_available:.1f} GB available")
        
        return {
            "system_total": memory_total_gb,
            "system_available": memory_available_gb,
            "gpu_total": gpu_memory_total,
            "gpu_available": gpu_memory_available
        }
    
    return {
        "system_total": memory_total_gb,
        "system_available": memory_available_gb
    }

def verify_specific_model(model_name, check_download=True):
    """Verify if a specific model can be loaded"""
    logger.info(f"Verifying model: {model_name}")
    
    # Check if model files exist
    if check_download and not check_model_files(model_name):
        return False
    
    # Try loading the model configuration (lightweight operation)
    try:
        from transformers import AutoConfig
        start_time = time.time()
        config = AutoConfig.from_pretrained(model_name)
        elapsed = time.time() - start_time
        logger.info(f"Successfully loaded config for {model_name} in {elapsed:.2f}s")
        return True
    except Exception as e:
        logger.error(f"Failed to load config for {model_name}: {str(e)}")
        return False

def check_system_stability():
    """Check overall system stability for running models"""
    logger.info("Checking system stability for model operation")
    
    # Check dependencies
    transformers_ok = check_transformers_installed()
    sentence_transformers_ok = check_sentence_transformers_installed()
    
    if not transformers_ok:
        logger.error("Critical dependency missing: transformers")
        return False
    
    # Check devices
    device_info, best_device = check_device_availability()
    
    # Check memory
    memory_info = check_memory_availability()
    
    # Determine if system has enough resources for large models
    has_enough_memory = memory_info.get("system_available", 0) > 4.0  # At least 4GB available
    
    if not has_enough_memory:
        logger.warning("System may not have enough memory for large models")
    
    # Overall assessment
    system_ok = transformers_ok and has_enough_memory
    
    if system_ok:
        logger.info("System appears capable of running CasaLingua models")
    else:
        logger.warning("System may encounter issues running large models")
    
    return system_ok

def main():
    """Main function to verify models"""
    parser = argparse.ArgumentParser(description="Verify CasaLingua model loading")
    parser.add_argument("--registry", default="config/model_registry.json", help="Path to model registry")
    parser.add_argument("--check-downloads", action="store_true", help="Check if models are downloaded")
    args = parser.parse_args()
    
    logger.info("Starting CasaLingua model verification")
    
    # Check system stability
    system_stable = check_system_stability()
    
    # Read registry
    registry = read_registry(args.registry)
    if not registry:
        logger.error("Failed to read model registry")
        return 1
    
    # Check primary models
    mbart_ok = False
    for model_id, config in registry.items():
        model_name = config.get("model_name")
        if not model_name:
            continue
            
        # Check if it's an important model
        is_primary = config.get("is_primary", False)
        is_mbart = "mbart" in model_name.lower()
        
        if is_primary or is_mbart:
            logger.info(f"Checking primary model: {model_id} ({model_name})")
            model_ok = verify_specific_model(model_name, args.check_downloads)
            
            if model_ok and is_mbart:
                mbart_ok = True
                logger.info("MBART model verification successful")
    
    # Overall assessment
    if mbart_ok and system_stable:
        logger.info("System appears ready to use MBART for translation")
        return 0
    elif not mbart_ok:
        logger.warning("MBART model verification failed - system may fall back to smaller models")
        return 2
    else:
        logger.warning("System stability concerns detected - model performance may be affected")
        return 3

if __name__ == "__main__":
    sys.exit(main())