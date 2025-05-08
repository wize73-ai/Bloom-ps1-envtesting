#!/usr/bin/env python3
"""
Script to check if MBART model is properly loaded
"""

import requests
import json

API_URL = "http://localhost:8000"

def check_model_health():
    """Check model health endpoint"""
    endpoint = f"{API_URL}/health/models"
    
    print(f"Checking model health at {endpoint}")
    
    try:
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            result = response.json()
            print("Model health check results:")
            print(json.dumps(result, indent=2))
            
            # Check for translation model specifically
            if "models" in result and "translation" in result["models"]:
                translation_model = result["models"]["translation"]
                print("\nTranslation model status:")
                print(f"Loaded: {translation_model.get('loaded', False)}")
                print(f"Model type: {translation_model.get('model_type', 'unknown')}")
                print(f"Status: {translation_model.get('status', 'unknown')}")
            else:
                print("Translation model information not found in health check")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception: {str(e)}")

def check_system_config():
    """Check system configuration"""
    endpoint = f"{API_URL}/admin/config"
    
    print(f"\nChecking system configuration at {endpoint}")
    
    try:
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            result = response.json()
            
            if "data" in result:
                config = result["data"]
                # Look for translation-related config
                translation_configs = {k: v for k, v in config.items() if "translation" in k.lower() or "mbart" in k.lower()}
                
                if translation_configs:
                    print("Translation-related configuration:")
                    print(json.dumps(translation_configs, indent=2))
                else:
                    print("No translation-specific configuration found")
            else:
                print("No configuration data found in response")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception: {str(e)}")

if __name__ == "__main__":
    check_model_health()
    check_system_config()