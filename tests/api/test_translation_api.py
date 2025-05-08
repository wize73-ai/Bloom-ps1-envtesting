"""
Simple test script to verify the translation API endpoint.
"""
import asyncio
import httpx
import json
from datetime import datetime

async def test_translation_endpoint():
    print(f"Testing translation endpoint at {datetime.now().isoformat()}")
    
    # API endpoint - Using the /pipeline/translate path from the router
    url = "http://localhost:8000/pipeline/translate"
    
    # Test payload - Match the parameter names in the API definition
    payload = {
        "text": "Hello, how are you?",
        "source_language": "en",
        "target_language": "es",
        "model_name": "mbart-large-50-many-to-many-mmt",
        "preserve_formatting": True,
        "formality": None,
        "context": [],  # Context should be a list not a string
        "domain": None,
        "glossary_id": None,
        "verify": False,
        "parameters": {}  # Additional parameters
    }
    
    # Headers - Add auth header
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token"  # Simple test token for authentication
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            
            print(f"Status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            
            if response.status_code == 200:
                print("Translation successful!")
                print(f"Response: {json.dumps(response.json(), indent=2)}")
                return True
            else:
                print(f"Error: {response.text}")
                return False
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_translation_endpoint())