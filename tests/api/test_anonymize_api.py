"""
Simple test script to verify the anonymization API endpoint.
"""
import asyncio
import httpx
import json
from datetime import datetime

async def test_anonymize_endpoint():
    print(f"Testing anonymize endpoint at {datetime.now().isoformat()}")
    
    # API endpoint
    url = "http://localhost:8000/pipeline/anonymize"
    
    # Test payload
    payload = {
        "text": "John Smith lives at 123 Main St, New York and his email is john.smith@example.com.",
        "language": "en",
        "strategy": "mask",
        "entities": ["PERSON", "EMAIL", "LOCATION"],
        "preserve_formatting": True
    }
    
    # Headers
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
                print("Anonymization successful!")
                print(f"Response: {json.dumps(response.json(), indent=2)}")
                return True
            else:
                print(f"Error: {response.text}")
                return False
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_anonymize_endpoint())