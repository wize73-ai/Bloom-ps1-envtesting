"""
Test script to verify the endpoints after fixes.
"""
import asyncio
import httpx
import json
from datetime import datetime

async def test_all_endpoints():
    print(f"Testing API endpoints at {datetime.now().isoformat()}")
    
    # API endpoints to test
    endpoints = [
        {
            "name": "Translation",
            "url": "http://localhost:8000/pipeline/translate",
            "method": "POST",
            "payload": {
                "text": "Hello, how are you?",
                "source_language": "en",
                "target_language": "es",
                "model_name": "mbart-large-50-many-to-many-mmt",
                "preserve_formatting": True,
                "formality": None,
                "context": [],
                "domain": None,
                "glossary_id": None,
                "verify": False,
                "parameters": {}
            }
        },
        {
            "name": "Language Detection",
            "url": "http://localhost:8000/pipeline/detect",
            "method": "POST",
            "payload": {
                "text": "Hello, how are you?",
                "detailed": True,
                "model_id": None
            }
        },
        {
            "name": "Simplification",
            "url": "http://localhost:8000/pipeline/simplify",
            "method": "POST",
            "payload": {
                "text": "The mitochondrion is a double membrane-bound organelle found in most eukaryotic organisms.",
                "language": "en",
                "target_level": "simple",
                "model_id": None,
                "preserve_formatting": True,
                "parameters": {}
            }
        },
        {
            "name": "Anonymization",
            "url": "http://localhost:8000/pipeline/anonymize",
            "method": "POST",
            "payload": {
                "text": "John Smith lives at 123 Main St, New York and his email is john.smith@example.com.",
                "language": "en",
                "strategy": "mask",
                "entities": ["PERSON", "EMAIL", "LOCATION"],
                "preserve_formatting": True
            }
        }
    ]
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token"  # Simple test token for authentication
    }
    
    results = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for endpoint in endpoints:
            try:
                print(f"\nTesting {endpoint['name']} endpoint: {endpoint['url']}")
                
                if endpoint["method"] == "POST":
                    response = await client.post(
                        endpoint["url"], 
                        json=endpoint["payload"], 
                        headers=headers
                    )
                else:
                    response = await client.get(
                        endpoint["url"],
                        headers=headers
                    )
                
                print(f"Status code: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"{endpoint['name']} test: SUCCESS")
                    print(f"Response: {json.dumps(response.json(), indent=2)[:200]}...")  # Show first 200 chars
                    results.append({
                        "endpoint": endpoint["name"],
                        "success": True,
                        "status_code": response.status_code
                    })
                else:
                    print(f"{endpoint['name']} test: FAILED")
                    print(f"Error: {response.text}")
                    results.append({
                        "endpoint": endpoint["name"],
                        "success": False,
                        "status_code": response.status_code,
                        "error": response.text
                    })
            except Exception as e:
                print(f"{endpoint['name']} test: ERROR - {str(e)}")
                results.append({
                    "endpoint": endpoint["name"],
                    "success": False,
                    "error": str(e)
                })
    
    # Summary
    print("\n=== TEST SUMMARY ===")
    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    print(f"Successful tests: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    for result in results:
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{status} - {result['endpoint']}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_all_endpoints())