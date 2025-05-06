import asyncio
import aiohttp
import json

BASE_URL = "http://localhost:8000"

TEST_ANALYZE_TEXT = "Apple Inc. is planning to open a new store in New York City next month. The company's CEO, Tim Cook, announced the news during a press conference on Thursday."
TEST_SUMMARIZE_TEXT = """
The United Nations climate conference held in Glasgow brought together world leaders to discuss measures to combat climate change. 
Many countries pledged to reach net-zero emissions by 2050, while developing nations advocated for more financial support from wealthy countries. 
The conference concluded with a new global agreement, though some environmental activists criticized it for not going far enough. 
Key outcomes included commitments to reduce methane emissions, halt deforestation, and transition away from coal power.
"""

async def test_analyze_endpoint():
    """Debug the analyze endpoint"""
    # Try different endpoints
    endpoints = [
        f"{BASE_URL}/pipeline/analyze",
        f"{BASE_URL}/analyze"
    ]
    
    # Multiple payloads to try
    payloads = [
        # Payload 1: Basic
        {
            "text": TEST_ANALYZE_TEXT,
            "language": "en",
        },
        # Payload 2: Include flags
        {
            "text": TEST_ANALYZE_TEXT,
            "language": "en",
            "include_sentiment": True,
            "include_entities": True,
            "include_topics": True,
            "include_summary": False,
        },
        # Payload 3: With analyses array
        {
            "text": TEST_ANALYZE_TEXT,
            "language": "en",
            "analyses": ["entities", "sentiment", "topics"],
        },
        # Payload 4: Full payload
        {
            "text": TEST_ANALYZE_TEXT,
            "language": "en",
            "include_sentiment": True,
            "include_entities": True,
            "include_topics": True,
            "include_summary": False,
            "analyses": ["entities", "sentiment", "topics"],
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            print(f"\nTesting endpoint: {endpoint}")
            
            for i, payload in enumerate(payloads):
                print(f"\nPayload {i+1}:")
                print(json.dumps(payload, indent=2))
                
                try:
                    async with session.post(endpoint, json=payload) as response:
                        status = response.status
                        
                        try:
                            result = await response.json()
                            result_str = json.dumps(result, indent=2)
                        except:
                            result_str = await response.text()
                        
                        print(f"Status: {status}")
                        print(f"Response: {result_str[:500]}...")
                        
                        if status == 200:
                            print("✅ SUCCESS!")
                except Exception as e:
                    print(f"Error: {str(e)}")

async def test_summarize_endpoint():
    """Debug the summarize endpoint"""
    # Try different endpoints
    endpoints = [
        f"{BASE_URL}/pipeline/summarize",
        f"{BASE_URL}/summarize"
    ]
    
    # Multiple payloads to try
    payloads = [
        # Payload 1: Basic
        {
            "text": TEST_SUMMARIZE_TEXT,
            "language": "en",
        },
        # Payload 2: With model_id
        {
            "text": TEST_SUMMARIZE_TEXT,
            "language": "en",
            "model_id": None,
        },
        # Payload 3: With model_id as empty string
        {
            "text": TEST_SUMMARIZE_TEXT,
            "language": "en",
            "model_id": "",
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            print(f"\nTesting endpoint: {endpoint}")
            
            for i, payload in enumerate(payloads):
                print(f"\nPayload {i+1}:")
                print(json.dumps(payload, indent=2))
                
                try:
                    async with session.post(endpoint, json=payload) as response:
                        status = response.status
                        
                        try:
                            result = await response.json()
                            result_str = json.dumps(result, indent=2)
                        except:
                            result_str = await response.text()
                        
                        print(f"Status: {status}")
                        print(f"Response: {result_str[:500]}...")
                        
                        if status == 200:
                            print("✅ SUCCESS!")
                except Exception as e:
                    print(f"Error: {str(e)}")

async def main():
    print("=== Debugging Analyze Endpoint ===")
    await test_analyze_endpoint()
    
    print("\n\n=== Debugging Summarize Endpoint ===")
    await test_summarize_endpoint()

if __name__ == "__main__":
    asyncio.run(main())