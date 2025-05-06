import asyncio
import aiohttp
import json
import time
import sys

BASE_URL = "http://localhost:8000"

async def test_health_endpoint():
    """Test the basic health endpoint"""
    endpoint = f"{BASE_URL}/health"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint) as response:
                status = response.status
                data = await response.json()
                return {
                    "endpoint": endpoint,
                    "status": status,
                    "data": data,
                    "successful": status == 200
                }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "status": None,
                "error": str(e),
                "successful": False
            }

async def test_health_details_endpoint():
    """Test the health details endpoint"""
    endpoint = f"{BASE_URL}/health/detailed"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint) as response:
                status = response.status
                data = await response.json()
                return {
                    "endpoint": endpoint,
                    "status": status,
                    "data": data,
                    "successful": status == 200
                }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "status": None,
                "error": str(e),
                "successful": False
            }

async def test_liveness_endpoint():
    """Test the liveness endpoint"""
    endpoint = f"{BASE_URL}/liveness"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint) as response:
                status = response.status
                data = await response.json()
                return {
                    "endpoint": endpoint,
                    "status": status,
                    "data": data,
                    "successful": status == 200
                }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "status": None,
                "error": str(e),
                "successful": False
            }

async def test_readiness_endpoint():
    """Test the readiness endpoint"""
    endpoint = f"{BASE_URL}/readiness"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint) as response:
                status = response.status
                data = await response.json()
                return {
                    "endpoint": endpoint,
                    "status": status,
                    "data": data,
                    "successful": status == 200
                }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "status": None,
                "error": str(e),
                "successful": False
            }

async def main():
    """Run all health endpoint tests"""
    print("Testing health endpoints...")
    
    # Try to connect to the server
    connected = False
    retries = 5
    
    while not connected and retries > 0:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{BASE_URL}/health") as response:
                    if response.status == 200:
                        connected = True
                        print(f"✅ Connected to server at {BASE_URL}")
        except Exception as e:
            print(f"⚠️ Connection attempt failed: {e}")
            print(f"Retrying in 5 seconds... ({retries} attempts left)")
            retries -= 1
            await asyncio.sleep(5)
    
    if not connected:
        print("❌ Failed to connect to server. Make sure the server is running.")
        sys.exit(1)
    
    # Run all tests
    results = await asyncio.gather(
        test_health_endpoint(),
        test_health_details_endpoint(),
        test_liveness_endpoint(),
        test_readiness_endpoint()
    )
    
    # Print results in a nice format
    print("\n=== Health Endpoint Test Results ===")
    all_successful = True
    
    for result in results:
        endpoint = result["endpoint"]
        status = result["status"]
        successful = result["successful"]
        
        if successful:
            print(f"✅ {endpoint} - Status: {status}")
        else:
            all_successful = False
            error = result.get("error", "Unknown error")
            print(f"❌ {endpoint} - Status: {status} - Error: {error}")
    
    # Print summary
    print("\n=== Summary ===")
    if all_successful:
        print("✅ All health endpoints are working correctly!")
    else:
        print("❌ Some health endpoints are not working correctly.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())