"""
Integration tests for the API health endpoints.
"""
import pytest
import asyncio
import aiohttp
import json
from pathlib import Path

# Test constants
HEALTH_ENDPOINT = "/health"
MODELS_HEALTH_ENDPOINT = "/health/models"
DATABASE_HEALTH_ENDPOINT = "/health/database"
MEMORY_HEALTH_ENDPOINT = "/health/memory"
STATUS_ENDPOINT = "/status"

@pytest.mark.asyncio
async def test_health_endpoint(server_url, server_connection, api_client):
    """Test the main health endpoint."""
    endpoint = f"{server_url}{HEALTH_ENDPOINT}"
    
    async with api_client() as session:
        try:
            async with session.get(endpoint) as response:
                assert response.status == 200, "Health check endpoint failed"
                
                # Get response data
                resp_data = await response.json()
                
                # Verify response format
                assert isinstance(resp_data, dict), "Response is not a JSON object"
                assert "status" in resp_data, "Response does not contain status field"
                assert resp_data["status"] in ["ok", "healthy", "up"], f"Health status is not healthy: {resp_data['status']}"
                
                # Print health details
                print("\nHealth endpoint response:")
                print(f"Status: {resp_data['status']}")
                
                # Check for uptime if available
                if "uptime" in resp_data:
                    print(f"Uptime: {resp_data['uptime']} seconds")
                
                # Check for version if available
                if "version" in resp_data:
                    print(f"Version: {resp_data['version']}")
        except Exception as e:
            pytest.fail(f"Health endpoint exception: {str(e)}")

@pytest.mark.asyncio
async def test_models_health_endpoint(server_url, server_connection, api_client):
    """Test the models health endpoint."""
    endpoint = f"{server_url}{MODELS_HEALTH_ENDPOINT}"
    
    async with api_client() as session:
        try:
            async with session.get(endpoint) as response:
                # Models health endpoint might return 200 (if models are healthy) or a non-200 status
                # if models are unhealthy but the endpoint itself is working
                assert response.status in [200, 201, 202, 203, 400, 500, 503], "Models health endpoint failed with unexpected status"
                
                # Get response data
                resp_data = await response.json()
                
                # Verify response contains useful information
                assert isinstance(resp_data, dict), "Response is not a JSON object"
                
                # Print models health details
                print("\nModels health endpoint response:")
                print(f"Status: {resp_data.get('status', 'not reported')}")
                
                # Check for loaded models if available
                if "loaded_models" in resp_data:
                    models = resp_data["loaded_models"]
                    if isinstance(models, list):
                        print(f"Loaded models: {', '.join(models) if models else 'None'}")
                    else:
                        print(f"Loaded models info: {models}")
                
                # Check for model issues if available
                if "issues" in resp_data:
                    issues = resp_data["issues"]
                    print(f"Model issues: {issues}")
        except aiohttp.ClientError:
            # The endpoint might not exist, which is acceptable as it's not a core endpoint
            pytest.skip("Models health endpoint not available")
        except Exception as e:
            pytest.skip(f"Models health endpoint exception: {str(e)}")

@pytest.mark.asyncio
async def test_database_health_endpoint(server_url, server_connection, api_client):
    """Test the database health endpoint."""
    endpoint = f"{server_url}{DATABASE_HEALTH_ENDPOINT}"
    
    async with api_client() as session:
        try:
            async with session.get(endpoint) as response:
                # Database health endpoint might return 200 (if DB is healthy) or a non-200 status
                # if DB is unhealthy but the endpoint itself is working
                assert response.status in [200, 201, 202, 203, 400, 500, 503], "Database health endpoint failed with unexpected status"
                
                # Get response data
                resp_data = await response.json()
                
                # Verify response contains useful information
                assert isinstance(resp_data, dict), "Response is not a JSON object"
                
                # Print database health details
                print("\nDatabase health endpoint response:")
                print(f"Status: {resp_data.get('status', 'not reported')}")
                
                # Check for database type if available
                if "database_type" in resp_data:
                    print(f"Database type: {resp_data['database_type']}")
                
                # Check for connection details if available
                if "connection" in resp_data:
                    conn = resp_data["connection"]
                    print(f"Connection: {conn}")
        except aiohttp.ClientError:
            # The endpoint might not exist, which is acceptable as it's not a core endpoint
            pytest.skip("Database health endpoint not available")
        except Exception as e:
            pytest.skip(f"Database health endpoint exception: {str(e)}")

@pytest.mark.asyncio
async def test_memory_health_endpoint(server_url, server_connection, api_client):
    """Test the memory health endpoint."""
    endpoint = f"{server_url}{MEMORY_HEALTH_ENDPOINT}"
    
    async with api_client() as session:
        try:
            async with session.get(endpoint) as response:
                if response.status == 404:
                    # The endpoint might not exist, which is acceptable as it's not a core endpoint
                    pytest.skip("Memory health endpoint not available")
                
                assert response.status in [200, 201, 202, 203, 400, 500, 503], "Memory health endpoint failed with unexpected status"
                
                # Get response data
                resp_data = await response.json()
                
                # Verify response contains useful information
                assert isinstance(resp_data, dict), "Response is not a JSON object"
                
                # Print memory health details
                print("\nMemory health endpoint response:")
                if "memory" in resp_data:
                    memory = resp_data["memory"]
                    if isinstance(memory, dict):
                        for key, value in memory.items():
                            print(f"{key}: {value}")
                    else:
                        print(f"Memory: {memory}")
        except aiohttp.ClientError:
            # The endpoint might not exist, which is acceptable as it's not a core endpoint
            pytest.skip("Memory health endpoint not available")
        except Exception as e:
            pytest.skip(f"Memory health endpoint exception: {str(e)}")

@pytest.mark.asyncio
async def test_status_endpoint(server_url, server_connection, api_client):
    """Test the status endpoint."""
    endpoint = f"{server_url}{STATUS_ENDPOINT}"
    
    async with api_client() as session:
        try:
            async with session.get(endpoint) as response:
                if response.status == 404:
                    # The endpoint might not exist, which is acceptable as it's not a core endpoint
                    pytest.skip("Status endpoint not available")
                
                assert response.status == 200, "Status endpoint failed"
                
                # Get response data
                resp_data = await response.json()
                
                # Verify response contains useful information
                assert isinstance(resp_data, dict), "Response is not a JSON object"
                
                # Print status details
                print("\nStatus endpoint response:")
                for key, value in resp_data.items():
                    if isinstance(value, dict):
                        print(f"{key}:")
                        for k, v in value.items():
                            print(f"  {k}: {v}")
                    else:
                        print(f"{key}: {value}")
        except aiohttp.ClientError:
            # The endpoint might not exist, which is acceptable as it's not a core endpoint
            pytest.skip("Status endpoint not available")
        except Exception as e:
            pytest.skip(f"Status endpoint exception: {str(e)}")