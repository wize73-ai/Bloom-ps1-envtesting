"""
Configuration and fixtures for functional tests.
"""
import os
import sys
import pytest
import asyncio
import aiohttp
from pathlib import Path
import time
import json

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Default server URL for tests
DEFAULT_SERVER_URL = os.environ.get("CASALINGUA_TEST_SERVER", "http://localhost:8000")

# Test data constants
TEST_DATA_DIR = Path(__file__).parent / "test_data"

# Timeout for server connection attempts
CONNECTION_TIMEOUT = 30  # seconds
CONNECTION_RETRY_INTERVAL = 5  # seconds

@pytest.fixture(scope="session")
def server_url():
    """Return the URL of the server to test against."""
    return DEFAULT_SERVER_URL

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def server_connection(server_url):
    """Check if the server is running and accessible."""
    import requests
    
    connected = False
    retries = CONNECTION_TIMEOUT // CONNECTION_RETRY_INTERVAL
    start_time = time.time()
    
    while not connected and retries > 0:
        try:
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                connected = True
                elapsed = time.time() - start_time
                print(f"✅ Connected to server at {server_url} after {elapsed:.2f}s")
                return True
        except Exception as e:
            print(f"⚠️ Connection attempt failed: {e}")
            print(f"Retrying in {CONNECTION_RETRY_INTERVAL} seconds... ({retries} attempts left)")
            retries -= 1
            time.sleep(CONNECTION_RETRY_INTERVAL)
    
    if not connected:
        print(f"❌ Failed to connect to server at {server_url} after {CONNECTION_TIMEOUT} seconds.")
        print("Make sure the server is running and accessible.")
        pytest.skip(f"Could not connect to server at {server_url}")
        return False

@pytest.fixture(scope="session")
def api_client():
    """Return an aiohttp ClientSession for making API requests."""
    # We need to create the session inside the test function itself
    # to avoid the "async_generator has no attribute get" error
    return aiohttp.ClientSession

@pytest.fixture
def load_test_data():
    """Function to load test data from the test_data directory."""
    def _load_test_data(filename):
        file_path = TEST_DATA_DIR / filename
        if not file_path.exists():
            pytest.fail(f"Test data file '{filename}' not found in {TEST_DATA_DIR}")
        
        if filename.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    return _load_test_data

@pytest.fixture
def save_test_results():
    """Function to save test results for analysis."""
    results_dir = ROOT_DIR / "tests" / "results"
    results_dir.mkdir(exist_ok=True)
    
    def _save_test_results(test_name, results):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"{test_name}_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return result_file
    
    return _save_test_results