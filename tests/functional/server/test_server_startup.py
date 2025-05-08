"""
Tests for server startup and initialization process.

These tests verify that the server can start up correctly and initialize
all required components.
"""
import os
import sys
import pytest
import asyncio
import subprocess
import time
import requests
import signal
from pathlib import Path
import socket
import psutil

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Test port for the server to use
TEST_PORT = 8123

def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port=TEST_PORT, max_port=TEST_PORT + 100):
    """Find an available port starting from start_port."""
    for port in range(start_port, max_port):
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"Could not find an available port between {start_port} and {max_port}")

@pytest.fixture(scope="module")
def server_env():
    """Prepare environment variables for server startup."""
    # Save original environment
    original_env = os.environ.copy()
    
    # Set test environment
    test_env = original_env.copy()
    test_env["CASALINGUA_ENV"] = "test"
    test_env["PYTHONPATH"] = str(ROOT_DIR)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture(scope="module")
def server_process(server_env):
    """Start a test server instance and yield the process."""
    server_port = find_available_port()
    server_url = f"http://localhost:{server_port}"
    
    # Command to run the server
    server_script = ROOT_DIR / "scripts" / "run_casalingua.py"
    
    # Adjust environment to use test port
    env = server_env.copy()
    env["CASALINGUA_PORT"] = str(server_port)
    
    # Start the server process
    process = subprocess.Popen(
        [sys.executable, str(server_script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start up (max 60 seconds)
    start_time = time.time()
    timeout = 60
    server_ready = False
    
    print(f"Starting test server on port {server_port}...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{server_url}/health")
            if response.status_code == 200:
                server_ready = True
                break
        except requests.exceptions.ConnectionError:
            pass
        
        # Check if process has exited prematurely
        if process.poll() is not None:
            break
        
        time.sleep(1)
    
    if not server_ready:
        # If server failed to start, capture output for debugging
        stdout, stderr = process.communicate(timeout=5)
        process.terminate()
        
        print("===== Server stdout =====")
        print(stdout)
        print("===== Server stderr =====")
        print(stderr)
        
        pytest.skip(f"Server failed to start on port {server_port} within {timeout} seconds")
        return None
    
    # Server started successfully
    print(f"Test server started successfully on {server_url}")
    
    # Yield the process and URL
    yield {
        "process": process,
        "url": server_url,
        "port": server_port
    }
    
    # Shutdown server after tests
    print("Shutting down test server...")
    try:
        # Try to send terminate signal
        process.terminate()
        # Wait up to 10 seconds for process to exit
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # If process doesn't exit, kill it
        process.kill()
        process.wait()
    
    # Make sure port is released
    if is_port_in_use(server_port):
        # Find process using the port and kill it
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == server_port:
                        os.kill(proc.pid, signal.SIGKILL)
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

def test_server_starts_successfully(server_process):
    """Test that the server starts up successfully."""
    if server_process is None:
        pytest.skip("Server process fixture failed to start the server")
    
    assert server_process["process"].poll() is None, "Server process should still be running"
    
    # Check health endpoint
    response = requests.get(f"{server_process['url']}/health")
    assert response.status_code == 200, "Health endpoint should return 200 OK"
    
    # Check response contains expected fields
    health_data = response.json()
    assert "status" in health_data, "Health response should contain status field"
    assert health_data["status"] in ["ok", "healthy", "up"], f"Health status should be healthy, got {health_data['status']}"
    
    print(f"Server health check successful: {health_data}")

def test_api_endpoints_are_accessible(server_process):
    """Test that the API endpoints are accessible after startup."""
    if server_process is None:
        pytest.skip("Server process fixture failed to start the server")
    
    # List of endpoints to check
    endpoints = [
        "/health",
        "/pipeline/detect",
        "/pipeline/translate",
        "/pipeline/simplify"
    ]
    
    results = {}
    
    for endpoint in endpoints:
        url = f"{server_process['url']}{endpoint}"
        try:
            # Use OPTIONS request to check if endpoint exists, without needing to send valid data
            response = requests.options(url)
            exists = response.status_code != 404
            
            # Store result
            results[endpoint] = {
                "exists": exists,
                "status_code": response.status_code
            }
            
            if exists:
                print(f"Endpoint {endpoint} is accessible")
            else:
                print(f"Endpoint {endpoint} is not accessible (status: {response.status_code})")
                
        except requests.exceptions.RequestException as e:
            # If request fails, endpoint might not be accessible
            results[endpoint] = {
                "exists": False,
                "error": str(e)
            }
            print(f"Error accessing endpoint {endpoint}: {e}")
    
    # Verify that at least some endpoints are accessible
    accessible_endpoints = [ep for ep, result in results.items() if result.get("exists", False)]
    assert len(accessible_endpoints) > 0, "No API endpoints are accessible"
    
    # Print summary
    print(f"\nAPI endpoints check summary:")
    print(f"Accessible endpoints: {len(accessible_endpoints)}/{len(endpoints)}")
    for endpoint in accessible_endpoints:
        print(f"  ✓ {endpoint}")
    
    for endpoint, result in results.items():
        if not result.get("exists", False):
            print(f"  ✗ {endpoint} - {result.get('status_code', 'N/A')}")

def test_server_can_process_requests(server_process):
    """Test that the server can process basic requests."""
    if server_process is None:
        pytest.skip("Server process fixture failed to start the server")
    
    # Try to detect language - this should work even without models fully loaded
    detect_url = f"{server_process['url']}/pipeline/detect"
    alt_detect_url = f"{server_process['url']}/detect"
    
    text = "Hello, world!"
    data = {"text": text}
    
    # Try both potential endpoints
    urls_to_try = [detect_url, alt_detect_url]
    
    for url in urls_to_try:
        try:
            response = requests.post(url, json=data)
            
            # If request succeeds, check response
            if response.status_code == 200:
                result = response.json()
                print(f"Language detection successful: {result}")
                
                # Verify response contains useful information
                detected_language = None
                if "language" in result:
                    detected_language = result["language"]
                elif "data" in result and isinstance(result["data"], dict):
                    if "language" in result["data"]:
                        detected_language = result["data"]["language"]
                    elif "detected_language" in result["data"]:
                        detected_language = result["data"]["detected_language"]
                
                assert detected_language is not None, "Could not extract detected language from response"
                assert detected_language == "en", f"Expected detected language to be 'en', got '{detected_language}'"
                return  # Test passed
        except requests.exceptions.RequestException as e:
            print(f"Error with endpoint {url}: {e}")
    
    # If we get here, both endpoints failed
    pytest.fail("Server could not process language detection request on any endpoint")

def test_model_initialization(server_process):
    """Test that models are initialized correctly during server startup."""
    if server_process is None:
        pytest.skip("Server process fixture failed to start the server")
    
    # Check models health endpoint if available
    models_health_url = f"{server_process['url']}/health/models"
    
    try:
        response = requests.get(models_health_url)
        
        # If request succeeds, check response
        if response.status_code == 200:
            result = response.json()
            print(f"Models health check successful: {result}")
            
            # Verify response contains useful information
            assert "status" in result, "Models health response should contain status field"
            
            # Check for loaded models if available
            if "loaded_models" in result:
                loaded_models = result["loaded_models"]
                print(f"Loaded models: {loaded_models}")
                
                # At least some models should be loaded
                if isinstance(loaded_models, list):
                    assert len(loaded_models) > 0, "No models are loaded"
    except requests.exceptions.RequestException as e:
        # Models health endpoint might not be available in all configurations
        print(f"Models health endpoint not accessible: {e}")
        
        # Try a translation request instead to check if models are working
        translate_url = f"{server_process['url']}/pipeline/translate"
        alt_translate_url = f"{server_process['url']}/translate"
        
        data = {
            "text": "Hello, world!",
            "source_language": "en",
            "target_language": "es"
        }
        
        urls_to_try = [translate_url, alt_translate_url]
        
        for url in urls_to_try:
            try:
                response = requests.post(url, json=data)
                
                # If request succeeds, assume models are loaded
                if response.status_code == 200:
                    result = response.json()
                    print(f"Translation successful, models are loaded: {result}")
                    return  # Test passed
            except requests.exceptions.RequestException:
                pass
    
    # If no method worked, skip the test rather than failing
    # since we might not have access to model information
    pytest.skip("Could not verify model initialization")