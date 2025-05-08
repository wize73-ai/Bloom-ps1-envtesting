#!/usr/bin/env python3
"""
Unit tests for app/utils/config.py
"""
import os
import json
import time
import pytest
import threading
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

# Add project root to path
import sys
ROOT_DIR = Path(__file__).parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import module to test
from app.utils.config import (
    load_config, 
    validate_config, 
    get_config_value, 
    get_nested_value,
    register_config_change_callback,
    unregister_config_change_callback,
    start_config_watcher,
    stop_config_watcher,
    set_config_watcher_interval,
    detect_macos_m4
)

# Reset module globals between tests to avoid cross-test interference
@pytest.fixture(autouse=True)
def reset_config_module():
    """Reset global state between tests"""
    import app.utils.config as config_module
    
    # Save originals
    original_cache = config_module._config_cache.copy()
    original_timestamps = config_module._config_file_timestamps.copy()
    original_callbacks = config_module._config_callbacks.copy()
    original_watcher_thread = config_module._config_watcher_thread
    original_watcher_active = config_module._watcher_active
    original_watcher_interval = config_module._watcher_interval
    
    # Reset globals
    config_module._config_cache = {}
    config_module._config_file_timestamps = {}
    config_module._config_callbacks = []
    config_module._config_watcher_thread = None
    config_module._watcher_active = False
    config_module._watcher_interval = 10
    
    yield
    
    # Restore globals
    config_module._config_cache = original_cache
    config_module._config_file_timestamps = original_timestamps
    config_module._config_callbacks = original_callbacks
    config_module._config_watcher_thread = original_watcher_thread
    config_module._watcher_active = original_watcher_active
    config_module._watcher_interval = original_watcher_interval

@pytest.fixture
def mock_config_files():
    """Mock configuration files for testing"""
    default_config = {
        "environment": "development",
        "debug": True,
        "log_level": "INFO",
        "server_host": "0.0.0.0",
        "server_port": 8000,
        "models_dir": "models",
        "test_value": "default",
        "nested": {
            "value": "default_nested"
        }
    }
    
    dev_config = {
        "environment": "development",
        "debug": True,
        "test_value": "development",
        "nested": {
            "value": "dev_nested"
        }
    }
    
    prod_config = {
        "environment": "production",
        "debug": False,
        "test_value": "production",
        "nested": {
            "value": "prod_nested"
        }
    }
    
    # Create temp directory for config files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create config directory
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Write config files
        with open(config_dir / "default.json", "w") as f:
            json.dump(default_config, f)
            
        with open(config_dir / "development.json", "w") as f:
            json.dump(dev_config, f)
            
        with open(config_dir / "production.json", "w") as f:
            json.dump(prod_config, f)
        
        # Patch Path to use our temp directory
        with patch("app.utils.config.Path", side_effect=lambda p: Path(temp_dir) / p):
            yield {
                "default": default_config,
                "development": dev_config,
                "production": prod_config,
                "config_dir": config_dir
            }

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    original_environ = os.environ.copy()
    
    # Set test environment variables
    os.environ["ENVIRONMENT"] = "development"
    os.environ["CASALINGUA_DEBUG"] = "true"
    os.environ["CASALINGUA_TEST_ENV_VAR"] = "env_value"
    os.environ["CASALINGUA_SERVER_PORT"] = "9000"
    os.environ["CASALINGUA_NUMERIC_VALUE"] = "123"
    os.environ["CASALINGUA_FLOAT_VALUE"] = "3.14"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_environ)

def test_validate_config():
    """Test configuration validation and default values"""
    # Test with empty config
    empty_config = {}
    validated = validate_config(empty_config)
    
    # Check defaults for required keys
    assert validated["environment"] == "development"
    assert validated["server_host"] == "0.0.0.0"
    assert validated["server_port"] == 8000
    assert validated["log_level"] == "INFO"
    assert validated["models_dir"] == "models"
    assert validated["log_dir"] == "logs"
    
    # Test with partial config
    partial_config = {
        "environment": "production",
        "server_host": "localhost"
    }
    validated = validate_config(partial_config)
    
    # Check provided values are preserved
    assert validated["environment"] == "production"
    assert validated["server_host"] == "localhost"
    
    # Check defaults for missing keys
    assert validated["server_port"] == 8000
    assert validated["log_level"] == "INFO"

def test_get_config_value():
    """Test getting values from the configuration"""
    config = {
        "key1": "value1",
        "key2": {
            "nested1": "nested_value1",
            "nested2": {
                "deep": "deep_value"
            }
        },
        "key3": 123
    }
    
    # Test simple key lookup
    assert get_config_value(config, "key1") == "value1"
    assert get_config_value(config, "key3") == 123
    
    # Test nested key lookup
    assert get_config_value(config, "key2.nested1") == "nested_value1"
    assert get_config_value(config, "key2.nested2.deep") == "deep_value"
    
    # Test default values
    assert get_config_value(config, "nonexistent", "default") == "default"
    assert get_config_value(config, "key2.nonexistent", "default") == "default"
    assert get_config_value(config, "key2.nested2.nonexistent", "default") == "default"

def test_get_nested_value():
    """Test getting nested values using dot notation"""
    config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "credentials": {
                "username": "admin",
                "password": "secret"
            }
        },
        "debug": True
    }
    
    # Test top-level key
    assert get_nested_value(config, "debug") == True
    
    # Test nested keys
    assert get_nested_value(config, "database.host") == "localhost"
    assert get_nested_value(config, "database.port") == 5432
    assert get_nested_value(config, "database.credentials.username") == "admin"
    
    # Test default values
    assert get_nested_value(config, "nonexistent", "default") == "default"
    assert get_nested_value(config, "database.nonexistent", "default") == "default"
    assert get_nested_value(config, "database.credentials.nonexistent", "default") == "default"

def test_load_config_with_mock_files(mock_config_files, mock_env_vars):
    """Test loading configuration from files and environment variables"""
    # Load configuration with direct mock path access to avoid file pathing issues
    config_dir = mock_config_files["config_dir"]
    
    # Patch the Path class to use the mocked config dir
    with patch("app.utils.config.Path") as mock_path:
        # Setup path returns
        mock_path.side_effect = lambda p: config_dir.parent / p if str(p).startswith("config/") else Path(p)
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.stat.return_value.st_mtime = time.time()
        
        # Load config
        config = load_config(force_reload=True)
        
        # Validate basic config values
        assert "environment" in config
        assert config.get("environment") == "development"
        
        # Check values from default config
        assert config.get("server_host") == "0.0.0.0"
        
        # Check that environment variables take precedence
        assert "test_env_var" in config
        assert config.get("numeric_value") == 123
        assert config.get("float_value") == 3.14
        
        # Test env var boolean conversion
        assert isinstance(config.get("debug"), bool)

def test_config_change_callback():
    """Test configuration change callback registration and notification"""
    # Create a mock callback
    callback_mock = MagicMock()
    
    # Register the callback
    callback_id = register_config_change_callback(callback_mock)
    
    # Check callback ID format
    assert isinstance(callback_id, str)
    
    # Simulate configuration change
    old_config = {"key": "old_value"}
    new_config = {"key": "new_value"}
    
    # Import the module directly to access private function
    import app.utils.config as config_module
    config_module._notify_config_changes(old_config, new_config)
    
    # Verify callback was called
    callback_mock.assert_called_once_with(old_config, new_config)
    
    # Unregister the callback
    result = unregister_config_change_callback(callback_id)
    assert result == True
    
    # Reset mock and verify no more calls
    callback_mock.reset_mock()
    config_module._notify_config_changes(old_config, {"key": "newer_value"})
    callback_mock.assert_not_called()
    
    # Test with non-existent callback ID
    result = unregister_config_change_callback("nonexistent")
    assert result == False

def test_callback_with_watched_keys():
    """Test callback notification with specific watched keys"""
    # Create a simpler test with direct access to the callback mechanism
    import app.utils.config as config_module
    
    # First reset the global callbacks list to ensure clean state
    config_module._config_callbacks = []
    
    # Create mock callbacks
    callback_all = MagicMock()
    callback_specific = MagicMock()
    
    # Create callback objects directly
    callback_obj_all = config_module.ConfigChangeCallback(callback_all)
    callback_obj_specific = config_module.ConfigChangeCallback(callback_specific, ["key1", "nested.key"])
    
    # Add callback objects directly to the global list
    config_module._config_callbacks.append(callback_obj_all)
    config_module._config_callbacks.append(callback_obj_specific)
    
    # Test case 1: Change in unwatched key
    old_config = {"key1": "old1", "key2": "old2", "nested": {"key": "old_nested"}}
    new_config = {"key1": "old1", "key2": "new2", "nested": {"key": "old_nested"}}
    
    # Call the callbacks directly
    callback_obj_all(old_config, new_config)
    callback_obj_specific(old_config, new_config)
    
    # Verify first callback was called but not the second
    callback_all.assert_called_once_with(old_config, new_config)
    callback_specific.assert_not_called()
    
    # Reset mocks
    callback_all.reset_mock()
    callback_specific.reset_mock()
    
    # Test case 2: Change in watched key
    old_config = {"key1": "old1", "key2": "old2", "nested": {"key": "old_nested"}}
    new_config = {"key1": "new1", "key2": "old2", "nested": {"key": "old_nested"}}
    
    # Call the callbacks directly
    callback_obj_all(old_config, new_config)
    callback_obj_specific(old_config, new_config)
    
    # Verify both callbacks were called
    callback_all.assert_called_once_with(old_config, new_config)
    callback_specific.assert_called_once_with(old_config, new_config)

def test_config_watcher():
    """Test starting and stopping the configuration watcher"""
    # Start the watcher
    start_config_watcher()
    
    # Import the module directly to access private variables
    import app.utils.config as config_module
    
    # Check that the watcher is active
    assert config_module._watcher_active == True
    assert config_module._config_watcher_thread is not None
    assert config_module._config_watcher_thread.is_alive() == True
    
    # Set a shorter interval
    set_config_watcher_interval(1)
    assert config_module._watcher_interval == 1
    
    # Set a very short interval (should be adjusted to minimum)
    set_config_watcher_interval(0)
    assert config_module._watcher_interval == 1
    
    # Stop the watcher
    stop_config_watcher()
    
    # Check that the watcher is inactive
    assert config_module._watcher_active == False
    
    # Allow time for thread to terminate
    time.sleep(0.5)
    
    # Try stopping again (should be a no-op)
    stop_config_watcher()

@patch("platform.machine")
@patch("platform.system")
@patch("torch.backends.mps.is_available")
def test_detect_macos_m4(mock_mps_available, mock_system, mock_machine):
    """Test macOS M4 detection function"""
    # Test non-macOS platform
    mock_machine.return_value = "x86_64"
    mock_system.return_value = "Linux"
    assert detect_macos_m4() == False
    
    # Test macOS ARM but MPS not available
    mock_machine.return_value = "arm64"
    mock_system.return_value = "Darwin"
    mock_mps_available.return_value = False
    assert detect_macos_m4() == False
    
    # Test macOS ARM with MPS available
    mock_machine.return_value = "arm64"
    mock_system.return_value = "Darwin"
    mock_mps_available.return_value = True
    assert detect_macos_m4() == True
    
    # Test ImportError handling
    mock_mps_available.side_effect = ImportError("torch not available")
    assert detect_macos_m4() == False

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])