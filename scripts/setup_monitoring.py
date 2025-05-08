#!/usr/bin/env python3
"""
Setup script for enhancing logging, metrics, and veracity monitoring in CasaLingua.

This script ensures that:
1. Log files are properly created and configured
2. Metrics collection is enabled
3. Veracity checking is properly configured
"""
import os
import sys
import json
import subprocess
from pathlib import Path

# Directories
BASE_DIR = "/Users/jameswilson/Desktop/PRODUCTION/test/casMay4"
LOG_DIR = f"{BASE_DIR}/logs"
CONFIG_DIR = f"{BASE_DIR}/config"

def ensure_dir_exists(path):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def update_config_file(file_path, updates):
    """Update a JSON configuration file with new values."""
    if not os.path.exists(file_path):
        print(f"Config file not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        # Apply updates
        for key, value in updates.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                # For nested dictionaries, update instead of replace
                config[key].update(value)
            else:
                # For simple values, replace
                config[key] = value
        
        # Write back the updated config
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Updated config file: {file_path}")
        print(f"Applied updates: {json.dumps(updates, indent=2)}")
        return True
    except Exception as e:
        print(f"Error updating config file: {e}")
        return False

def main():
    """Main function to set up monitoring."""
    # Ensure log directory exists
    ensure_dir_exists(LOG_DIR)
    
    # Create empty log files if they don't exist
    log_files = [
        "server.log",
        "metrics.log",
        "veracity.log",
        "errors.log"
    ]
    
    for log_file in log_files:
        log_path = os.path.join(LOG_DIR, log_file)
        if not os.path.exists(log_path):
            with open(log_path, 'w') as f:
                pass  # Create empty file
            print(f"Created log file: {log_path}")
        else:
            print(f"Log file already exists: {log_path}")
    
    # Update development.json with monitoring settings
    dev_config_path = os.path.join(CONFIG_DIR, "development.json")
    monitoring_updates = {
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/server.log",
            "max_size": 10485760,  # 10 MB
            "backup_count": 5
        },
        "metrics": {
            "enabled": True,
            "file": "logs/metrics.log",
            "interval": 60,  # seconds
            "include_memory": True,
            "include_performance": True
        },
        "veracity": {
            "enabled": True,
            "log_file": "logs/veracity.log",
            "min_score": 0.7,
            "check_translations": True,
            "check_simplifications": True,
            "check_summaries": True,
            "metrics": {
                "accuracy_threshold": 0.8,
                "content_integrity_threshold": 0.85,
                "semantic_threshold": 0.7
            }
        },
        "audit": {
            "enabled": True,
            "log_file": "logs/audit.log",
            "include_request_headers": False,
            "include_response_headers": False
        }
    }
    
    update_config_file(dev_config_path, monitoring_updates)
    
    # Restart is needed to apply changes
    print("\nMonitoring has been configured. Please restart the server for changes to take effect.")
    print("You can restart the server with: cd", BASE_DIR, "&& python -m app.main")

if __name__ == "__main__":
    main()