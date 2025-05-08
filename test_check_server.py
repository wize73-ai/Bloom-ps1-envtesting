#!/usr/bin/env python3
"""
Simple script to check server status and debug issues.
"""

import requests
import json
import sys

# Server URL
SERVER_URL = "http://localhost:8000"

def check_server_health():
    """Check if the server is up and running."""
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code == 200:
            print(f"Server is healthy. Response: {response.json()}")
        else:
            print(f"Server returned status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error connecting to server: {str(e)}")

def test_simple_request():
    """Test a simple API request."""
    try:
        # Simple language detection request
        data = {
            "text": "Hello world"
        }
        response = requests.post(f"{SERVER_URL}/pipeline/detect", json=data)
        if response.status_code == 200:
            print(f"Language detection successful. Response: {response.json()}")
        else:
            print(f"Language detection failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error making language detection request: {str(e)}")

if __name__ == "__main__":
    print("Checking server health...")
    check_server_health()
    
    print("\nTesting simple language detection request...")
    test_simple_request()