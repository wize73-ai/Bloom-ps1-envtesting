#!/usr/bin/env python3
"""
Test the simplification API with domain-specific simplification and domain options.
"""

import requests
import json
import sys

# ANSI color codes for prettier output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
ENDC = "\033[0m"
BOLD = "\033[1m"

# Test server URL
SERVER_URL = "http://localhost:8000"
SIMPLIFY_ENDPOINT = "/pipeline/simplify"

def test_legal_domain_simplification():
    """Test domain-specific simplification for legal texts."""
    print(f"\n{BOLD}{BLUE}Testing Domain-Specific Simplification{ENDC}")
    print("-" * 80)
    
    # Legal test text
    text = "The tenant shall indemnify and hold harmless the landlord from and against any and all claims, actions, suits, judgments and demands brought or recovered against the landlord by reason of any negligent or willful act or omission of the tenant."
    
    print(f"{BOLD}Test Text:{ENDC} {text}")
    print("-" * 80)
    
    # Test with and without domain specification
    domains = [None, "legal", "legal-housing"]
    
    for domain in domains:
        domain_name = domain if domain else "none"
        print(f"\n{BOLD}Testing with domain: {domain_name}{ENDC}")
        
        # Test each simplification level (1-5)
        for level in range(1, 6):
            print(f"\n{BOLD}Level {level}:{ENDC}")
            
            # Prepare request data
            data = {
                "text": text,
                "language": "en",
                "target_level": str(level)
            }
            
            # Add domain if specified
            if domain:
                data["parameters"] = {"domain": domain}
            
            # Make request
            try:
                response = requests.post(
                    f"{SERVER_URL}{SIMPLIFY_ENDPOINT}",
                    json=data
                )
                
                # Process response
                if response.status_code == 200:
                    result = response.json()
                    simplified_text = result["data"]["simplified_text"]
                    model_used = result["data"]["model_used"]
                    
                    print(f"{GREEN}Model: {model_used}{ENDC}")
                    print(simplified_text)
                else:
                    print(f"{RED}Request failed: {response.status_code}{ENDC}")
                    print(response.text)
            except Exception as e:
                print(f"{RED}Error: {str(e)}{ENDC}")

if __name__ == "__main__":
    test_legal_domain_simplification()