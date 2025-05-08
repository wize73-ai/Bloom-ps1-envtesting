#!/usr/bin/env python3
"""
Test script for the simplification endpoint in CasaLingua.

This script tests the simplification endpoint with different levels to
verify our enhancements are working properly on the running server.
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

# Test cases with different complexity levels
TEST_CASES = [
    {
        "name": "Legal Text",
        "text": "The tenant shall indemnify and hold harmless the landlord from and against any and all claims, actions, suits, judgments and demands brought or recovered against the landlord by reason of any negligent or willful act or omission of the tenant, its agents, servants, employees, licensees, visitors, invitees, and any of the tenant's contractors and subcontractors.",
    },
    {
        "name": "Technical Text",
        "text": "Prior to commencement of the installation process, ensure that all prerequisite components have been obtained and are readily accessible for utilization. Failure to verify component availability may result in procedural delays.",
    },
    {
        "name": "Financial Text",
        "text": "The applicant must furnish documentation verifying income and employment status in accordance with the requirements delineated in section 8 of the aforementioned application procedure.",
    }
]

def test_simplification_levels():
    """Test the simplification endpoint with different levels."""
    print(f"\n{BOLD}{BLUE}Testing Simplification Endpoint with Different Levels{ENDC}")
    print("-" * 80)
    
    # Check endpoint accessibility with a test request
    try:
        test_data = {
            "text": "Test text",
            "language": "en",
            "target_level": "1"  # String format as required by API
        }
        response = requests.post(f"{SERVER_URL}{SIMPLIFY_ENDPOINT}", json=test_data)
        if response.status_code >= 400:
            print(f"{RED}Simplification endpoint not accessible. Status: {response.status_code}{ENDC}")
            print(f"Response: {response.text}")
            sys.exit(1)
    except Exception as e:
        print(f"{RED}Error accessing simplification endpoint: {str(e)}{ENDC}")
        sys.exit(1)
        
    print(f"{GREEN}Simplification endpoint is accessible{ENDC}")
    
    # Test each case with different levels
    for test_case in TEST_CASES:
        print(f"\n{BOLD}Test Case: {test_case['name']}{ENDC}")
        print(f"{BOLD}Original Text:{ENDC} {test_case['text']}")
        print("-" * 80)
        
        # Test each simplification level (1-5)
        simplified_texts = []
        for level in range(1, 6):
            print(f"\n{BOLD}Testing Level {level}:{ENDC}")
            
            # Prepare request data
            data = {
                "text": test_case["text"],
                "language": "en",
                "target_level": str(level)  # String format as required by API
            }
            
            # Make request
            try:
                response = requests.post(
                    f"{SERVER_URL}{SIMPLIFY_ENDPOINT}",
                    json=data
                )
                
                # Process response
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract simplified text - checking all possible response formats
                    simplified_text = None
                    if "simplified_text" in result:
                        simplified_text = result["simplified_text"]
                    elif "data" in result and isinstance(result["data"], dict):
                        if "simplified_text" in result["data"]:
                            simplified_text = result["data"]["simplified_text"]
                        elif "text" in result["data"]:
                            simplified_text = result["data"]["text"]
                    
                    # Print the full response for debugging
                    print(f"{BLUE}Response structure: {json.dumps(result, indent=2)}{ENDC}")
                    
                    if simplified_text:
                        simplified_texts.append(simplified_text)
                        print(f"{simplified_text}")
                        
                        # Calculate difference from original
                        orig_words = len(test_case["text"].split())
                        simp_words = len(simplified_text.split())
                        word_diff = (simp_words - orig_words) / orig_words * 100
                        
                        if word_diff < 0:
                            print(f"{GREEN}Words: {simp_words} ({word_diff:.1f}% from original){ENDC}")
                        else:
                            print(f"{YELLOW}Words: {simp_words} ({word_diff:.1f}% from original){ENDC}")
                    else:
                        print(f"{RED}Could not extract simplified text from response{ENDC}")
                        simplified_texts.append(None)
                else:
                    print(f"{RED}Request failed with status code: {response.status_code}{ENDC}")
                    print(f"Response: {response.text}")
                    simplified_texts.append(None)
            except Exception as e:
                print(f"{RED}Error making request: {str(e)}{ENDC}")
                simplified_texts.append(None)
        
        # Check uniqueness of results
        valid_texts = [t for t in simplified_texts if t]
        unique_texts = set(valid_texts)
        
        print(f"\n{BOLD}Summary for {test_case['name']}:{ENDC}")
        print(f"Unique simplified outputs: {len(unique_texts)} out of {len(valid_texts)} levels")
        
        if len(unique_texts) >= 4:
            print(f"{GREEN}The simplification levels are producing different outputs as expected.{ENDC}")
        elif len(unique_texts) >= 3:
            print(f"{YELLOW}The simplification levels are producing somewhat different outputs.{ENDC}")
        else:
            print(f"{RED}The simplification levels are not producing sufficiently different outputs.{ENDC}")
        
        print("-" * 80)

if __name__ == "__main__":
    test_simplification_levels()