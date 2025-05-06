#!/usr/bin/env python3
"""
CasaLingua Minimal Demo

The simplest possible demo for a teaching session.
Just makes a few API requests and shows the results.
"""

import os
import sys
import time
import subprocess
import json
import requests
import random

API_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

# Sample texts for demonstration
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Learning a new language opens doors to new cultures and perspectives.",
    "The housing agreement must be signed by all tenants prior to occupancy."
]

COMPLEX_TEXTS = [
    "The aforementioned contractual obligations shall be considered null and void if the party of the first part fails to remit payment within the specified timeframe.",
    "The novel's byzantine plot structure, replete with labyrinthine narrative diversions and oblique character motivations, confounded even the most perspicacious readers."
]

LANGUAGES = ["es", "fr", "de"]
LANG_NAMES = {"es": "Spanish", "fr": "French", "de": "German"}

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_separator(title=None):
    """Print a separator line with optional title"""
    width = 80
    print("\n" + "=" * width)
    if title:
        title_str = f" {title} "
        padding = (width - len(title_str)) // 2
        print(" " * padding + title_str)
        print("=" * width)
    print()

def check_health():
    """Check server health"""
    print("Checking API health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✓ API is online")
            print(f"Status: {result.get('status', 'Unknown')}")
            return True
        else:
            print(f"✗ API health check failed ({response.status_code})")
            return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def translate_text(text, target_lang):
    """Translate text to target language"""
    print(f"Translating to {LANG_NAMES.get(target_lang, target_lang)}:")
    print(f"Input: \"{text}\"")
    
    payload = {
        "text": text,
        "source_language": "en",
        "target_language": target_lang
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/pipeline/translate",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            data = result.get("data", {})
            
            if "translated_text" in data:
                print(f"Translation: \"{data['translated_text']}\"")
                print(f"Model used: {data.get('model_used', 'Unknown')}")
                print(f"Time: {elapsed:.2f} seconds")
                return True
            else:
                print("✗ No translation in response")
                return False
        else:
            print(f"✗ Translation failed ({response.status_code})")
            return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def simplify_text(text):
    """Simplify complex text"""
    print(f"Simplifying text:")
    print(f"Input: \"{text}\"")
    
    payload = {
        "text": text,
        "target_grade_level": "5"
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/pipeline/simplify",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            data = result.get("data", {})
            
            if "simplified_text" in data:
                print(f"Simplified: \"{data['simplified_text']}\"")
                print(f"Time: {elapsed:.2f} seconds")
                return True
            else:
                print("✗ No simplification in response")
                return False
        else:
            print(f"✗ Simplification failed ({response.status_code})")
            return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def run_demo(duration=120):
    """Run the demo for the specified duration"""
    clear_screen()
    print_separator("CasaLingua Minimal Demo")
    
    print("This demo will run for 2 minutes, showing examples of:")
    print("- Translation")
    print("- Text simplification")
    print("- Health monitoring")
    print("\nPress Ctrl+C at any time to stop the demo.\n")
    
    # Check health first
    if not check_health():
        print("\nServer health check failed. Would you like to continue anyway? (y/n)")
        if input().lower() != 'y':
            return 1
    
    # Set up timing
    start_time = time.time()
    end_time = start_time + duration
    
    # Main demo loop
    while time.time() < end_time:
        remaining_time = int(end_time - time.time())
        
        # Choose a random operation
        operation = random.choice(["translate", "simplify", "health"])
        
        if operation == "translate":
            print_separator("Translation Demo")
            translate_text(random.choice(TEXTS), random.choice(LANGUAGES))
        elif operation == "simplify":
            print_separator("Simplification Demo")
            simplify_text(random.choice(COMPLEX_TEXTS))
        else:
            print_separator("Health Check")
            check_health()
            
        # Show time remaining
        if remaining_time > 0:
            print(f"\nDemo will continue for approximately {remaining_time} more seconds...")
            time.sleep(5)  # Pause between operations
    
    print_separator("Demo Complete")
    print("Thank you for exploring CasaLingua!")
    return 0

def main():
    try:
        return run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        return 0
    except Exception as e:
        print(f"\n\nError during demo: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())