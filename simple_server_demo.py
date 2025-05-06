#!/usr/bin/env python3
"""
CasaLingua Simple Server Demo

This script tests the running CasaLingua server in a very simple way.
It focuses on demonstrating:
- Translation
- Simplification
- Health status

The demo runs for approximately 2 minutes, focusing on reliability over fancy features.
"""

import os
import sys
import time
import json
import random
import subprocess
import requests
import datetime

# Basic colored output, no external dependencies
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Sample texts for demonstration
SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Learning a new language opens doors to new cultures and perspectives.",
    "The housing agreement must be signed by all tenants prior to occupancy.",
    "The patient should take this medication twice daily with food.",
    "Climate change is one of the most pressing challenges of our time.",
]

COMPLEX_SENTENCES = [
    "The aforementioned contractual obligations shall be considered null and void if the party of the first part fails to remit payment within the specified timeframe.",
    "Notwithstanding the provisions outlined in section 3.2, the tenant hereby acknowledges that the landlord retains the right to access the premises for inspection purposes given reasonable notice.",
    "The novel's byzantine plot structure, replete with labyrinthine narrative diversions and oblique character motivations, confounded even the most perspicacious readers.",
    "The acquisition of language proficiency necessitates consistent immersion in linguistic contexts that facilitate the assimilation of vocabulary and grammatical constructs.",
]

TARGET_LANGUAGES = ["es", "fr", "de"]
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "zh": "Chinese"
}

API_BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def print_header(title):
    """Print a nicely formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*50}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title.center(50)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*50}{Colors.ENDC}\n")

def print_success(message):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def print_error(message):
    """Print an error message"""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def print_info(message):
    """Print an info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")

def check_server_availability():
    """Check if the server is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            return True
        return False
    except requests.RequestException:
        return False

def get_system_memory():
    """Get system memory information"""
    memory_info = {"total": "Unknown", "available": "Unknown", "used": "Unknown"}
    
    try:
        # Try to use psutil if available
        import psutil
        vm = psutil.virtual_memory()
        memory_info["total"] = f"{vm.total / (1024**3):.2f} GB"
        memory_info["available"] = f"{vm.available / (1024**3):.2f} GB"
        memory_info["used"] = f"{vm.used / (1024**3):.2f} GB"
    except ImportError:
        # Fall back to subprocess
        try:
            if sys.platform == "darwin":  # macOS
                cmd = "top -l 1 | grep PhysMem"
                output = subprocess.check_output(cmd, shell=True).decode().strip()
                memory_info["used"] = output.split(",")[0].split(":")[1].strip()
                memory_info["available"] = output.split(",")[1].strip()
            elif sys.platform == "linux":
                cmd = "free -h | grep Mem:"
                output = subprocess.check_output(cmd, shell=True).decode().strip()
                parts = output.split()
                memory_info["total"] = parts[1]
                memory_info["used"] = parts[2]
                memory_info["available"] = parts[6]
        except:
            pass
    
    return memory_info

def check_server_process():
    """Check for the server process"""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['name'] == 'python' and any('main.py' in cmd for cmd in proc.info['cmdline'] if cmd):
                return proc.pid
        return None
    except ImportError:
        return None

def check_health():
    """Check system health and display results"""
    print_header("CasaLingua System Health Check")
    
    # Check if API is running
    try:
        print("Checking API status...")
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            print_success("API is online")
            
            health_data = response.json()
            
            # Check detailed health if available
            try:
                print("Fetching detailed health information...")
                detailed_response = requests.get(f"{API_BASE_URL}/health/detailed", timeout=5)
                
                if detailed_response.status_code == 200:
                    detailed_data = detailed_response.json()
                    
                    # Display model information
                    if "models" in detailed_data:
                        print_info(f"Loaded Models: {len(detailed_data['models'])}")
                        for model in detailed_data["models"]:
                            model_name = model.get("name", "Unknown")
                            model_status = model.get("status", "unknown")
                            if model_status == "loaded":
                                print_success(f"  Model: {model_name}")
                            else:
                                print_warning(f"  Model: {model_name} ({model_status})")
                    
                    # Display system information
                    if "system" in detailed_data:
                        sys_info = detailed_data["system"]
                        print_info(f"System Memory: {sys_info.get('memory_available', 'Unknown')} available")
                        print_info(f"System Load: {sys_info.get('cpu_usage', 'Unknown')}%")
                else:
                    print_warning("Detailed health information not available")
            
            except requests.RequestException:
                print_warning("Detailed health check failed")
        else:
            print_error(f"API health check failed ({response.status_code})")
    except requests.RequestException:
        print_error("API is offline or unreachable")
    
    # Display memory information
    memory_info = get_system_memory()
    print_info("\nSystem Memory:")
    print_info(f"  Total:     {memory_info['total']}")
    print_info(f"  Used:      {memory_info['used']}")
    print_info(f"  Available: {memory_info['available']}")
    
    # Check server process
    pid = check_server_process()
    if pid:
        print_info(f"Server process running with PID: {pid}")
    
    print()

def demonstrate_translation():
    """Demonstrate translation capabilities using the API"""
    print_header("Translation Demonstration")
    
    # Select a random sentence
    text = random.choice(SAMPLE_SENTENCES)
    target_lang = random.choice(TARGET_LANGUAGES)
    
    # Display source text
    print_info(f"Source Text (English):")
    print(f'"{text}"')
    print()
    
    # Prepare request
    payload = {
        "text": text,
        "source_language": "en",
        "target_language": target_lang,
        "preserve_formatting": True
    }
    
    # Call the translation API
    try:
        print(f"Translating to {LANGUAGE_NAMES.get(target_lang)}...")
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/pipeline/translate", 
            headers=HEADERS,
            json=payload,
            timeout=10
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Display result
            lang_name = LANGUAGE_NAMES.get(target_lang, target_lang.upper())
            translated_text = result.get("data", {}).get("translated_text", "Translation failed")
            model_used = result.get("data", {}).get("model_used", "Unknown")
            
            print_success(f"Translation complete in {elapsed:.2f} seconds")
            print_info(f"Translated Text ({lang_name}):")
            print(f'"{translated_text}"')
            print_info(f"Model Used: {model_used}")
        else:
            print_error(f"Translation failed ({response.status_code})")
            try:
                error_data = response.json()
                print_error(f"Error: {error_data.get('message', 'Unknown error')}")
            except:
                print_error(f"Error: {response.text}")
            
    except requests.RequestException as e:
        print_error(f"Translation request failed: {str(e)}")
    
    print()

def demonstrate_simplification():
    """Demonstrate text simplification capabilities using the API"""
    print_header("Text Simplification Demonstration")
    
    # Select a complex sentence
    text = random.choice(COMPLEX_SENTENCES)
    
    # Display source text
    print_info(f"Complex Text:")
    print(f'"{text}"')
    print()
    
    # Prepare request
    payload = {
        "text": text,
        "target_grade_level": "5"  # Simplify to 5th grade reading level
    }
    
    # Call the simplification API
    try:
        print("Simplifying text to 5th grade reading level...")
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/pipeline/simplify", 
            headers=HEADERS,
            json=payload,
            timeout=15
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Display result
            simplified_text = result.get("data", {}).get("simplified_text", "Simplification failed")
            
            print_success(f"Simplification complete in {elapsed:.2f} seconds")
            print_info(f"Simplified Text (5th grade level):")
            print(f'"{simplified_text}"')
        else:
            print_error(f"Simplification failed ({response.status_code})")
            try:
                error_data = response.json()
                print_error(f"Error: {error_data.get('message', 'Unknown error')}")
            except:
                print_error(f"Error: {response.text}")
            
    except requests.RequestException as e:
        print_error(f"Simplification request failed: {str(e)}")
    
    print()

def run_memory_check():
    """Run a simple memory check"""
    print_header("Memory Usage Check")
    
    # Check process memory
    try:
        import psutil
        server_process = None
        
        # Try to find the Python process running the server
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['name'] == 'python' and any('main.py' in cmd for cmd in proc.info['cmdline'] if cmd):
                server_process = proc
                break
        
        if server_process:
            memory_info = server_process.memory_info()
            print_info(f"Server process memory: {memory_info.rss / (1024**3):.2f} GB")
            
            # Try to get detailed process memory
            try:
                import resource
                max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                print_info(f"Peak memory usage: {max_rss / (1024**2):.2f} GB")
            except:
                pass
            
            # Check if GPU is being used
            try:
                import torch
                if torch.cuda.is_available():
                    print_info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
                    print_info(f"GPU Memory cached: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
            except:
                pass
    except ImportError:
        print_warning("Psutil not available, cannot get detailed memory info")
    
    # Get system memory
    memory_info = get_system_memory()
    print_info(f"System memory total: {memory_info['total']}")
    print_info(f"System memory used: {memory_info['used']}")
    print_info(f"System memory available: {memory_info['available']}")
    
    print()

def main():
    """Main function to run the demo"""
    try:
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"{Colors.BOLD}{Colors.BLUE}CasaLingua Server Demo{Colors.ENDC}")
        print(f"{Colors.BLUE}Testing the running CasaLingua server{Colors.ENDC}")
        print(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check if server is available
        server_available = check_server_availability()
        if not server_available:
            print_error("ERROR: CasaLingua server is not accessible!")
            print("Please make sure the server is running at http://localhost:8000")
            print("Exiting demo.")
            return 1
        
        # Set start and end times
        start_time = time.time()
        end_time = start_time + 120  # 2 minutes
        
        # First run health check
        check_health()
        
        # Main demo loop
        demo_sequence = [
            demonstrate_translation,
            demonstrate_simplification,
            run_memory_check
        ]
        
        sequence_index = 0
        while time.time() < end_time:
            # Run the next demo in sequence
            demo_sequence[sequence_index]()
            
            # Move to next demo
            sequence_index = (sequence_index + 1) % len(demo_sequence)
            
            # Show remaining time
            remaining = int(end_time - time.time())
            if remaining > 0:
                print(f"Demo will continue for approximately {remaining} more seconds...")
                print()
            
            # Short delay between demonstrations
            time.sleep(1)
        
        # Completion message
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*50}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.GREEN}Demo Complete!{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*50}{Colors.ENDC}")
        print("\nThank you for exploring CasaLingua's capabilities!")
        print()
        return 0
        
    except KeyboardInterrupt:
        print_warning("\nDemo interrupted by user")
        return 1
    except Exception as e:
        print_error(f"\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())