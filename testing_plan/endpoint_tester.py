#!/usr/bin/env python3
"""
Comprehensive API Endpoint Testing Script for CasaLingua

This script systematically tests all API endpoints in the CasaLingua application,
recording results and helping identify issues that need fixing.

Usage:
    python endpoint_tester.py [--url BASE_URL] [--env ENVIRONMENT] [--group GROUP_NAME]

Options:
    --url BASE_URL       Base URL of the API (default: http://localhost:8000)
    --env ENVIRONMENT    Environment setting (default: development)
    --group GROUP_NAME   Only test endpoints in this group (default: all)
                         Options: health, translation, language, text_processing, 
                                 admin, rag, streaming, bloom
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import aiohttp
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"endpoint_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("endpoint_tester")

# ANSI colors for better output readability
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_success(message):
    logger.info(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def log_failure(message):
    logger.error(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def log_warning(message):
    logger.warning(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def log_info(message):
    logger.info(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")

def log_header(message):
    logger.info(f"\n{Colors.HEADER}{Colors.BOLD}=== {message} ==={Colors.ENDC}")

class EndpointTester:
    def __init__(self, base_url: str, env: str = "development"):
        self.base_url = base_url
        self.env = env
        self.session = None
        self.results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "tests": [],
            "timestamp": datetime.now().isoformat(),
            "base_url": base_url,
            "environment": env
        }
        
        # Store auth token if obtained
        self.auth_token = None
        
        # API test definitions (method, endpoint, payload, expected_status, description, group)
        self.test_definitions = self._get_test_definitions()
    
    def _get_test_definitions(self) -> List[Dict[str, Any]]:
        """Define all API test cases"""
        return [
            # Health endpoints (Group: health)
            {"method": "GET", "endpoint": "/health", "payload": None, "expected_status": 200, 
                "description": "Basic health check", "group": "health"},
            {"method": "GET", "endpoint": "/health/detailed", "payload": None, "expected_status": 200, 
                "description": "Detailed health check", "group": "health"},
            {"method": "GET", "endpoint": "/health/models", "payload": None, "expected_status": 200, 
                "description": "Model health check", "group": "health"},
            {"method": "GET", "endpoint": "/health/database", "payload": None, "expected_status": 200, 
                "description": "Database health check", "group": "health"},
            {"method": "GET", "endpoint": "/readiness", "payload": None, "expected_status": 200, 
                "description": "Readiness probe", "group": "health"},
            {"method": "GET", "endpoint": "/liveness", "payload": None, "expected_status": 200, 
                "description": "Liveness probe", "group": "health"},
            {"method": "GET", "endpoint": "/health/metrics", "payload": None, "expected_status": 200, 
                "description": "Memory metrics", "group": "health"},
            
            # Translation endpoints (Group: translation)
            {"method": "POST", "endpoint": "/pipeline/translate", 
                "payload": {"text": "Hello, how are you?", "source_language": "en", "target_language": "es"}, 
                "expected_status": 200, "description": "Basic text translation (EN to ES)", "group": "translation"},
            {"method": "POST", "endpoint": "/pipeline/translate", 
                "payload": {"text": "Bonjour, comment ça va?", "source_language": "auto", "target_language": "en"}, 
                "expected_status": 200, "description": "Translation with auto-detection (FR to EN)", "group": "translation"},
            {"method": "POST", "endpoint": "/pipeline/translate/batch", 
                "payload": {"texts": ["Hello, how are you?", "The weather is nice today"], 
                            "source_language": "en", "target_language": "es"}, 
                "expected_status": 200, "description": "Batch translation (EN to ES)", "group": "translation"},
            
            # Language detection endpoints (Group: language)
            {"method": "POST", "endpoint": "/pipeline/detect", 
                "payload": {"text": "Hello, how are you?"}, 
                "expected_status": 200, "description": "Language detection (English)", "group": "language"},
            {"method": "POST", "endpoint": "/pipeline/detect", 
                "payload": {"text": "Hola, ¿cómo estás?", "detailed": True}, 
                "expected_status": 200, "description": "Detailed language detection (Spanish)", "group": "language"},
            {"method": "POST", "endpoint": "/pipeline/detect-language", 
                "payload": {"text": "Guten Tag, wie geht es Ihnen?"}, 
                "expected_status": 200, "description": "Language detection alias endpoint (German)", "group": "language"},
            
            # Text processing endpoints (Group: text_processing)
            {"method": "POST", "endpoint": "/pipeline/simplify", 
                "payload": {"text": "The mitochondrion is a double membrane-bound organelle found in most eukaryotic organisms.", 
                            "language": "en", "target_level": "simple"}, 
                "expected_status": 200, "description": "Text simplification (English)", "group": "text_processing"},
            {"method": "POST", "endpoint": "/pipeline/anonymize", 
                "payload": {"text": "John Smith lives at 123 Main St, New York and his email is john.smith@example.com.", 
                            "language": "en", "strategy": "mask"}, 
                "expected_status": 200, "description": "Text anonymization (English)", "group": "text_processing"},
            {"method": "POST", "endpoint": "/pipeline/analyze", 
                "payload": {"text": "I love this product! It's amazing and works really well.", 
                            "language": "en", "analyses": ["sentiment"]}, 
                "expected_status": 200, "description": "Sentiment analysis (Positive)", "group": "text_processing"},
            {"method": "POST", "endpoint": "/pipeline/analyze", 
                "payload": {"text": "Apple Inc. is headquartered in Cupertino, California.", 
                            "language": "en", "analyses": ["entities"]}, 
                "expected_status": 200, "description": "Entity recognition", "group": "text_processing"},
            {"method": "POST", "endpoint": "/pipeline/summarize", 
                "payload": {"text": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.", 
                            "language": "en"}, 
                "expected_status": 200, "description": "Text summarization (English)", "group": "text_processing"},
            
            # Admin endpoints (Group: admin)
            {"method": "GET", "endpoint": "/system/info", "payload": None, 
                "expected_status": 200, "description": "Get system information", "group": "admin"},
            {"method": "GET", "endpoint": "/system/config", "payload": None, 
                "expected_status": 200, "description": "Get system configuration", "group": "admin"},
            {"method": "GET", "endpoint": "/models", "payload": None, 
                "expected_status": 200, "description": "List available models", "group": "admin"},
            {"method": "GET", "endpoint": "/languages", "payload": None, 
                "expected_status": 200, "description": "List supported languages", "group": "admin"},
            {"method": "GET", "endpoint": "/metrics", "payload": None, 
                "expected_status": 200, "description": "Get system metrics", "group": "admin"},
            
            # RAG endpoints (Group: rag)
            {"method": "POST", "endpoint": "/rag/query", 
                "payload": {"query": "How does translation work?", "language": "en", "max_results": 3}, 
                "expected_status": 200, "description": "Query knowledge base", "group": "rag"},
            {"method": "POST", "endpoint": "/rag/translate", 
                "payload": {"text": "The court ruled in favor of the plaintiff.", "source_language": "en", 
                           "target_language": "es", "max_retrieval_results": 3}, 
                "expected_status": 200, "description": "RAG-enhanced translation", "group": "rag"},
            
            # Streaming endpoints (Group: streaming)
            {"method": "POST", "endpoint": "/streaming/translate", 
                "payload": {"text": "Hello, how are you?", "source_language": "en", "target_language": "es"}, 
                "expected_status": 200, "description": "Stream translation", "group": "streaming"},
            
            # Bloom Housing compatibility endpoints (Group: bloom)
            {"method": "POST", "endpoint": "/bloom-housing/translate", 
                "payload": {"text": "Welcome to our housing portal", "sourceLanguage": "en", "targetLanguage": "es"}, 
                "expected_status": 200, "description": "Bloom Housing translation", "group": "bloom"},
            {"method": "POST", "endpoint": "/bloom-housing/detect-language", 
                "payload": {"text": "Welcome to our housing portal"}, 
                "expected_status": 200, "description": "Bloom Housing language detection", "group": "bloom"}
        ]
    
    async def setup(self):
        """Initialize the HTTP session with proper headers"""
        # Set environment variable for auth bypass if needed
        if self.env == "development":
            os.environ["CASALINGUA_ENV"] = "development"
            log_info(f"Setting CASALINGUA_ENV=development for auth bypass")
        
        # Create HTTP session
        self.session = aiohttp.ClientSession(
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
    
    async def teardown(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
        
        # Unset environment variable
        if "CASALINGUA_ENV" in os.environ:
            del os.environ["CASALINGUA_ENV"]
    
    async def test_endpoint(self, method: str, endpoint: str, payload: Optional[Dict] = None, 
                         expected_status: int = 200, description: str = "", group: str = "misc") -> Tuple[Optional[int], Any]:
        """Test an API endpoint and record the result"""
        self.results["total"] += 1
        url = f"{self.base_url}{endpoint}"
        
        log_info(f"Testing {method} {endpoint} - {description}")
        
        start_time = time.time()
        try:
            headers = self.session.headers.copy()
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
                
            if method.upper() == "GET":
                async with self.session.get(url, headers=headers) as response:
                    status = response.status
                    try:
                        data = await response.json()
                    except:
                        data = await response.text()
            elif method.upper() == "POST":
                async with self.session.post(url, json=payload, headers=headers) as response:
                    status = response.status
                    try:
                        data = await response.json()
                    except:
                        data = await response.text()
            elif method.upper() == "DELETE":
                async with self.session.delete(url, headers=headers) as response:
                    status = response.status
                    try:
                        data = await response.json()
                    except:
                        data = await response.text()
            else:
                log_failure(f"Unsupported method: {method}")
                return None, f"Unsupported method: {method}"
            
            duration = time.time() - start_time
            
            # Format the response data for display
            if isinstance(data, dict):
                formatted_data = json.dumps(data, indent=2)
                # Truncate if too long
                if len(formatted_data) > 500:
                    formatted_data = formatted_data[:500] + "..."
            else:
                formatted_data = str(data)
                if len(formatted_data) > 500:
                    formatted_data = formatted_data[:500] + "..."
            
            # Check if status matches expected
            if status == expected_status:
                log_success(f"Status: {status} (Expected: {expected_status}) - {duration:.2f}s")
                self.results["passed"] += 1
                
                # Additional validation for successful responses
                if status == 200 and isinstance(data, dict):
                    if "status" in data and data["status"] == "success":
                        log_success(f"Response indicates success")
                    elif "status" in data and data["status"] != "success":
                        log_warning(f"Response status is {data['status']}, expected 'success'")
            else:
                log_failure(f"Status: {status} (Expected: {expected_status}) - {duration:.2f}s")
                self.results["failed"] += 1
            
            log_info(f"Response: {formatted_data}")
            
            # Record test result
            self.results["tests"].append({
                "endpoint": endpoint,
                "method": method,
                "description": description,
                "group": group,
                "status": status,
                "expected_status": expected_status,
                "passed": status == expected_status,
                "duration": duration,
                "response": data if isinstance(data, dict) else str(data),
                "timestamp": datetime.now().isoformat()
            })
            
            return status, data
            
        except Exception as e:
            duration = time.time() - start_time
            log_failure(f"Request failed: {str(e)} - {duration:.2f}s")
            self.results["failed"] += 1
            
            # Record test failure
            self.results["tests"].append({
                "endpoint": endpoint,
                "method": method,
                "description": description,
                "group": group,
                "status": None,
                "expected_status": expected_status,
                "passed": False,
                "duration": duration,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            
            return None, str(e)
    
    async def run_tests(self, group: Optional[str] = None):
        """Run tests for all endpoints or a specific group"""
        await self.setup()
        
        try:
            log_header(f"Running API Endpoint Tests for {self.base_url}")
            log_info(f"Environment: {self.env}")
            
            if group:
                log_info(f"Testing only group: {group}")
                tests = [t for t in self.test_definitions if t["group"] == group]
                if not tests:
                    log_warning(f"No tests found for group: {group}")
                    return 1
            else:
                tests = self.test_definitions
            
            # Group tests by their group for better organization
            test_groups = {}
            for test in tests:
                test_group = test["group"]
                if test_group not in test_groups:
                    test_groups[test_group] = []
                test_groups[test_group].append(test)
            
            # Run tests by group
            for test_group, group_tests in test_groups.items():
                log_header(f"Testing {test_group.upper()} Endpoints")
                
                for test in group_tests:
                    await self.test_endpoint(
                        method=test["method"],
                        endpoint=test["endpoint"],
                        payload=test["payload"],
                        expected_status=test["expected_status"],
                        description=test["description"],
                        group=test["group"]
                    )
            
            # Print summary and save results
            return await self.print_summary()
            
        finally:
            await self.teardown()
    
    async def print_summary(self):
        """Print a summary of all test results and save to file"""
        log_header("Test Summary")
        
        # Calculate overall pass rate
        pass_rate = (self.results["passed"] / self.results["total"]) * 100 if self.results["total"] > 0 else 0
        
        log_info(f"Total Tests: {self.results['total']}")
        log_success(f"Passed: {self.results['passed']}")
        log_failure(f"Failed: {self.results['failed']}")
        
        if self.results["skipped"] > 0:
            log_warning(f"Skipped: {self.results['skipped']}")
        
        if pass_rate >= 90:
            log_success(f"Pass Rate: {pass_rate:.1f}%")
        elif pass_rate >= 75:
            log_warning(f"Pass Rate: {pass_rate:.1f}%")
        else:
            log_failure(f"Pass Rate: {pass_rate:.1f}%")
        
        # Generate a detailed report of failed tests
        if self.results["failed"] > 0:
            log_header("Failed Tests")
            failed_tests = [test for test in self.results["tests"] if not test["passed"]]
            for i, test in enumerate(failed_tests, 1):
                log_failure(f"{i}. {test['method']} {test['endpoint']} - {test['description']}")
                if "error" in test:
                    log_failure(f"   Error: {test['error']}")
                else:
                    log_failure(f"   Status: {test['status']} (Expected: {test['expected_status']})")
        
        # Group results by endpoint group
        group_results = {}
        for test in self.results["tests"]:
            group = test.get("group", "misc")
            if group not in group_results:
                group_results[group] = {"total": 0, "passed": 0, "failed": 0}
            
            group_results[group]["total"] += 1
            if test.get("passed", False):
                group_results[group]["passed"] += 1
            else:
                group_results[group]["failed"] += 1
        
        # Print group results
        log_header("Results by Group")
        for group, stats in group_results.items():
            group_pass_rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            
            if group_pass_rate >= 90:
                log_success(f"{group}: {stats['passed']}/{stats['total']} passed ({group_pass_rate:.1f}%)")
            elif group_pass_rate >= 75:
                log_warning(f"{group}: {stats['passed']}/{stats['total']} passed ({group_pass_rate:.1f}%)")
            else:
                log_failure(f"{group}: {stats['passed']}/{stats['total']} passed ({group_pass_rate:.1f}%)")
        
        # Add group results to main results
        self.results["group_results"] = group_results
        
        # Save the full results to a JSON file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"api_test_results_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        
        log_info(f"Full test results saved to: {filename}")
        
        # Return code based on pass rate
        return 0 if pass_rate == 100 else 1

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test API endpoints")
    parser.add_argument("--url", type=str, default="http://localhost:8000", 
                        help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--env", type=str, default="development", 
                        help="Environment setting (default: development)")
    parser.add_argument("--group", type=str, default=None,
                        help="Only test endpoints in this group (default: all)")
    args = parser.parse_args()
    
    # Create and run the tester
    tester = EndpointTester(args.url, args.env)
    return await tester.run_tests(args.group)

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log_warning("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_failure(f"Unhandled exception: {str(e)}")
        sys.exit(1)