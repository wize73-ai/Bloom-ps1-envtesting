#!/usr/bin/env python
"""
Monitoring script for veracity checking in CasaLingua.

This script sends test requests to the API and monitors the veracity checking
in action, tracking success rates, scores, and issues found.
"""

import requests
import json
import time
import argparse
import sys
import os
from typing import Dict, Any, List
from collections import defaultdict
import random

# Test data with various known issues to check veracity detection
TEST_CASES = [
    # Good translations
    {
        "text": "Hello, how are you today?",
        "source_language": "en",
        "target_language": "es",
        "expected_result": "Hola, ¿cómo estás hoy?",
        "expected_issues": []
    },
    {
        "text": "I have two dogs and three cats at home.",
        "source_language": "en",
        "target_language": "es",
        "expected_result": "Tengo dos perros y tres gatos en casa.",
        "expected_issues": []
    },
    
    # Missing numbers
    {
        "text": "I have 5 apples and 3 oranges in my basket.",
        "source_language": "en",
        "target_language": "es",
        "expected_result": "Tengo manzanas y naranjas en mi canasta.",
        "expected_issues": ["missing_numbers"]
    },
    {
        "text": "The product costs $159.99 plus tax.",
        "source_language": "en",
        "target_language": "es",
        "expected_result": "El producto cuesta $ más impuestos.",
        "expected_issues": ["missing_numbers"]
    },
    
    # Untranslated content
    {
        "text": "This text should be translated.",
        "source_language": "en",
        "target_language": "es",
        "expected_result": "This text should be translated.",
        "expected_issues": ["untranslated"]
    },
    
    # Entity preservation issues
    {
        "text": "John Smith visited New York last summer.",
        "source_language": "en", 
        "target_language": "es",
        "expected_result": "Alguien visitó alguna ciudad el verano pasado.",
        "expected_issues": ["missing_entities"]
    }
]

def send_translation_request(url: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a translation request to the API.
    
    Args:
        url: The API endpoint URL
        data: The request data
        
    Returns:
        The API response as a dictionary
    """
    try:
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending request: {e}")
        return {"error": str(e)}

def analyze_response(response: Dict[str, Any], expected_issues: List[str]) -> Dict[str, Any]:
    """
    Analyze the API response for veracity checking results.
    
    Args:
        response: The API response
        expected_issues: List of expected issue types
        
    Returns:
        Analysis results
    """
    results = {
        "success": False,
        "has_veracity_data": False,
        "veracity_score": None,
        "issues": [],
        "expected_issues_found": False,
        "translation_quality": None
    }
    
    if "error" in response:
        return results
    
    # Check if the response was successful
    if response.get("status") == "success" and "data" in response:
        results["success"] = True
        data = response["data"]
        
        # Get veracity data
        verification_score = data.get("verification_score")
        verified = data.get("verified", False)
        
        results["has_veracity_data"] = verification_score is not None
        results["veracity_score"] = verification_score
        
        # Check for veracity metadata
        if "metadata" in data and "veracity" in data["metadata"]:
            veracity = data["metadata"]["veracity"]
            results["issues"] = veracity.get("checks_failed", []) + veracity.get("warnings", [])
            
            # Check if expected issues were found
            expected_found = all(issue in results["issues"] for issue in expected_issues)
            results["expected_issues_found"] = expected_found
            
            # Rate translation quality
            results["translation_quality"] = "good" if len(results["issues"]) == 0 else "poor"
    
    return results

def run_monitoring(base_url: str, iterations: int = 10, delay: float = 1.0):
    """
    Run the monitoring for a specified number of iterations.
    
    Args:
        base_url: The base API URL
        iterations: Number of monitoring iterations
        delay: Delay between requests in seconds
    """
    endpoint = f"{base_url.rstrip('/')}/pipeline/translate"
    
    # Statistics
    stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "requests_with_veracity": 0,
        "veracity_scores": [],
        "issue_counts": defaultdict(int),
        "expected_issues_found": 0,
        "translation_quality": {"good": 0, "poor": 0}
    }
    
    print(f"Starting veracity monitoring for {iterations} iterations...")
    print(f"API endpoint: {endpoint}")
    print("-" * 50)
    
    for i in range(iterations):
        # Select a random test case
        test_case = random.choice(TEST_CASES)
        
        print(f"\nIteration {i+1}/{iterations}")
        print(f"Source text: {test_case['text']}")
        print(f"Languages: {test_case['source_language']} -> {test_case['target_language']}")
        print(f"Expected issues: {', '.join(test_case['expected_issues']) if test_case['expected_issues'] else 'None'}")
        
        # Send the request
        stats["total_requests"] += 1
        response = send_translation_request(endpoint, {
            "text": test_case["text"],
            "source_language": test_case["source_language"],
            "target_language": test_case["target_language"],
            "verify": True  # Explicitly request verification
        })
        
        # Analyze the response
        results = analyze_response(response, test_case["expected_issues"])
        
        # Update statistics
        if results["success"]:
            stats["successful_requests"] += 1
            
            if results["has_veracity_data"]:
                stats["requests_with_veracity"] += 1
                
                if results["veracity_score"] is not None:
                    stats["veracity_scores"].append(results["veracity_score"])
                
                for issue in results["issues"]:
                    stats["issue_counts"][issue] += 1
                
                if results["expected_issues_found"]:
                    stats["expected_issues_found"] += 1
                
                if results["translation_quality"]:
                    stats["translation_quality"][results["translation_quality"]] += 1
        
        # Print results
        print("\nResults:")
        print(f"  Success: {results['success']}")
        print(f"  Has veracity data: {results['has_veracity_data']}")
        if results["has_veracity_data"]:
            print(f"  Veracity score: {results['veracity_score']}")
            print(f"  Issues: {', '.join(results['issues']) if results['issues'] else 'None'}")
            print(f"  Expected issues found: {results['expected_issues_found']}")
            print(f"  Translation quality: {results['translation_quality']}")
        
        # Add delay between requests
        if i < iterations - 1:
            time.sleep(delay)
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("Monitoring Summary")
    print("=" * 50)
    print(f"Total requests: {stats['total_requests']}")
    print(f"Successful requests: {stats['successful_requests']} ({stats['successful_requests'] / stats['total_requests'] * 100:.1f}%)")
    print(f"Requests with veracity data: {stats['requests_with_veracity']} ({stats['requests_with_veracity'] / stats['total_requests'] * 100:.1f}%)")
    
    if stats["veracity_scores"]:
        avg_score = sum(stats["veracity_scores"]) / len(stats["veracity_scores"])
        print(f"Average veracity score: {avg_score:.2f}")
    
    if stats["issue_counts"]:
        print("\nIssues Found:")
        for issue, count in sorted(stats["issue_counts"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {issue}: {count}")
    
    if stats["expected_issues_found"] > 0:
        accuracy = stats["expected_issues_found"] / stats["total_requests"] * 100
        print(f"\nExpected issues found: {stats['expected_issues_found']} ({accuracy:.1f}%)")
    
    if stats["translation_quality"]["good"] + stats["translation_quality"]["poor"] > 0:
        good_pct = stats["translation_quality"]["good"] / (stats["translation_quality"]["good"] + stats["translation_quality"]["poor"]) * 100
        print(f"\nTranslation quality: {stats['translation_quality']['good']} good ({good_pct:.1f}%), {stats['translation_quality']['poor']} poor")

def main():
    """Main function to parse arguments and run the monitoring."""
    parser = argparse.ArgumentParser(description="Monitor veracity checking in CasaLingua")
    parser.add_argument("--url", default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--iterations", type=int, default=10, help="Number of monitoring iterations")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    
    args = parser.parse_args()
    run_monitoring(args.url, args.iterations, args.delay)

if __name__ == "__main__":
    main()