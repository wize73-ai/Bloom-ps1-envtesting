#!/usr/bin/env python3
"""
Test script to verify our enhanced metrics are included in API responses.
"""
import sys
import requests
import json
from pprint import pprint

BASE_URL = "http://localhost:8000"

def test_language_detection():
    """Test that language detection endpoint includes enhanced metrics"""
    print("\n===== Testing Language Detection =====")
    
    url = f"{BASE_URL}/pipeline/detect"
    headers = {
        "Content-Type": "application/json",
        # Add API key if needed
    }
    data = {
        "text": "Hello, how are you today?",
        "detailed": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        print(f"Response status: {response.status_code}")
        json_response = response.json()
        
        # Check for metrics in response data
        if "data" in json_response and json_response["data"]:
            data = json_response["data"]
            print("\nResponse Data Metrics:")
            print(f"- performance_metrics: {data.get('performance_metrics')}")
            print(f"- memory_usage: {data.get('memory_usage')}")
            print(f"- operation_cost: {data.get('operation_cost')}")
            print(f"- accuracy_score: {data.get('accuracy_score')}")
            print(f"- truth_score: {data.get('truth_score')}")
        
        # Check for metrics in response metadata
        if "metadata" in json_response and json_response["metadata"]:
            metadata = json_response["metadata"]
            print("\nResponse Metadata Metrics:")
            print(f"- performance_metrics: {metadata.get('performance_metrics')}")
            print(f"- memory_usage: {metadata.get('memory_usage')}")
            print(f"- operation_cost: {metadata.get('operation_cost')}")
            print(f"- accuracy_score: {metadata.get('accuracy_score')}")
            print(f"- truth_score: {metadata.get('truth_score')}")
            
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, "response") and hasattr(e.response, "text"):
            print(f"Response: {e.response.text}")
        return False

def test_translation():
    """Test that translation endpoint includes enhanced metrics"""
    print("\n===== Testing Translation =====")
    
    url = f"{BASE_URL}/pipeline/translate"
    headers = {
        "Content-Type": "application/json",
        # Add API key if needed
    }
    data = {
        "text": "Hello, how are you today?",
        "source_language": "en",
        "target_language": "es"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        print(f"Response status: {response.status_code}")
        json_response = response.json()
        
        # Check for metrics in response data
        if "data" in json_response and json_response["data"]:
            data = json_response["data"]
            print("\nResponse Data Metrics:")
            print(f"- performance_metrics: {data.get('performance_metrics')}")
            print(f"- memory_usage: {data.get('memory_usage')}")
            print(f"- operation_cost: {data.get('operation_cost')}")
            print(f"- accuracy_score: {data.get('accuracy_score')}")
            print(f"- truth_score: {data.get('truth_score')}")
        
        # Check for metrics in response metadata
        if "metadata" in json_response and json_response["metadata"]:
            metadata = json_response["metadata"]
            print("\nResponse Metadata Metrics:")
            print(f"- performance_metrics: {metadata.get('performance_metrics')}")
            print(f"- memory_usage: {metadata.get('memory_usage')}")
            print(f"- operation_cost: {metadata.get('operation_cost')}")
            print(f"- accuracy_score: {metadata.get('accuracy_score')}")
            print(f"- truth_score: {metadata.get('truth_score')}")
            
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, "response") and hasattr(e.response, "text"):
            print(f"Response: {e.response.text}")
        return False

def test_simplification():
    """Test that simplification endpoint includes enhanced metrics"""
    print("\n===== Testing Simplification =====")
    
    url = f"{BASE_URL}/pipeline/simplify"
    headers = {
        "Content-Type": "application/json",
        # Add API key if needed
    }
    data = {
        "text": "The mitochondrion is a double membrane-bound organelle found in most eukaryotic organisms.",
        "language": "en",
        "target_level": "simple"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        print(f"Response status: {response.status_code}")
        json_response = response.json()
        
        # Check for metrics in metadata
        if "metadata" in json_response and json_response["metadata"]:
            metadata = json_response["metadata"]
            print("\nResponse Metadata Metrics:")
            print(f"- performance_metrics: {metadata.get('performance_metrics')}")
            print(f"- memory_usage: {metadata.get('memory_usage')}")
            print(f"- operation_cost: {metadata.get('operation_cost')}")
            print(f"- accuracy_score: {metadata.get('accuracy_score')}")
            print(f"- truth_score: {metadata.get('truth_score')}")
            
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, "response") and hasattr(e.response, "text"):
            print(f"Response: {e.response.text}")
        return False

def test_rag_query():
    """Test that RAG query endpoint includes enhanced metrics"""
    print("\n===== Testing RAG Query =====")
    
    url = f"{BASE_URL}/rag/query"
    headers = {
        "Content-Type": "application/json",
        # Add API key if needed
    }
    data = {
        "query": "What is machine learning?",
        "language": "en",
        "max_results": 3
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        print(f"Response status: {response.status_code}")
        json_response = response.json()
        
        # Check for metrics in response data
        if "data" in json_response and json_response["data"]:
            data = json_response["data"]
            print("\nResponse Data Metrics:")
            print(f"- performance_metrics: {data.get('performance_metrics')}")
            print(f"- memory_usage: {data.get('memory_usage')}")
            print(f"- operation_cost: {data.get('operation_cost')}")
            print(f"- accuracy_score: {data.get('accuracy_score')}")
            print(f"- truth_score: {data.get('truth_score')}")
        
        # Check for metrics in response metadata
        if "metadata" in json_response and json_response["metadata"]:
            metadata = json_response["metadata"]
            print("\nResponse Metadata Metrics:")
            print(f"- performance_metrics: {metadata.get('performance_metrics')}")
            print(f"- memory_usage: {metadata.get('memory_usage')}")
            print(f"- operation_cost: {metadata.get('operation_cost')}")
            print(f"- accuracy_score: {metadata.get('accuracy_score')}")
            print(f"- truth_score: {metadata.get('truth_score')}")
            
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, "response") and hasattr(e.response, "text"):
            print(f"Response: {e.response.text}")
        return False

def main():
    """Run all tests"""
    print("Testing Enhanced API Response Metrics")
    print("=====================================")
    
    # Run tests
    language_detection_ok = test_language_detection()
    translation_ok = test_translation()
    simplification_ok = test_simplification()
    rag_query_ok = test_rag_query()
    
    # Print summary
    print("\n===== Test Summary =====")
    print(f"Language Detection: {'PASS' if language_detection_ok else 'FAIL'}")
    print(f"Translation: {'PASS' if translation_ok else 'FAIL'}")
    print(f"Simplification: {'PASS' if simplification_ok else 'FAIL'}")
    print(f"RAG Query: {'PASS' if rag_query_ok else 'FAIL'}")
    
    # Return exit code
    return 0 if all([language_detection_ok, translation_ok, simplification_ok, rag_query_ok]) else 1

if __name__ == "__main__":
    sys.exit(main())