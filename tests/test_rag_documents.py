"""
Test Script for RAG Document Processing Endpoints.

This script tests the RAG document indexing capabilities of the CasaLingua API,
including document indexing and session-based document management.
"""

import aiohttp
import asyncio
import json
import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create a test configuration with authentication token
API_URL = "http://localhost:8000"
AUTH_HEADER = {"Authorization": "Bearer test_token"}  # Update with a valid token if needed

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)

# Create a simple text file for testing
def create_test_text_file():
    test_file = TEST_DATA_DIR / "test_document.txt"
    with open(test_file, 'w') as f:
        f.write("This is a test document for CasaLingua RAG indexing API. " +
                "It contains information about language processing and translation. " +
                "The RAG system should be able to retrieve this context when asked about CasaLingua capabilities.")
    return test_file


@pytest.mark.asyncio
async def test_rag_document_indexing():
    """Test the RAG document indexing endpoint."""
    # Create a test text file
    test_file = create_test_text_file()
    
    async with aiohttp.ClientSession() as session:
        # Test RAG document indexing endpoint
        with open(test_file, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f.read(), 
                          filename='test_document.txt',
                          content_type='text/plain')
            data.add_field('store_in_session', 'true')
            
            async with session.post(
                f"{API_URL}/rag/index/document",
                data=data,
                headers=AUTH_HEADER
            ) as response:
                assert response.status == 200
                response_data = await response.json()
                
                # Check response structure
                assert response_data["status"] == "success"
                assert "data" in response_data
                assert "document_id" in response_data["data"]
                assert response_data["data"]["status"] == "indexing"
                
                # Store session ID for next test
                session_id = response_data["data"].get("session_id")
                
                # Verify session cookie was set
                assert "session_id" in response.cookies
                
                return session_id


@pytest.mark.asyncio
async def test_rag_session_indexing(session_id=None):
    """Test the RAG session document indexing endpoint."""
    if not session_id:
        # Create a session with a document first
        session_id = await test_rag_document_indexing()
        
        # Short delay to ensure document is processed
        await asyncio.sleep(2)
    
    async with aiohttp.ClientSession() as session:
        # Add session cookie
        cookies = {"session_id": session_id}
        
        # Test session indexing endpoint
        async with session.post(
            f"{API_URL}/rag/index/session",
            headers=AUTH_HEADER,
            cookies=cookies
        ) as response:
            assert response.status == 200
            response_data = await response.json()
            
            # Check response structure
            assert response_data["status"] == "success"
            assert "data" in response_data
            assert "document_count" in response_data["data"]
            assert response_data["data"]["document_count"] >= 1
            assert response_data["data"]["status"] == "indexing"


@pytest.mark.asyncio
async def test_rag_with_indexed_document():
    """Test RAG query with an indexed document."""
    # First index a document
    session_id = await test_rag_document_indexing()
    
    # Short delay to ensure document is indexed
    await asyncio.sleep(2)
    
    async with aiohttp.ClientSession() as session:
        # Add session cookie
        cookies = {"session_id": session_id}
        
        # Test RAG query with the indexed document
        query_data = {
            "query": "What capabilities does CasaLingua have?",
            "max_results": 3
        }
        
        async with session.post(
            f"{API_URL}/rag/query",
            json=query_data,
            headers=AUTH_HEADER,
            cookies=cookies
        ) as response:
            assert response.status == 200
            response_data = await response.json()
            
            # Check response structure
            assert response_data["status"] == "success"
            assert "data" in response_data
            assert "results" in response_data["data"]
            
            # Verify document content was retrieved
            results = response_data["data"]["results"]
            assert len(results) > 0
            
            # Check if any result contains content from our test document
            has_match = any("language processing" in result.get("text", "").lower() for result in results)
            assert has_match, "Document content not found in RAG results"


if __name__ == "__main__":
    # Create test files
    create_test_text_file()
    
    # Run the tests
    asyncio.run(test_rag_document_indexing())
    asyncio.run(test_rag_session_indexing())
    asyncio.run(test_rag_with_indexed_document())
    
    print("All RAG document processing tests completed successfully!")