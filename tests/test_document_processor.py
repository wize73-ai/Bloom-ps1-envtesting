"""
Test Script for Document Processing Endpoints.

This script tests the document processing capabilities of the CasaLingua API,
including document text extraction, processing, and analysis.
"""

import aiohttp
import asyncio
import json
import pytest
import os
import sys
import base64
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
        f.write("This is a test document for CasaLingua document processing API.")
    return test_file

# Create a simple PDF file for testing using fpdf if available
def create_test_pdf_file():
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt="This is a test PDF document for CasaLingua.")
        pdf.cell(0, 10, txt="It contains multiple lines of text.", ln=True)
        pdf.cell(0, 10, txt="Testing document processing capabilities.", ln=True)
        
        test_file = TEST_DATA_DIR / "test_document.pdf"
        pdf.output(str(test_file))
        return test_file
    except ImportError:
        return None


@pytest.mark.asyncio
async def test_document_extraction():
    """Test the document text extraction endpoint."""
    # Create a test text file
    test_file = create_test_text_file()
    
    async with aiohttp.ClientSession() as session:
        # Test document extraction endpoint
        with open(test_file, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f.read(), 
                          filename='test_document.txt',
                          content_type='text/plain')
            
            async with session.post(
                f"{API_URL}/document/extract",
                data=data,
                headers=AUTH_HEADER
            ) as response:
                assert response.status == 200
                response_data = await response.json()
                
                # Check response structure
                assert response_data["status"] == "success"
                assert "data" in response_data
                assert "original_text" in response_data["data"]
                
                # Verify the extracted text
                assert "test document" in response_data["data"]["original_text"]


@pytest.mark.asyncio
async def test_document_processing():
    """Test the document processing endpoint."""
    # Create a test text file
    test_file = create_test_text_file()
    
    async with aiohttp.ClientSession() as session:
        # Test document processing endpoint
        with open(test_file, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f.read(), 
                          filename='test_document.txt',
                          content_type='text/plain')
            data.add_field('translate', 'true')
            data.add_field('source_language', 'en')
            data.add_field('target_language', 'es')
            
            async with session.post(
                f"{API_URL}/document/process",
                data=data,
                headers=AUTH_HEADER
            ) as response:
                assert response.status == 200
                response_data = await response.json()
                
                # Check response structure
                assert response_data["status"] == "success"
                assert "data" in response_data
                assert "original_text" in response_data["data"]
                assert "processed_text" in response_data["data"]
                
                # Verify processing occurred
                assert response_data["data"]["original_text"] != response_data["data"]["processed_text"]


@pytest.mark.asyncio
async def test_document_analysis():
    """Test the document analysis endpoint."""
    # Create a test text file
    test_file = create_test_text_file()
    
    async with aiohttp.ClientSession() as session:
        # Test document analysis endpoint
        with open(test_file, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f.read(), 
                          filename='test_document.txt',
                          content_type='text/plain')
            data.add_field('detect_language', 'true')
            data.add_field('extract_entities', 'true')
            
            async with session.post(
                f"{API_URL}/document/analyze",
                data=data,
                headers=AUTH_HEADER
            ) as response:
                assert response.status == 200
                response_data = await response.json()
                
                # Check response structure
                assert response_data["status"] == "success"
                assert "data" in response_data
                assert "document_type" in response_data["data"]
                assert "word_count" in response_data["data"]
                assert "languages" in response_data["data"]
                
                # Verify analysis results
                assert response_data["data"]["word_count"] > 0
                assert len(response_data["data"]["languages"]) > 0
                assert response_data["data"]["languages"][0]["language"] == "en"


@pytest.mark.asyncio
async def test_pdf_extraction():
    """Test extraction from a PDF file if possible."""
    # Create a test PDF file
    test_file = create_test_pdf_file()
    if not test_file:
        pytest.skip("PDF creation library not available")
    
    async with aiohttp.ClientSession() as session:
        # Test PDF extraction
        with open(test_file, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f.read(), 
                          filename='test_document.pdf',
                          content_type='application/pdf')
            
            async with session.post(
                f"{API_URL}/document/extract",
                data=data,
                headers=AUTH_HEADER
            ) as response:
                assert response.status == 200
                response_data = await response.json()
                
                # Check response structure
                assert response_data["status"] == "success"
                assert "data" in response_data
                assert "original_text" in response_data["data"]
                
                # Verify the extracted text
                assert "test PDF document" in response_data["data"]["original_text"]


if __name__ == "__main__":
    # Create test files
    create_test_text_file()
    create_test_pdf_file()
    
    # Run the tests
    asyncio.run(test_document_extraction())
    asyncio.run(test_document_processing())
    asyncio.run(test_document_analysis())
    asyncio.run(test_pdf_extraction())
    
    print("All document processing tests completed successfully!")