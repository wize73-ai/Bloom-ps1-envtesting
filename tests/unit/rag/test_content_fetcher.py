"""
Test suite for the RAG Content Fetcher component.

Tests basic functionality of the content fetcher for RAG source URL management.
"""

import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock
import json
from pathlib import Path

from app.core.rag.content_fetcher import ContentFetcher, ContentProcessor

@pytest.fixture
def mock_response():
    """Create a mock HTTP response with HTML content."""
    mock = MagicMock()
    mock.status = 200
    mock.text.return_value = asyncio.Future()
    mock.text.return_value.set_result("<html><head><title>Test Page</title></head><body><p>Test content</p></body></html>")
    mock.read.return_value = asyncio.Future()
    mock.read.return_value.set_result(b"<html><head><title>Test Page</title></head><body><p>Test content</p></body></html>")
    return mock

@pytest.fixture
def content_fetcher():
    """Create a ContentFetcher instance with test configuration."""
    config = {
        "cache_dir": "./cache/test_content_fetcher",
        "cache_ttl": 60,
        "use_cache": False,
        "rate_limits": {
            "default": {"calls": 10, "period": 1},
            "github": {"calls": 10, "period": 1}
        }
    }
    return ContentFetcher(config)

@pytest.fixture
def content_processor(content_fetcher):
    """Create a ContentProcessor instance with test configuration."""
    config = {
        "cache_dir": "./cache/test_content_processor",
        "cache_ttl": 60,
        "use_cache": False,
    }
    processor = ContentProcessor(config)
    return processor

@pytest.mark.asyncio
@patch("aiohttp.ClientSession.get")
async def test_fetch_url_html(mock_get, content_fetcher, mock_response):
    """Test fetching HTML content from a URL."""
    mock_get.return_value.__aenter__.return_value = mock_response
    
    result = await content_fetcher.fetch_url(
        url="https://example.com",
        content_format="html"
    )
    
    assert result["success"] is True
    assert result["url"] == "https://example.com"
    assert "content" in result
    assert "metadata" in result
    assert result["metadata"].get("content_type") == "html"

@pytest.mark.asyncio
@patch("app.core.rag.content_fetcher.requests.get")
async def test_crawl_github_repo(mock_get, content_fetcher):
    """Test fetching content from a GitHub repository."""
    # This test mocks the GitHub API response
    
    # Mock the HTTP response for GitHub API
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Test Markdown Content"
    mock_get.return_value = mock_response
    
    # Replace the fetch_url method to avoid complexity
    with patch.object(content_fetcher, "fetch_url") as mock_fetch:
        mock_fetch.return_value = {
            "url": "https://raw.githubusercontent.com/example/repo/main/README.md",
            "content": "# Test Repository\nThis is a test.",
            "content_format": "markdown",
            "metadata": {"title": "Test Repository"},
            "timestamp": "2023-01-01T00:00:00Z",
            "status": "success"
        }
        
        result = await content_fetcher.fetch_github_repo(
            repo_url="https://github.com/example/repo",
            branch="main",
            max_files=1
        )
        
        assert len(result) > 0
        assert "content" in result[0]
        assert result[0]["content_format"] == "markdown"

@pytest.mark.asyncio
async def test_content_processor_process_text(content_processor):
    """Test processing text content."""
    text = """
    # Sample Document
    
    This is a test document with multiple paragraphs.
    
    Here is the second paragraph with some information.
    
    And here is a third paragraph to demonstrate chunking.
    """
    
    result = await content_processor.process_text(
        text=text,
        options={"chunk_size": 50, "chunk_overlap": 10}
    )
    
    assert result["success"] is True
    assert "content" in result
    assert "chunks" in result
    assert len(result["chunks"]) > 0
    assert "metadata" in result
    assert "language" in result["metadata"]

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])