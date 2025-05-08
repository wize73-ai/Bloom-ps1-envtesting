"""
Content Fetcher for RAG Sources

This module handles fetching and processing content from various web sources
for the Retrieval-Augmented Generation system.
"""

# Import hashlib at the top level
import hashlib

import time
import asyncio
import logging
import uuid
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from urllib.parse import urlparse, urljoin
import json

import aiohttp
from bs4 import BeautifulSoup
import markdown
from pydantic import HttpUrl

from app.api.schemas.rag_sources import (
    RAGSource, SourceType, ContentFormat,
    SourceCredentials, ProcessingOptions,
    SourceStatus, AccessMethod
)
from app.utils.logging import get_logger
from app.core.rag.indexer import Indexer, TextProcessor

logger = get_logger(__name__)

class ContentProcessor:
    """
    Process and transform content for RAG indexing.
    
    This class handles text processing for various content formats 
    to prepare them for RAG indexing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the content processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
    def process_content(
        self, 
        content: str, 
        content_format: ContentFormat,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process content based on its format.
        
        Args:
            content: The content to process
            content_format: The format of the content
            metadata: Optional metadata to include
            
        Returns:
            Processed content with metadata
        """
        metadata = metadata or {}
        
        if content_format == ContentFormat.HTML:
            return self._process_html(content, metadata)
        elif content_format == ContentFormat.MARKDOWN:
            return self._process_markdown(content, metadata)
        elif content_format == ContentFormat.JSON:
            return self._process_json(content, metadata)
        elif content_format == ContentFormat.TEXT:
            return self._process_text(content, metadata)
        else:
            # Default to text processing
            return self._process_text(content, metadata)
    
    def _process_html(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process HTML content."""
        try:
            soup = BeautifulSoup(content, "html.parser")
            
            # Remove unwanted elements
            for element in soup.select("script, style, meta, link"):
                element.decompose()
                
            # Extract title if available
            title = soup.title.text if soup.title else ""
            
            # Extract text
            text = soup.get_text(separator="\n", strip=True)
            
            return {
                "content": text,
                "metadata": {
                    **metadata,
                    "format": "html",
                    "title": title
                }
            }
        except Exception as e:
            logger.error(f"Error processing HTML: {str(e)}")
            return {
                "content": content,
                "metadata": {**metadata, "format": "html", "error": str(e)}
            }
    
    def _process_markdown(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process Markdown content."""
        try:
            # Extract title from first heading if available
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else ""
            
            return {
                "content": content,
                "metadata": {
                    **metadata,
                    "format": "markdown",
                    "title": title
                }
            }
        except Exception as e:
            logger.error(f"Error processing Markdown: {str(e)}")
            return {
                "content": content,
                "metadata": {**metadata, "format": "markdown", "error": str(e)}
            }
    
    def _process_json(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process JSON content."""
        try:
            data = json.loads(content)
            
            # Extract text from JSON
            extracted_text = self._extract_text_from_json(data)
            
            return {
                "content": "\n\n".join(extracted_text),
                "metadata": {
                    **metadata,
                    "format": "json"
                }
            }
        except Exception as e:
            logger.error(f"Error processing JSON: {str(e)}")
            return {
                "content": content,
                "metadata": {**metadata, "format": "json", "error": str(e)}
            }
    
    def _process_text(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process plain text content."""
        return {
            "content": content,
            "metadata": {
                **metadata,
                "format": "text"
            }
        }
    
    def _extract_text_from_json(self, data: Any) -> List[str]:
        """Extract text fields from JSON data."""
        result = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 10:
                    result.append(f"{key}: {value}")
                elif isinstance(value, (dict, list)):
                    result.extend(self._extract_text_from_json(value))
        elif isinstance(data, list):
            for item in data:
                result.extend(self._extract_text_from_json(item))
                
        return result

class ContentFetcher:
    """
    Fetches and processes content from web sources for RAG.
    
    This class handles:
    - Fetching content from URLs
    - Processing HTML, Markdown, and other formats
    - Extracting text for indexing
    - Following links for recursive crawling
    - Handling rate limits and authentication
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        indexer: Optional[Indexer] = None
    ):
        """
        Initialize the content fetcher.
        
        Args:
            config: Configuration dictionary
            indexer: Optional Indexer instance for processing content
        """
        self.config = config or {}
        self.indexer = indexer
        
        # Initialize cache directory
        self.cache_dir = Path(self.config.get("content_cache_dir", "cache/sources"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Timeouts and rate limits
        self.request_timeout = self.config.get("request_timeout", 30)
        self.rate_limit_delay = self.config.get("rate_limit_delay", 1.0)
        self.max_content_size = self.config.get("max_content_size", 10 * 1024 * 1024)  # 10 MB
        self.cache_ttl = self.config.get("content_cache_ttl", 86400)  # 24 hours
        
        # Session for HTTP requests
        self.session = None
        
        # Initialize chunker for text processing
        self.text_processor = TextProcessor(
            chunk_size=self.config.get("chunk_size", 500),
            chunk_overlap=self.config.get("chunk_overlap", 50)
        )
        
        # Track visited URLs to avoid duplicates
        self.visited_urls: Set[str] = set()
        self.pending_urls: Set[str] = set()
        self.url_contents: Dict[str, str] = {}
        
        logger.info("ContentFetcher initialized")
    
    async def initialize(self) -> None:
        """Initialize the content fetcher."""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.request_timeout))
            logger.info("ContentFetcher HTTP session created")
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("ContentFetcher HTTP session closed")
    
    async def process_source(
        self,
        source: RAGSource,
        options: Optional[ProcessingOptions] = None
    ) -> Dict[str, Any]:
        """
        Process a RAG source by fetching and indexing its content.
        
        Args:
            source: The RAG source to process
            options: Optional processing options to override source options
            
        Returns:
            Dictionary with processing statistics and results
        """
        await self.initialize()
        
        # Start timing
        start_time = datetime.now()
        
        # Create processing statistics
        stats = {
            "url": str(source.url),
            "start_time": start_time,
            "success": False,
            "chunks_created": 0,
            "links_found": 0,
            "links_followed": 0,
            "content_size": 0
        }
        
        # Reset tracking sets
        self.visited_urls = set()
        self.pending_urls = set()
        self.url_contents = {}
        
        try:
            # Use provided options or source options
            processing_options = options or source.options
            
            # Process the URL based on source type
            if source.source_type == SourceType.WEBPAGE:
                await self._process_webpage(source, processing_options, stats)
            elif source.source_type == SourceType.GITHUB_REPO:
                await self._process_github_repo(source, processing_options, stats)
            elif source.source_type == SourceType.API:
                await self._process_api(source, processing_options, stats)
            elif source.source_type == SourceType.RSS:
                await self._process_rss(source, processing_options, stats)
            elif source.source_type == SourceType.SITEMAP:
                await self._process_sitemap(source, processing_options, stats)
            else:
                logger.warning(f"Unsupported source type: {source.source_type}")
                stats["error"] = f"Unsupported source type: {source.source_type}"
                stats["success"] = False
                return stats
            
            # Mark as success if we got here
            stats["success"] = True
            
            # Finish timing and update stats
            end_time = datetime.now()
            stats["end_time"] = end_time
            stats["duration_seconds"] = (end_time - start_time).total_seconds()
            
            return stats
            
        except Exception as e:
            # Log and capture error
            logger.error(f"Error processing source {source.url}: {str(e)}", exc_info=True)
            
            # Update stats with error info
            stats["success"] = False
            stats["error"] = str(e)
            
            # Finish timing and update stats
            end_time = datetime.now()
            stats["end_time"] = end_time
            stats["duration_seconds"] = (end_time - start_time).total_seconds()
            
            return stats
    
    async def _process_webpage(
        self,
        source: RAGSource,
        options: ProcessingOptions,
        stats: Dict[str, Any]
    ) -> None:
        """
        Process a webpage source.
        
        Args:
            source: The source to process
            options: Processing options
            stats: Statistics dictionary to update
        """
        # Start with the main URL
        main_url = str(source.url)
        self.pending_urls.add(main_url)
        
        # Process the main URL and follow links if requested
        depth = 0
        while self.pending_urls and depth <= options.max_depth:
            # Process URLs at current depth
            current_depth_urls = list(self.pending_urls)
            self.pending_urls = set()
            
            # Process each URL at the current depth
            for url in current_depth_urls:
                if url in self.visited_urls:
                    continue
                    
                # Fetch and process the page
                content, links = await self._fetch_and_process_url(
                    url, source, options
                )
                
                if content:
                    self.visited_urls.add(url)
                    self.url_contents[url] = content
                    
                    # Update stats
                    stats["content_size"] += len(content)
                    
                    # Add new links to pending if we're not at max depth yet
                    if depth < options.max_depth and options.extract_links:
                        stats["links_found"] += len(links)
                        for link in links:
                            if link not in self.visited_urls and link not in self.pending_urls:
                                self.pending_urls.add(link)
                                stats["links_followed"] += 1
            
            # Increment depth for next iteration
            depth += 1
        
        # Once we have all content, index it
        await self._index_all_content(source, options, stats)
    
    async def _process_github_repo(
        self,
        source: RAGSource,
        options: ProcessingOptions,
        stats: Dict[str, Any]
    ) -> None:
        """
        Process a GitHub repository source.
        
        Args:
            source: The source to process
            options: Processing options
            stats: Statistics dictionary to update
        """
        # Extract repo owner and name from URL
        url_parts = str(source.url).strip("/").split("/")
        if len(url_parts) < 5 or url_parts[2] != "github.com":
            raise ValueError(f"Invalid GitHub repository URL: {source.url}")
        
        repo_owner = url_parts[3]
        repo_name = url_parts[4]
        
        # Construct API URL for repo contents
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"
        
        # Set up GitHub API headers
        headers = {"Accept": "application/vnd.github.v3+json"}
        
        # Add credentials if provided
        if source.credentials:
            if source.credentials.method == AccessMethod.API_KEY and source.credentials.api_key:
                headers["Authorization"] = f"token {source.credentials.api_key}"
            elif source.credentials.method == AccessMethod.OAUTH and source.credentials.token:
                headers["Authorization"] = f"token {source.credentials.token}"
            elif source.credentials.method == AccessMethod.BASIC_AUTH:
                # In aiohttp, basic auth is passed separately
                auth = aiohttp.BasicAuth(
                    login=source.credentials.username or "",
                    password=source.credentials.password or ""
                )
            else:
                auth = None
        else:
            auth = None
        
        # Recursive function to process contents
        async def process_repo_path(path: str = "") -> None:
            nonlocal stats
            
            # Construct path URL
            path_url = api_url
            if path:
                path_url = f"{api_url}/{path}"
            
            # Fetch and parse JSON
            try:
                async with self.session.get(path_url, headers=headers, auth=auth) as response:
                    if response.status == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
                        # Rate limited - sleep and retry
                        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                        sleep_time = max(1, reset_time - time.time())
                        logger.warning(f"GitHub API rate limited. Sleeping for {sleep_time} seconds.")
                        await asyncio.sleep(sleep_time)
                        return await process_repo_path(path)
                    
                    response.raise_for_status()
                    contents = await response.json()
                    
                    # Handle single file response
                    if not isinstance(contents, list):
                        contents = [contents]
                    
                    for item in contents:
                        item_type = item.get("type")
                        item_path = item.get("path")
                        
                        if item_type == "dir" and options.extract_links:
                            # Recurse into subdirectory
                            await process_repo_path(item_path)
                            stats["links_followed"] += 1
                        elif item_type == "file":
                            # Check file type before processing
                            file_ext = os.path.splitext(item_path)[1].lower()
                            if file_ext in [".md", ".txt", ".py", ".js", ".html", ".css", ".json", ".yml", ".yaml"]:
                                # Fetch and process file content
                                download_url = item.get("download_url")
                                if download_url:
                                    async with self.session.get(download_url, headers=headers) as file_response:
                                        file_response.raise_for_status()
                                        content = await file_response.text()
                                        
                                        # Store content with GitHub URL as key
                                        self.url_contents[f"{source.url}/blob/main/{item_path}"] = content
                                        stats["content_size"] += len(content)
                        
                        stats["links_found"] += 1
            
            except Exception as e:
                logger.error(f"Error processing GitHub repo path {path_url}: {str(e)}")
                stats["error"] = f"Error processing GitHub repo: {str(e)}"
        
        # Start processing from root
        await process_repo_path()
        
        # Index all content
        await self._index_all_content(source, options, stats)
    
    async def _process_api(
        self,
        source: RAGSource,
        options: ProcessingOptions,
        stats: Dict[str, Any]
    ) -> None:
        """
        Process an API source.
        
        Args:
            source: The source to process
            options: Processing options
            stats: Statistics dictionary to update
        """
        # Set up headers
        headers = {"Accept": "application/json"}
        
        # Add credentials if provided
        if source.credentials:
            if source.credentials.method == AccessMethod.API_KEY and source.credentials.api_key:
                headers["Authorization"] = f"Bearer {source.credentials.api_key}"
            elif source.credentials.method == AccessMethod.OAUTH and source.credentials.token:
                headers["Authorization"] = f"Bearer {source.credentials.token}"
            elif source.credentials.method == AccessMethod.BASIC_AUTH:
                # In aiohttp, basic auth is passed separately
                auth = aiohttp.BasicAuth(
                    login=source.credentials.username or "",
                    password=source.credentials.password or ""
                )
            else:
                auth = None
        else:
            auth = None
        
        # Fetch and process API response
        try:
            async with self.session.get(str(source.url), headers=headers, auth=auth) as response:
                response.raise_for_status()
                content = await response.text()
                
                # Parse JSON if possible
                try:
                    json_content = json.loads(content)
                    
                    # Extract text content from JSON (this is a simplified example)
                    if isinstance(json_content, dict):
                        extracted_texts = self._extract_text_from_json(json_content)
                        content = "\n\n".join(extracted_texts)
                    elif isinstance(json_content, list):
                        all_texts = []
                        for item in json_content:
                            if isinstance(item, dict):
                                texts = self._extract_text_from_json(item)
                                all_texts.extend(texts)
                        content = "\n\n".join(all_texts)
                except Exception as json_error:
                    logger.warning(f"Error parsing JSON from API: {json_error}")
                
                # Store the content
                self.url_contents[str(source.url)] = content
                stats["content_size"] = len(content)
        
        except Exception as e:
            logger.error(f"Error processing API source {source.url}: {str(e)}")
            stats["error"] = f"Error processing API source: {str(e)}"
            return
        
        # Index the content
        await self._index_all_content(source, options, stats)
    
    async def _process_rss(
        self,
        source: RAGSource,
        options: ProcessingOptions,
        stats: Dict[str, Any]
    ) -> None:
        """
        Process an RSS feed source.
        
        Args:
            source: The source to process
            options: Processing options
            stats: Statistics dictionary to update
        """
        try:
            # Fetch RSS feed
            async with self.session.get(str(source.url)) as response:
                response.raise_for_status()
                feed_content = await response.text()
                
                # Parse RSS with BeautifulSoup
                soup = BeautifulSoup(feed_content, "xml")
                
                # Extract items/entries
                items = soup.find_all(["item", "entry"])  # RSS or Atom
                
                for i, item in enumerate(items):
                    # Extract title, link, and content
                    title = item.find(["title"])
                    title_text = title.text if title else ""
                    
                    link = item.find(["link"])
                    link_url = link.text if link and link.text else link.get("href", "") if link else ""
                    
                    # Try different content tags (RSS feeds vary in structure)
                    content_element = item.find(["content:encoded", "content", "description", "summary"])
                    content_text = content_element.text if content_element else ""
                    
                    # Combine into a single document
                    item_content = f"# {title_text}\n\n{content_text}"
                    
                    # Store with link as key if available, otherwise use a generated key
                    key = link_url if link_url else f"{source.url}#item-{i}"
                    self.url_contents[key] = item_content
                    stats["content_size"] += len(item_content)
                    stats["links_found"] += 1
                
                # Follow links to full content if requested
                if options.extract_links:
                    for key in list(self.url_contents.keys()):
                        if key.startswith("http"):
                            await self._fetch_and_process_url(key, source, options)
                            stats["links_followed"] += 1
        
        except Exception as e:
            logger.error(f"Error processing RSS source {source.url}: {str(e)}")
            stats["error"] = f"Error processing RSS source: {str(e)}"
            return
        
        # Index all content
        await self._index_all_content(source, options, stats)
    
    async def _process_sitemap(
        self,
        source: RAGSource,
        options: ProcessingOptions,
        stats: Dict[str, Any]
    ) -> None:
        """
        Process a sitemap source.
        
        Args:
            source: The source to process
            options: Processing options
            stats: Statistics dictionary to update
        """
        try:
            # Fetch sitemap
            async with self.session.get(str(source.url)) as response:
                response.raise_for_status()
                sitemap_content = await response.text()
                
                # Parse sitemap with BeautifulSoup
                soup = BeautifulSoup(sitemap_content, "xml")
                
                # Extract URLs
                urls = []
                
                # Check if it's a sitemap index
                sitemap_tags = soup.find_all("sitemap")
                if sitemap_tags:
                    # This is a sitemap index, get all sub-sitemaps
                    for sitemap_tag in sitemap_tags:
                        loc = sitemap_tag.find("loc")
                        if loc:
                            urls.append(loc.text)
                    
                    # Process each sub-sitemap
                    for sitemap_url in urls:
                        try:
                            async with self.session.get(sitemap_url) as sub_response:
                                sub_response.raise_for_status()
                                sub_content = await sub_response.text()
                                sub_soup = BeautifulSoup(sub_content, "xml")
                                for url_tag in sub_soup.find_all("url"):
                                    loc = url_tag.find("loc")
                                    if loc:
                                        self.pending_urls.add(loc.text)
                                        stats["links_found"] += 1
                        except Exception as sub_e:
                            logger.warning(f"Error processing sub-sitemap {sitemap_url}: {str(sub_e)}")
                else:
                    # This is a regular sitemap
                    for url_tag in soup.find_all("url"):
                        loc = url_tag.find("loc")
                        if loc:
                            self.pending_urls.add(loc.text)
                            stats["links_found"] += 1
                
                # Process all discovered URLs
                max_urls = options.max_depth * 10  # Limit the number of URLs to process
                url_count = 0
                for url in list(self.pending_urls):
                    if url_count >= max_urls:
                        logger.info(f"Reached maximum URL limit of {max_urls}")
                        break
                        
                    if url in self.visited_urls:
                        continue
                        
                    await self._fetch_and_process_url(url, source, options)
                    stats["links_followed"] += 1
                    url_count += 1
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
        
        except Exception as e:
            logger.error(f"Error processing sitemap source {source.url}: {str(e)}")
            stats["error"] = f"Error processing sitemap source: {str(e)}"
            return
        
        # Index all content
        await self._index_all_content(source, options, stats)
    
    async def _fetch_and_process_url(
        self,
        url: str,
        source: RAGSource,
        options: ProcessingOptions
    ) -> Tuple[Optional[str], List[str]]:
        """
        Fetch and process a single URL.
        
        Args:
            url: The URL to fetch
            source: The original source
            options: Processing options
            
        Returns:
            Tuple of (processed content, list of discovered links)
        """
        links = []
        
        # Check cache first
        cached_content = self._check_cache(url)
        if cached_content is not None:
            # If we have cached content, extract links if needed
            if options.extract_links:
                links = self._extract_links(cached_content, url)
            return cached_content, links
        
        try:
            # Fetch the URL
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return None, []
                
                # Check content type
                content_type = response.headers.get("Content-Type", "")
                
                # Check content length
                content_length = int(response.headers.get("Content-Length", "0"))
                if content_length > self.max_content_size:
                    logger.warning(f"Content too large for {url}: {content_length} bytes")
                    return None, []
                
                # Get content
                content = await response.text()
                
                # Process content based on type
                if "text/html" in content_type:
                    processed_content, links = self._process_html(content, url, options)
                elif "text/markdown" in content_type or url.endswith(".md"):
                    processed_content = self._process_markdown(content, options)
                    links = self._extract_links(content, url) if options.extract_links else []
                elif "application/json" in content_type:
                    processed_content = self._process_json(content, options)
                    links = []
                elif "text/plain" in content_type:
                    processed_content = content
                    links = []
                else:
                    # Default to treating as HTML
                    processed_content, links = self._process_html(content, url, options)
                
                # Cache the content
                self._update_cache(url, processed_content)
                
                return processed_content, links
        
        except Exception as e:
            logger.warning(f"Error fetching and processing {url}: {str(e)}")
            return None, []
    
    def _process_html(
        self,
        html_content: str,
        base_url: str,
        options: ProcessingOptions
    ) -> Tuple[str, List[str]]:
        """
        Process HTML content.
        
        Args:
            html_content: The HTML content to process
            base_url: The base URL for resolving relative links
            options: Processing options
            
        Returns:
            Tuple of (processed text content, list of discovered links)
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Extract links if requested
            links = []
            if options.extract_links:
                links = self._extract_links(html_content, base_url)
            
            # Remove unwanted elements
            for element in soup.select("script, style, meta, link, nav, footer, header, aside"):
                element.decompose()
            
            # If extract_text_only is True, get just the text
            if options.extract_text_only:
                text_content = soup.get_text(separator="\n", strip=True)
                
                # Clean up the text (remove excessive whitespace)
                text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
                
                return text_content, links
            else:
                # Otherwise return the cleaned HTML
                return str(soup), links
        
        except Exception as e:
            logger.warning(f"Error processing HTML: {str(e)}")
            return html_content, []
    
    def _process_markdown(
        self,
        md_content: str,
        options: ProcessingOptions
    ) -> str:
        """
        Process Markdown content.
        
        Args:
            md_content: The Markdown content
            options: Processing options
            
        Returns:
            Processed content
        """
        if options.extract_text_only:
            # Convert markdown to HTML, then extract text
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            return text
        else:
            return md_content
    
    def _process_json(
        self,
        json_content: str,
        options: ProcessingOptions
    ) -> str:
        """
        Process JSON content.
        
        Args:
            json_content: The JSON content as string
            options: Processing options
            
        Returns:
            Processed content
        """
        try:
            data = json.loads(json_content)
            
            if options.extract_text_only:
                # Extract text values from JSON
                extracted_texts = self._extract_text_from_json(data)
                return "\n\n".join(extracted_texts)
            else:
                # Return formatted JSON
                return json.dumps(data, indent=2)
        
        except Exception as e:
            logger.warning(f"Error processing JSON: {str(e)}")
            return json_content
    
    def _extract_text_from_json(self, data: Any, current_path: str = "") -> List[str]:
        """
        Recursively extract text from JSON data.
        
        Args:
            data: The JSON data (can be dict, list, or primitive)
            current_path: Current path for nested values
            
        Returns:
            List of extracted text strings
        """
        texts = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                path = f"{current_path}.{key}" if current_path else key
                
                # Skip keys that are likely not to contain meaningful content
                if key.lower() in ("id", "uuid", "created_at", "updated_at", "timestamp", "date"):
                    continue
                
                # Check for text in the current key/value
                if isinstance(value, str) and len(value) > 10:
                    texts.append(f"{key}: {value}")
                
                # Recurse for nested structures
                texts.extend(self._extract_text_from_json(value, path))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                path = f"{current_path}[{i}]"
                texts.extend(self._extract_text_from_json(item, path))
        
        elif isinstance(data, str) and len(data) > 10:
            if current_path:
                texts.append(f"{current_path}: {data}")
            else:
                texts.append(data)
        
        return texts
    
    def _extract_links(self, content: str, base_url: str) -> List[str]:
        """
        Extract links from HTML content.
        
        Args:
            content: The HTML content
            base_url: The base URL for resolving relative links
            
        Returns:
            List of extracted links
        """
        links = []
        
        try:
            soup = BeautifulSoup(content, "html.parser")
            
            # Parse the base URL
            parsed_base = urlparse(base_url)
            base_netloc = parsed_base.netloc
            
            # Extract links from <a> tags
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                
                # Skip anchors, javascript, and mail links
                if href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
                    continue
                
                # Resolve relative URLs
                absolute_url = urljoin(base_url, href)
                
                # Parse the absolute URL
                parsed_url = urlparse(absolute_url)
                
                # Only include links from the same domain
                if parsed_url.netloc == base_netloc:
                    links.append(absolute_url)
        
        except Exception as e:
            logger.warning(f"Error extracting links: {str(e)}")
        
        return links
    
    def _check_cache(self, url: str) -> Optional[str]:
        """
        Check if URL content is in cache.
        
        Args:
            url: The URL to check
            
        Returns:
            Cached content if available, None otherwise
        """
        # Create a deterministic filename from URL
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{url_hash}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                
                # Check if the cache is still valid
                if time.time() - cache_data["timestamp"] <= self.cache_ttl:
                    return cache_data["content"]
                else:
                    # Cache expired
                    return None
        except Exception as e:
            logger.warning(f"Error reading cache for {url}: {str(e)}")
            return None
    
    def _update_cache(self, url: str, content: str) -> None:
        """
        Update cache with URL content.
        
        Args:
            url: The URL
            content: The content to cache
        """
        # Create a deterministic filename from URL
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{url_hash}.json"
        
        try:
            cache_data = {
                "url": url,
                "timestamp": time.time(),
                "content": content
            }
            
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Error updating cache for {url}: {str(e)}")
    
    async def _index_all_content(
        self,
        source: RAGSource,
        options: ProcessingOptions,
        stats: Dict[str, Any]
    ) -> None:
        """
        Index all collected content.
        
        Args:
            source: The source being processed
            options: Processing options
            stats: Statistics to update
        """
        if not self.indexer:
            logger.warning("No indexer available for content indexing")
            return
        
        total_chunks = 0
        
        # Index each piece of content
        for url, content in self.url_contents.items():
            # Skip empty content
            if not content:
                continue
            
            try:
                # Extract document title if possible
                title = url.split("/")[-1]
                if not title or title.isspace():
                    title = urlparse(url).netloc
                
                # Create a unique ID for the document
                document_id = str(uuid.uuid4())
                
                # Determine the content format
                content_format = options.content_format
                if content_format == ContentFormat.AUTO:
                    # Auto-detect format
                    if url.endswith(".md"):
                        content_format = ContentFormat.MARKDOWN
                    elif url.endswith(".json"):
                        content_format = ContentFormat.JSON
                    elif url.endswith(".txt"):
                        content_format = ContentFormat.TEXT
                    else:
                        content_format = ContentFormat.HTML
                
                # Process into chunks using the text processor
                chunks = self.text_processor._chunk_text(content, source=url)
                total_chunks += len(chunks)
                
                # Add source metadata to each chunk
                for chunk in chunks:
                    if chunk.get("metadata"):
                        chunk["metadata"]["source_url"] = url
                        chunk["metadata"]["source_type"] = source.source_type.value
                        if source.metadata and source.metadata.title:
                            chunk["metadata"]["source_title"] = source.metadata.title
                        else:
                            chunk["metadata"]["source_title"] = title
                
                # Save the indexed chunks
                output_file = os.path.join("knowledge_base", f"{document_id}.json")
                self.indexer.save_index(chunks, output_file)
                
            except Exception as e:
                logger.error(f"Error indexing content from {url}: {str(e)}")
        
        # Update stats
        stats["chunks_created"] = total_chunks
        
    import hashlib