"""
RAG Sources Schemas for CasaLingua

This module defines the schema models for RAG source URLs and related entities.
"""

from enum import Enum
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, HttpUrl, validator

from app.api.schemas.base import BaseRequest, BaseResponse


class SourceStatus(str, Enum):
    """Status of a RAG source."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PROCESSING = "processing"
    ERROR = "error"


class SourceType(str, Enum):
    """Type of a RAG source."""
    WEBPAGE = "webpage"
    GITHUB_REPO = "github_repo"
    API = "api"
    RSS = "rss"
    SITEMAP = "sitemap"


class AccessMethod(str, Enum):
    """Access method for the source."""
    PUBLIC = "public"
    API_KEY = "api_key"
    OAUTH = "oauth"
    BASIC_AUTH = "basic_auth"


class ProcessingFrequency(str, Enum):
    """Frequency of processing for the source."""
    ONCE = "once"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ContentFormat(str, Enum):
    """Format of the content to process."""
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"
    AUTO = "auto"


class SourceCredentials(BaseModel):
    """Credentials for accessing protected sources."""
    method: AccessMethod = Field(
        default=AccessMethod.PUBLIC,
        description="Method of access authentication"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for protected sources"
    )
    username: Optional[str] = Field(
        default=None,
        description="Username for basic auth"
    )
    password: Optional[str] = Field(
        default=None,
        description="Password for basic auth"
    )
    token: Optional[str] = Field(
        default=None,
        description="OAuth token"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "method": "api_key",
                "api_key": "your_api_key_here"
            }
        }


class ProcessingOptions(BaseModel):
    """Options for processing the source."""
    extract_links: bool = Field(
        default=False,
        description="Whether to extract and follow links from the source"
    )
    max_depth: int = Field(
        default=1,
        description="Maximum depth for link crawling"
    )
    content_format: ContentFormat = Field(
        default=ContentFormat.AUTO,
        description="Format of the content to process"
    )
    extract_text_only: bool = Field(
        default=True,
        description="Whether to extract only text content"
    )
    chunk_size: int = Field(
        default=500,
        description="Size of text chunks for indexing"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "extract_links": True,
                "max_depth": 2,
                "content_format": "auto",
                "extract_text_only": True,
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        }


class SourceMetadata(BaseModel):
    """Metadata about a source."""
    title: Optional[str] = Field(
        default=None,
        description="Title of the source"
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the source"
    )
    author: Optional[str] = Field(
        default=None,
        description="Author of the content"
    )
    language: Optional[str] = Field(
        default=None,
        description="Primary language of the source"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing the source"
    )
    domain: Optional[str] = Field(
        default=None,
        description="Domain of the source URL"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Housing Resources",
                "description": "Information about housing resources and programs",
                "author": "Housing Authority",
                "language": "en",
                "tags": ["housing", "resources", "programs"],
                "domain": "example.org"
            }
        }


class RAGSource(BaseModel):
    """RAG source model."""
    id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the source"
    )
    url: HttpUrl = Field(
        ...,
        description="URL of the source"
    )
    source_type: SourceType = Field(
        default=SourceType.WEBPAGE,
        description="Type of the source"
    )
    status: SourceStatus = Field(
        default=SourceStatus.ACTIVE,
        description="Status of the source"
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="When the source was created"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="When the source was last updated"
    )
    last_processed: Optional[datetime] = Field(
        default=None,
        description="When the source was last processed"
    )
    processing_frequency: ProcessingFrequency = Field(
        default=ProcessingFrequency.ONCE,
        description="How often to process the source"
    )
    credentials: Optional[SourceCredentials] = Field(
        default=None,
        description="Credentials for accessing the source"
    )
    options: ProcessingOptions = Field(
        default_factory=ProcessingOptions,
        description="Options for processing the source"
    )
    metadata: Optional[SourceMetadata] = Field(
        default=None,
        description="Metadata about the source"
    )
    enabled: bool = Field(
        default=True,
        description="Whether the source is enabled for processing"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "url": "https://example.org/housing-resources",
                "source_type": "webpage",
                "status": "active",
                "processing_frequency": "weekly",
                "options": {
                    "extract_links": True,
                    "max_depth": 2,
                    "content_format": "auto",
                    "extract_text_only": True
                },
                "metadata": {
                    "title": "Housing Resources",
                    "tags": ["housing", "resources"]
                },
                "enabled": True
            }
        }


class AddSourceRequest(BaseRequest):
    """Request model for adding a new source."""
    url: HttpUrl = Field(
        ...,
        description="URL of the source"
    )
    source_type: SourceType = Field(
        default=SourceType.WEBPAGE,
        description="Type of the source"
    )
    processing_frequency: Optional[ProcessingFrequency] = Field(
        default=ProcessingFrequency.ONCE,
        description="How often to process the source"
    )
    credentials: Optional[SourceCredentials] = Field(
        default=None,
        description="Credentials for accessing the source"
    )
    options: Optional[ProcessingOptions] = Field(
        default=None,
        description="Options for processing the source"
    )
    metadata: Optional[SourceMetadata] = Field(
        default=None,
        description="Metadata about the source"
    )
    process_now: bool = Field(
        default=False,
        description="Whether to process the source immediately after adding"
    )


class UpdateSourceRequest(BaseRequest):
    """Request model for updating an existing source."""
    url: Optional[HttpUrl] = Field(
        default=None,
        description="URL of the source"
    )
    source_type: Optional[SourceType] = Field(
        default=None,
        description="Type of the source"
    )
    status: Optional[SourceStatus] = Field(
        default=None,
        description="Status of the source"
    )
    processing_frequency: Optional[ProcessingFrequency] = Field(
        default=None,
        description="How often to process the source"
    )
    credentials: Optional[SourceCredentials] = Field(
        default=None,
        description="Credentials for accessing the source"
    )
    options: Optional[ProcessingOptions] = Field(
        default=None,
        description="Options for processing the source"
    )
    metadata: Optional[SourceMetadata] = Field(
        default=None,
        description="Metadata about the source"
    )
    enabled: Optional[bool] = Field(
        default=None,
        description="Whether the source is enabled for processing"
    )


class ProcessSourceRequest(BaseRequest):
    """Request model for processing a source."""
    force: bool = Field(
        default=False,
        description="Whether to force processing even if already processed recently"
    )
    options: Optional[ProcessingOptions] = Field(
        default=None,
        description="Processing options override"
    )


class SourceListResponse(BaseResponse):
    """Response model for listing sources."""
    data: List[RAGSource] = Field(
        default_factory=list,
        description="List of sources"
    )


class SourceResponse(BaseResponse):
    """Response model for a single source."""
    data: RAGSource = Field(
        ...,
        description="Source data"
    )


class ProcessingStats(BaseModel):
    """Statistics about source processing."""
    url: HttpUrl = Field(
        ...,
        description="URL that was processed"
    )
    start_time: datetime = Field(
        ...,
        description="When processing started"
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="When processing completed"
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Processing duration in seconds"
    )
    content_size: Optional[int] = Field(
        default=None,
        description="Size of the content in bytes"
    )
    chunks_created: Optional[int] = Field(
        default=None,
        description="Number of text chunks created"
    )
    links_found: Optional[int] = Field(
        default=None,
        description="Number of links found"
    )
    links_followed: Optional[int] = Field(
        default=None,
        description="Number of links followed"
    )
    success: bool = Field(
        default=False,
        description="Whether processing was successful"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )


class ProcessingResponse(BaseResponse):
    """Response model for source processing."""
    data: ProcessingStats = Field(
        ...,
        description="Processing statistics"
    )