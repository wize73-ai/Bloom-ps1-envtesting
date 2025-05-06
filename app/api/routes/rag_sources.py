"""
RAG Sources Routes for CasaLingua

This module provides API endpoints for managing RAG source URLs
that are used for training the Retrieval-Augmented Generation model.
"""

import os
import json
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form, Query, Path, Request, status, Response

from app.api.schemas.rag_sources import (
    RAGSource, SourceStatus, SourceType, ProcessingFrequency, ContentFormat,
    SourceCredentials, ProcessingOptions, SourceMetadata,
    AddSourceRequest, UpdateSourceRequest, ProcessSourceRequest,
    SourceListResponse, SourceResponse, ProcessingResponse, ProcessingStats
)
from app.api.schemas.base import BaseResponse, StatusEnum, MetadataModel, ErrorDetail
from app.api.middleware.auth import get_current_user
from app.utils.logging import get_logger
from app.core.rag.indexer import Indexer
from app.core.rag.content_fetcher import ContentFetcher

logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["RAG Sources"])

# Path to RAG sources config file
CONFIG_DIR = os.path.join(os.getcwd(), "config")
SOURCES_CONFIG_PATH = os.path.join(CONFIG_DIR, "rag_sources.json")

# In-memory sources cache
_sources_cache: Dict[str, RAGSource] = {}
_sources_loaded = False


async def _load_sources() -> Dict[str, RAGSource]:
    """
    Load RAG sources from configuration file.
    
    Returns:
        Dictionary of sources by ID
    """
    global _sources_cache, _sources_loaded
    
    if _sources_loaded:
        return _sources_cache
    
    try:
        if os.path.exists(SOURCES_CONFIG_PATH):
            with open(SOURCES_CONFIG_PATH, 'r') as f:
                sources_config = json.load(f)
            
            # Convert legacy format if needed (just a list of URLs)
            if "github_repos" in sources_config and isinstance(sources_config["github_repos"], list):
                github_repos = sources_config.get("github_repos", [])
                
                # Create RAGSource objects for existing GitHub repos
                for repo_url in github_repos:
                    source_id = str(uuid.uuid4())
                    _sources_cache[source_id] = RAGSource(
                        id=source_id,
                        url=repo_url,
                        source_type=SourceType.GITHUB_REPO,
                        status=SourceStatus.ACTIVE,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        processing_frequency=ProcessingFrequency.WEEKLY
                    )
            
            # Load sources from modern format
            if "sources" in sources_config and isinstance(sources_config["sources"], list):
                for source_data in sources_config["sources"]:
                    # Create source object from data
                    if "id" not in source_data:
                        source_data["id"] = str(uuid.uuid4())
                    
                    # Convert string dates to datetime objects
                    if "created_at" in source_data and isinstance(source_data["created_at"], str):
                        try:
                            source_data["created_at"] = datetime.fromisoformat(source_data["created_at"].replace("Z", "+00:00"))
                        except ValueError:
                            source_data["created_at"] = datetime.now()
                    
                    if "updated_at" in source_data and isinstance(source_data["updated_at"], str):
                        try:
                            source_data["updated_at"] = datetime.fromisoformat(source_data["updated_at"].replace("Z", "+00:00"))
                        except ValueError:
                            source_data["updated_at"] = datetime.now()
                    
                    if "last_processed" in source_data and isinstance(source_data["last_processed"], str):
                        try:
                            source_data["last_processed"] = datetime.fromisoformat(source_data["last_processed"].replace("Z", "+00:00"))
                        except ValueError:
                            source_data["last_processed"] = None
                    
                    # Create the source object
                    try:
                        source = RAGSource(**source_data)
                        _sources_cache[source.id] = source
                    except Exception as e:
                        logger.warning(f"Error loading source from config: {e}")
        else:
            logger.info(f"No RAG sources configuration found at {SOURCES_CONFIG_PATH}, starting with empty sources")
        
        _sources_loaded = True
        logger.info(f"Loaded {len(_sources_cache)} RAG sources")
        return _sources_cache
    
    except Exception as e:
        logger.error(f"Error loading RAG sources: {e}")
        _sources_loaded = True
        return {}


async def _save_sources() -> bool:
    """
    Save RAG sources to configuration file.
    
    Returns:
        True if successful, False otherwise
    """
    global _sources_cache
    
    try:
        # Convert sources to dict for JSON serialization
        sources_data = []
        for source in _sources_cache.values():
            source_dict = source.dict()
            
            # Convert datetime objects to ISO strings
            if source_dict.get("created_at"):
                source_dict["created_at"] = source_dict["created_at"].isoformat()
            if source_dict.get("updated_at"):
                source_dict["updated_at"] = source_dict["updated_at"].isoformat()
            if source_dict.get("last_processed"):
                source_dict["last_processed"] = source_dict["last_processed"].isoformat()
            
            sources_data.append(source_dict)
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(SOURCES_CONFIG_PATH), exist_ok=True)
        
        # Save to file
        with open(SOURCES_CONFIG_PATH, 'w') as f:
            json.dump({"sources": sources_data}, f, indent=2)
        
        logger.info(f"Saved {len(_sources_cache)} RAG sources to {SOURCES_CONFIG_PATH}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving RAG sources: {e}")
        return False


@router.get(
    "/sources",
    response_model=SourceListResponse,
    summary="List all RAG sources",
    description="Get a list of all configured RAG sources for training the RAG model."
)
async def list_sources(
    request: Request,
    source_type: Optional[SourceType] = Query(None, description="Filter by source type"),
    status: Optional[SourceStatus] = Query(None, description="Filter by status"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    List all configured RAG sources.
    
    Optionally filter by source type or status.
    """
    # Load sources
    sources = await _load_sources()
    
    # Apply filters if provided
    filtered_sources = list(sources.values())
    
    if source_type:
        filtered_sources = [s for s in filtered_sources if s.source_type == source_type]
    
    if status:
        filtered_sources = [s for s in filtered_sources if s.status == status]
    
    # Sort by updated_at (newest first)
    filtered_sources.sort(key=lambda s: s.updated_at or datetime.min, reverse=True)
    
    return SourceListResponse(
        status=StatusEnum.SUCCESS,
        message=f"Retrieved {len(filtered_sources)} RAG sources",
        data=filtered_sources,
        metadata=MetadataModel(
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
            process_time=0.0
        )
    )


@router.get(
    "/sources/{source_id}",
    response_model=SourceResponse,
    summary="Get a specific RAG source",
    description="Get details of a specific RAG source by ID."
)
async def get_source(
    request: Request,
    source_id: str = Path(..., description="The ID of the source to retrieve"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get details of a specific RAG source by ID.
    """
    # Load sources
    sources = await _load_sources()
    
    # Check if source exists
    if source_id not in sources:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RAG source with ID {source_id} not found"
        )
    
    source = sources[source_id]
    
    return SourceResponse(
        status=StatusEnum.SUCCESS,
        message=f"Retrieved RAG source {source_id}",
        data=source,
        metadata=MetadataModel(
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
            process_time=0.0
        )
    )


@router.post(
    "/sources",
    response_model=SourceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a new RAG source",
    description="Add a new RAG source URL to train the RAG model."
)
async def add_source(
    request: Request,
    background_tasks: BackgroundTasks,
    source_request: AddSourceRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Add a new RAG source.
    
    Optionally process the source immediately after adding.
    """
    # Load sources
    sources = await _load_sources()
    
    # Check if source with same URL already exists
    for existing_source in sources.values():
        if str(existing_source.url) == str(source_request.url):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"RAG source with URL {source_request.url} already exists"
            )
    
    # Generate a new source ID
    source_id = str(uuid.uuid4())
    
    # Create source object
    now = datetime.now()
    source = RAGSource(
        id=source_id,
        url=source_request.url,
        source_type=source_request.source_type,
        status=SourceStatus.ACTIVE,
        created_at=now,
        updated_at=now,
        processing_frequency=source_request.processing_frequency or ProcessingFrequency.ONCE,
        credentials=source_request.credentials,
        options=source_request.options or ProcessingOptions(),
        metadata=source_request.metadata or SourceMetadata()
    )
    
    # Add to sources cache
    sources[source_id] = source
    
    # Save sources
    await _save_sources()
    
    # Process source if requested
    if source_request.process_now:
        # Update status to processing
        source.status = SourceStatus.PROCESSING
        await _save_sources()
        
        # Process in background
        background_tasks.add_task(
            _process_source_task,
            request.app,
            source_id
        )
    
    return SourceResponse(
        status=StatusEnum.SUCCESS,
        message=f"Added RAG source {source_id}",
        data=source,
        metadata=MetadataModel(
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
            process_time=0.0
        )
    )


@router.put(
    "/sources/{source_id}",
    response_model=SourceResponse,
    summary="Update a RAG source",
    description="Update an existing RAG source configuration."
)
async def update_source(
    request: Request,
    source_id: str,
    source_update: UpdateSourceRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update an existing RAG source configuration.
    """
    # Load sources
    sources = await _load_sources()
    
    # Check if source exists
    if source_id not in sources:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RAG source with ID {source_id} not found"
        )
    
    # Get existing source
    source = sources[source_id]
    
    # Update source with provided values
    update_data = source_update.dict(exclude_unset=True)
    
    # Handle nested updates for options, credentials, and metadata
    if "options" in update_data and update_data["options"]:
        if source.options:
            # Update existing options
            for key, value in update_data["options"].items():
                if value is not None:
                    setattr(source.options, key, value)
        else:
            # Create new options
            source.options = ProcessingOptions(**update_data["options"])
        # Remove from update data to prevent overwriting
        del update_data["options"]
    
    if "credentials" in update_data and update_data["credentials"]:
        if source.credentials:
            # Update existing credentials
            for key, value in update_data["credentials"].items():
                if value is not None:
                    setattr(source.credentials, key, value)
        else:
            # Create new credentials
            source.credentials = SourceCredentials(**update_data["credentials"])
        # Remove from update data to prevent overwriting
        del update_data["credentials"]
    
    if "metadata" in update_data and update_data["metadata"]:
        if source.metadata:
            # Update existing metadata
            for key, value in update_data["metadata"].items():
                if value is not None:
                    setattr(source.metadata, key, value)
        else:
            # Create new metadata
            source.metadata = SourceMetadata(**update_data["metadata"])
        # Remove from update data to prevent overwriting
        del update_data["metadata"]
    
    # Update remaining fields
    for key, value in update_data.items():
        if value is not None:
            setattr(source, key, value)
    
    # Update timestamp
    source.updated_at = datetime.now()
    
    # Save sources
    await _save_sources()
    
    return SourceResponse(
        status=StatusEnum.SUCCESS,
        message=f"Updated RAG source {source_id}",
        data=source,
        metadata=MetadataModel(
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
            process_time=0.0
        )
    )


@router.delete(
    "/sources/{source_id}",
    response_model=BaseResponse,
    summary="Delete a RAG source",
    description="Delete an existing RAG source configuration."
)
async def delete_source(
    request: Request,
    source_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete an existing RAG source configuration.
    """
    # Load sources
    sources = await _load_sources()
    
    # Check if source exists
    if source_id not in sources:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RAG source with ID {source_id} not found"
        )
    
    # Remove source
    del sources[source_id]
    
    # Save sources
    await _save_sources()
    
    return BaseResponse(
        status=StatusEnum.SUCCESS,
        message=f"Deleted RAG source {source_id}",
        metadata=MetadataModel(
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
            process_time=0.0
        )
    )


@router.post(
    "/sources/{source_id}/process",
    response_model=ProcessingResponse,
    summary="Process a RAG source",
    description="Process a RAG source to fetch its content and train the RAG model."
)
async def process_source(
    request: Request,
    background_tasks: BackgroundTasks,
    source_id: str,
    process_request: ProcessSourceRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Process a RAG source to fetch its content and train the RAG model.
    
    The processing will be performed in the background.
    """
    # Load sources
    sources = await _load_sources()
    
    # Check if source exists
    if source_id not in sources:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RAG source with ID {source_id} not found"
        )
    
    source = sources[source_id]
    
    # Check if already processing
    if source.status == SourceStatus.PROCESSING and not process_request.force:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"RAG source {source_id} is already being processed"
        )
    
    # Update status to processing
    source.status = SourceStatus.PROCESSING
    source.updated_at = datetime.now()
    await _save_sources()
    
    # Initial processing stats
    stats = ProcessingStats(
        url=source.url,
        start_time=datetime.now(),
        success=False
    )
    
    # Process in background
    background_tasks.add_task(
        _process_source_task,
        request.app,
        source_id,
        process_request.options
    )
    
    return ProcessingResponse(
        status=StatusEnum.SUCCESS,
        message=f"Processing RAG source {source_id} (in background)",
        data=stats,
        metadata=MetadataModel(
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
            process_time=0.0
        )
    )


@router.post(
    "/process/all",
    response_model=BaseResponse,
    summary="Process all RAG sources",
    description="Process all active RAG sources to fetch their content and train the RAG model."
)
async def process_all_sources(
    request: Request,
    background_tasks: BackgroundTasks,
    source_type: Optional[SourceType] = Query(None, description="Filter by source type"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Process all active RAG sources.
    
    The processing will be performed in the background.
    """
    # Load sources
    sources = await _load_sources()
    
    # Filter active sources
    active_sources = [
        s for s in sources.values() 
        if s.status != SourceStatus.PROCESSING and s.enabled and
        (source_type is None or s.source_type == source_type)
    ]
    
    # Check if any sources to process
    if not active_sources:
        return BaseResponse(
            status=StatusEnum.SUCCESS,
            message="No active RAG sources to process",
            metadata=MetadataModel(
                request_id=str(uuid.uuid4()),
                timestamp=time.time(),
                process_time=0.0
            )
        )
    
    # Update status to processing for all sources
    for source in active_sources:
        source.status = SourceStatus.PROCESSING
        source.updated_at = datetime.now()
    
    # Save sources
    await _save_sources()
    
    # Process all sources in background
    for source in active_sources:
        background_tasks.add_task(
            _process_source_task,
            request.app,
            source.id
        )
    
    return BaseResponse(
        status=StatusEnum.SUCCESS,
        message=f"Processing {len(active_sources)} RAG sources (in background)",
        metadata=MetadataModel(
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
            process_time=0.0
        )
    )


async def _process_source_task(
    app,
    source_id: str,
    options: Optional[ProcessingOptions] = None
) -> None:
    """
    Background task to process a RAG source.
    
    Args:
        app: FastAPI application instance
        source_id: Source ID to process
        options: Optional processing options to override source options
    """
    # Load sources
    sources = await _load_sources()
    
    # Check if source exists
    if source_id not in sources:
        logger.error(f"RAG source {source_id} not found for processing")
        return
    
    source = sources[source_id]
    
    try:
        # Get indexer from app state
        indexer = None
        rag_expert = None
        
        if hasattr(app.state, "processor"):
            if hasattr(app.state.processor, "indexer"):
                indexer = app.state.processor.indexer
            if hasattr(app.state.processor, "rag_expert"):
                rag_expert = app.state.processor.rag_expert
        
        # Create content fetcher
        content_fetcher = ContentFetcher(
            config=app.state.config if hasattr(app.state, "config") else None,
            indexer=indexer
        )
        
        # Process the source
        process_result = await content_fetcher.process_source(source, options)
        
        # Update source based on result
        source.last_processed = datetime.now()
        
        if process_result.get("success", False):
            source.status = SourceStatus.ACTIVE
            logger.info(f"Successfully processed RAG source {source_id}: {process_result.get('chunks_created', 0)} chunks created")
        else:
            source.status = SourceStatus.ERROR
            error_message = process_result.get("error", "Unknown error")
            logger.error(f"Error processing RAG source {source_id}: {error_message}")
        
        # Save sources
        await _save_sources()
        
        # Clean up content fetcher
        await content_fetcher.cleanup()
        
        # Rebuild RAG index if available
        if rag_expert and process_result.get("success", False):
            try:
                await rag_expert._build_index()
                logger.info(f"Rebuilt RAG index after processing source {source_id}")
            except Exception as e:
                logger.error(f"Error rebuilding RAG index: {e}")
    
    except Exception as e:
        # Update source status on error
        source.status = SourceStatus.ERROR
        source.last_processed = datetime.now()
        
        # Save sources
        await _save_sources()
        
        logger.error(f"Unhandled error processing RAG source {source_id}: {e}", exc_info=True)