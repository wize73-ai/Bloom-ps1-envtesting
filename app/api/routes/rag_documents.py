"""
RAG Document Routes for CasaLingua

This module provides API endpoints for indexing documents into the
Retrieval-Augmented Generation (RAG) system.
"""

import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form, Query, Path, Request, status, Cookie, Response

from app.api.schemas.base import BaseResponse, StatusEnum, MetadataModel, ErrorDetail
from app.api.middleware.auth import get_current_user
from app.utils.logging import get_logger
from app.services.storage.session_manager import SessionManager
from app.core.rag.indexer import Indexer

logger = get_logger(__name__)

# Create router
router = APIRouter()

# Get the session manager instance
session_manager = SessionManager()

@router.post(
    "/index/document",
    response_model=BaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Index a document for RAG",
    description="Index a document for use with the RAG system."
)
async def index_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    store_in_session: bool = Form(True, description="Whether to store the document in the user's session"),
    session_id: Optional[str] = Cookie(None, description="Session identifier for document storage"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Index a document for RAG (Retrieval-Augmented Generation).
    
    This endpoint processes and indexes a document, making its content available 
    for context-aware language processing via the RAG system.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Generate session ID if not provided and if storing in session
    if store_in_session and not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        
        rag_expert = None
        if hasattr(request.app.state, "processor") and hasattr(request.app.state.processor, "rag_expert"):
            rag_expert = request.app.state.processor.rag_expert
        
        metrics = request.app.state.metrics
        audit_logger = request.app.state.audit_logger
        
        # Determine document type from file content type or extension
        content_type = file.content_type
        filename = file.filename
        
        # Map content type to document type
        if "pdf" in content_type.lower():
            document_type = "application/pdf"
        elif "word" in content_type.lower() or filename.lower().endswith((".docx", ".doc")):
            document_type = "application/docx"
        elif any(img_type in content_type.lower() for img_type in ["image", "png", "jpeg", "jpg"]):
            document_type = "image"
        else:
            document_type = content_type
            
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/rag/index/document",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "filename": filename,
                "document_type": document_type,
                "store_in_session": store_in_session,
                "session_id": session_id
            }
        )
        
        # Read file content
        content = await file.read()
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Store in session if requested
        if store_in_session and session_id:
            await session_manager.add_document(
                session_id=session_id,
                document_id=document_id,
                content=content,
                metadata={
                    "filename": filename,
                    "content_type": content_type,
                    "document_type": document_type,
                    "size": len(content),
                    "user_id": current_user["id"],
                    "created_at": time.time()
                }
            )
        
        # Create indexer
        indexer = Indexer(
            output_dir="knowledge_base",
            rag_expert=rag_expert,
            processor=processor
        )
        
        # Index the document
        if store_in_session and session_id:
            # Index from session
            background_tasks.add_task(
                indexer.index_session_document,
                session_id=session_id,
                document_id=document_id
            )
            result = {
                "document_id": document_id,
                "session_id": session_id,
                "status": "indexing",
                "message": "Document indexing started in background"
            }
        else:
            # Index directly
            background_tasks.add_task(
                indexer.index_document_content,
                document_content=content,
                document_type=document_type,
                filename=filename,
                metadata={
                    "document_id": document_id,
                    "user_id": current_user["id"]
                }
            )
            result = {
                "document_id": document_id,
                "status": "indexing",
                "message": "Document indexing started in background"
            }
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="rag_indexing",
            operation="index_document",
            duration=process_time,
            input_size=len(content),
            output_size=0,
            success=True,
            metadata={
                "document_type": document_type,
                "filename": filename,
                "document_id": document_id,
                "session_id": session_id if store_in_session else None
            }
        )
        
        # Create response
        response_data = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Document indexing initiated successfully",
            data=result,
            metadata=MetadataModel(
                request_id=request_id,
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=process_time
            ),
            errors=None,
            pagination=None
        )
        
        # Set session cookie in response if storing in session
        if store_in_session and session_id:
            response_obj = Response(response_data.dict())
            response_obj.set_cookie(
                key="session_id",
                value=session_id,
                max_age=3600,  # 1 hour
                httponly=True,
                samesite="lax"
            )
            return response_obj
        
        return response_data
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}", exc_info=True)
        
        # Record error metrics
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="rag_indexing",
            operation="index_document",
            duration=time.time() - start_time,
            input_size=len(await file.read()) if not file.closed else 0,
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "filename": file.filename
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document indexing error: {str(e)}"
        )
    finally:
        # Make sure to close the file
        await file.close()


@router.post(
    "/index/session",
    response_model=BaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Index all session documents for RAG",
    description="Index all documents from the current session for use with the RAG system."
)
async def index_session_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    session_id: str = Cookie(None, description="Session identifier"),
    document_ids: Optional[List[str]] = Query(None, description="Optional list of document IDs to index"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Index all documents from the current session for RAG.
    
    This endpoint processes and indexes all documents in the user's session,
    making their content available for context-aware language processing via the RAG system.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Check if session ID is provided
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID is required"
        )
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        if processor is None:
            raise HTTPException(status_code=503, detail="Processor not initialized")
        
        rag_expert = None
        if hasattr(request.app.state, "processor") and hasattr(request.app.state.processor, "rag_expert"):
            rag_expert = request.app.state.processor.rag_expert
        
        metrics = request.app.state.metrics
        audit_logger = request.app.state.audit_logger
        
        # Log request to audit log
        await audit_logger.log_api_request(
            endpoint="/rag/index/session",
            method="POST",
            user_id=current_user["id"],
            source_ip=request.client.host,
            request_id=request_id,
            request_params={
                "session_id": session_id,
                "document_ids": document_ids
            }
        )
        
        # Get session document count
        session_docs = await session_manager.get_all_documents(session_id)
        if not session_docs:
            return BaseResponse(
                status=StatusEnum.ERROR,
                message="No documents found in session",
                data={"session_id": session_id, "document_count": 0},
                metadata=MetadataModel(
                    request_id=request_id,
                    timestamp=time.time(),
                    version=request.app.state.config.get("version", "1.0.0"),
                    process_time=time.time() - start_time
                ),
                errors=[ErrorDetail(
                    code="no_documents",
                    message="No documents found in session",
                    location="session"
                )],
                pagination=None
            )
        
        # Filter documents if specific IDs were provided
        if document_ids:
            session_docs = [doc for doc in session_docs if doc.get("document_id") in document_ids]
            if not session_docs:
                return BaseResponse(
                    status=StatusEnum.ERROR,
                    message="No matching documents found in session",
                    data={"session_id": session_id, "document_count": 0},
                    metadata=MetadataModel(
                        request_id=request_id,
                        timestamp=time.time(),
                        version=request.app.state.config.get("version", "1.0.0"),
                        process_time=time.time() - start_time
                    ),
                    errors=[ErrorDetail(
                        code="no_matching_documents",
                        message="No matching documents found in session",
                        location="document_ids"
                    )],
                    pagination=None
                )
        
        # Create indexer
        indexer = Indexer(
            output_dir="knowledge_base",
            rag_expert=rag_expert,
            processor=processor
        )
        
        # Start indexing in background
        doc_ids_to_index = [doc.get("document_id") for doc in session_docs]
        background_tasks.add_task(
            indexer.index_session_documents,
            session_id=session_id,
            document_ids=doc_ids_to_index
        )
        
        # Create result
        result = {
            "session_id": session_id,
            "document_count": len(session_docs),
            "document_ids": doc_ids_to_index,
            "status": "indexing",
            "message": "Session documents indexing started in background"
        }
        
        # Calculate process time
        process_time = time.time() - start_time
        
        # Record metrics
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="rag_indexing",
            operation="index_session",
            duration=process_time,
            input_size=len(session_docs),
            output_size=0,
            success=True,
            metadata={
                "session_id": session_id,
                "document_count": len(session_docs)
            }
        )
        
        # Create response
        return BaseResponse(
            status=StatusEnum.SUCCESS,
            message=f"Indexing {len(session_docs)} documents from session",
            data=result,
            metadata=MetadataModel(
                request_id=request_id,
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=process_time
            ),
            errors=None,
            pagination=None
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error indexing session documents: {str(e)}", exc_info=True)
        
        # Record error metrics
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="rag_indexing",
            operation="index_session",
            duration=time.time() - start_time,
            input_size=0,
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "session_id": session_id
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Session document indexing error: {str(e)}"
        )