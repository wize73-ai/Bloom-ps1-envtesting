"""
Session Manager for CasaLingua

This module manages user session data, including temporary document storage
and cleanup functionality when sessions expire.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
import shutil
import os
import uuid

from app.utils.logging import get_logger

logger = get_logger(__name__)

class SessionManager:
    """
    Manages user session data, including document storage with automatic cleanup.
    
    Features:
    - Store document data and metadata per user session
    - File system storage for large documents
    - Automatic cleanup when sessions expire
    - Thread-safe session access and management
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one session manager exists."""
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the session manager.
        
        Args:
            config: Optional configuration dictionary
        """
        # Only initialize once due to singleton pattern
        if self._initialized:
            return
            
        self.config = config or {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_times: Dict[str, float] = {}
        self.temp_session_dirs: Dict[str, Path] = {}
        
        # Configuration
        self.session_ttl = self.config.get("session_ttl_seconds", 3600)  # 1 hour default
        self.cleanup_interval = self.config.get("cleanup_interval_seconds", 300)  # 5 minutes
        self.max_sessions = self.config.get("max_sessions", 1000)
        self.temp_dir = Path(self.config.get("temp_dir", "temp/sessions"))
        
        # Create temp directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Session access lock
        self.lock = asyncio.Lock()
        
        # Set up cleanup task (will be started during app initialization)
        self.cleanup_task = None
        
        self._initialized = True
        logger.info(f"Session manager initialized with {self.session_ttl}s TTL, {self.cleanup_interval}s cleanup interval")
    
    async def get_session(self, session_id: str, create_if_missing: bool = True) -> Dict[str, Any]:
        """
        Get or create a session.
        
        Args:
            session_id: Session identifier
            create_if_missing: Whether to create a new session if not found
            
        Returns:
            Session data dictionary
        """
        async with self.lock:
            current_time = time.time()
            
            # Check if session exists
            if session_id in self.sessions:
                # Update session time
                self.session_times[session_id] = current_time
                return self.sessions[session_id]
            
            # Create new session if requested
            if create_if_missing:
                # Ensure session ID is valid (UUID format)
                try:
                    # Attempt to parse as UUID to validate format
                    uuid.UUID(session_id)
                except ValueError:
                    # Generate a new valid session ID if the provided one is invalid
                    session_id = str(uuid.uuid4())
                    logger.warning(f"Invalid session ID provided, generated new ID: {session_id}")
                
                # Check if max sessions reached
                if len(self.sessions) >= self.max_sessions:
                    await self._evict_oldest_session()
                
                # Create session data
                self.sessions[session_id] = {
                    "session_id": session_id,
                    "created_at": current_time,
                    "documents": {},
                    "metadata": {}
                }
                
                # Set session time
                self.session_times[session_id] = current_time
                
                # Create session temp directory
                session_dir = self.temp_dir / session_id
                session_dir.mkdir(exist_ok=True)
                self.temp_session_dirs[session_id] = session_dir
                
                logger.debug(f"Created new session with ID: {session_id}")
                return self.sessions[session_id]
            
            # Return empty dictionary if session not found and not creating
            return {}
    
    async def add_document(self, 
                          session_id: str, 
                          document_id: str, 
                          content: bytes, 
                          metadata: Dict[str, Any]) -> str:
        """
        Add a document to a session.
        
        Args:
            session_id: Session identifier
            document_id: Document identifier (or None to generate)
            content: Document content as bytes
            metadata: Document metadata
            
        Returns:
            Document identifier
        """
        # Generate document ID if not provided
        if not document_id:
            document_id = str(uuid.uuid4())
        
        async with self.lock:
            # Get or create session
            session = await self.get_session(session_id)
            
            # Store document in session directory
            session_dir = self.temp_session_dirs[session_id]
            document_path = session_dir / f"{document_id}.bin"
            
            # Write document content to file
            with open(document_path, 'wb') as f:
                f.write(content)
            
            # Store document metadata in session
            session["documents"][document_id] = {
                "document_id": document_id,
                "file_path": str(document_path),
                "created_at": time.time(),
                "size": len(content),
                "metadata": metadata
            }
            
            # Update session time
            self.session_times[session_id] = time.time()
            
            logger.debug(f"Added document {document_id} to session {session_id}")
            return document_id
    
    async def get_document(self, 
                          session_id: str, 
                          document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document metadata and content.
        
        Args:
            session_id: Session identifier
            document_id: Document identifier
            
        Returns:
            Dictionary with document metadata and content, or None if not found
        """
        async with self.lock:
            # Check if session exists
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found")
                return None
            
            # Get session data
            session = self.sessions[session_id]
            
            # Check if document exists in session
            if document_id not in session["documents"]:
                logger.warning(f"Document {document_id} not found in session {session_id}")
                return None
            
            # Get document metadata
            document_info = session["documents"][document_id]
            
            # Read document content from file
            try:
                with open(document_info["file_path"], 'rb') as f:
                    content = f.read()
                
                # Update session time
                self.session_times[session_id] = time.time()
                
                # Return document info with content
                return {
                    **document_info,
                    "content": content
                }
            except Exception as e:
                logger.error(f"Error reading document {document_id}: {str(e)}")
                return None
    
    async def get_all_documents(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get metadata for all documents in a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of document metadata dictionaries (without content)
        """
        async with self.lock:
            # Check if session exists
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found")
                return []
            
            # Get session data
            session = self.sessions[session_id]
            
            # Update session time
            self.session_times[session_id] = time.time()
            
            # Return list of document metadata
            return list(session["documents"].values())
    
    async def remove_document(self, session_id: str, document_id: str) -> bool:
        """
        Remove a document from a session.
        
        Args:
            session_id: Session identifier
            document_id: Document identifier
            
        Returns:
            True if document was removed, False otherwise
        """
        async with self.lock:
            # Check if session exists
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            # Get session data
            session = self.sessions[session_id]
            
            # Check if document exists in session
            if document_id not in session["documents"]:
                logger.warning(f"Document {document_id} not found in session {session_id}")
                return False
            
            # Get document file path
            document_info = session["documents"][document_id]
            document_path = document_info["file_path"]
            
            # Remove document file
            try:
                os.remove(document_path)
            except Exception as e:
                logger.warning(f"Error removing document file {document_path}: {str(e)}")
            
            # Remove document from session
            del session["documents"][document_id]
            
            # Update session time
            self.session_times[session_id] = time.time()
            
            logger.debug(f"Removed document {document_id} from session {session_id}")
            return True
    
    async def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update session metadata.
        
        Args:
            session_id: Session identifier
            metadata: Metadata to update
            
        Returns:
            True if session was updated, False otherwise
        """
        async with self.lock:
            # Check if session exists
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            # Get session data
            session = self.sessions[session_id]
            
            # Update metadata
            session["metadata"].update(metadata)
            
            # Update session time
            self.session_times[session_id] = time.time()
            
            logger.debug(f"Updated metadata for session {session_id}")
            return True
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was deleted, False otherwise
        """
        async with self.lock:
            # Check if session exists
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            # Clean up session directory
            if session_id in self.temp_session_dirs:
                session_dir = self.temp_session_dirs[session_id]
                try:
                    shutil.rmtree(session_dir)
                except Exception as e:
                    logger.warning(f"Error removing session directory {session_dir}: {str(e)}")
                
                # Remove from session directories
                del self.temp_session_dirs[session_id]
            
            # Remove session data
            del self.sessions[session_id]
            
            # Remove session time
            if session_id in self.session_times:
                del self.session_times[session_id]
            
            logger.debug(f"Deleted session {session_id}")
            return True
    
    async def _evict_oldest_session(self) -> None:
        """Evict the oldest session when max sessions is reached."""
        # Find oldest session based on last access time
        oldest_session_id = None
        oldest_time = float('inf')
        
        for session_id, timestamp in self.session_times.items():
            if timestamp < oldest_time:
                oldest_time = timestamp
                oldest_session_id = session_id
        
        if oldest_session_id:
            logger.info(f"Evicting oldest session {oldest_session_id} due to max sessions limit")
            await self.delete_session(oldest_session_id)
    
    async def _cleanup_task(self) -> None:
        """Background task for cleaning up expired sessions."""
        logger.info(f"Starting session cleanup task with interval {self.cleanup_interval}s")
        
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                async with self.lock:
                    current_time = time.time()
                    expired_sessions = []
                    
                    # Find expired sessions
                    for session_id, last_access_time in self.session_times.items():
                        if current_time - last_access_time > self.session_ttl:
                            expired_sessions.append(session_id)
                    
                    # Clean up expired sessions
                    for session_id in expired_sessions:
                        logger.info(f"Cleaning up expired session {session_id}")
                        
                        # Clean up session directory
                        if session_id in self.temp_session_dirs:
                            session_dir = self.temp_session_dirs[session_id]
                            try:
                                shutil.rmtree(session_dir)
                                logger.debug(f"Removed session directory {session_dir}")
                            except Exception as e:
                                logger.warning(f"Error removing session directory {session_dir}: {str(e)}")
                            
                            # Remove from session directories
                            del self.temp_session_dirs[session_id]
                        
                        # Remove session data
                        if session_id in self.sessions:
                            del self.sessions[session_id]
                        
                        # Remove session time
                        if session_id in self.session_times:
                            del self.session_times[session_id]
                    
                    if expired_sessions:
                        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            except asyncio.CancelledError:
                logger.info("Session cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in session cleanup task: {str(e)}", exc_info=True)
                await asyncio.sleep(30)  # Wait before retrying
    
    async def start_cleanup_task(self) -> None:
        """
        Start the cleanup task.
        
        This should be called during application startup once the event loop is running.
        """
        if self.cleanup_task is None or self.cleanup_task.done():
            logger.info("Starting session cleanup task")
            self.cleanup_task = asyncio.create_task(self._cleanup_task())
    
    async def cleanup(self) -> None:
        """
        Clean up and stop the session manager.
        
        This should be called during application shutdown.
        """
        logger.info("Cleaning up session manager")
        
        # Cancel cleanup task
        if hasattr(self, 'cleanup_task') and self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all session directories
        async with self.lock:
            for session_id, session_dir in self.temp_session_dirs.items():
                try:
                    if session_dir.exists():
                        shutil.rmtree(session_dir)
                        logger.debug(f"Removed session directory {session_dir}")
                except Exception as e:
                    logger.warning(f"Error removing session directory {session_dir}: {str(e)}")
            
            # Clear all data
            self.sessions.clear()
            self.session_times.clear()
            self.temp_session_dirs.clear()
            
            logger.info("Session manager cleanup complete")