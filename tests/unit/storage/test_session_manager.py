"""
Tests for the SessionManager class.

This test suite covers the SessionManager functionality for handling user sessions
and temporary document storage.
"""

import os
import pytest
import time
import asyncio
import uuid
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from app.services.storage.session_manager import SessionManager


class TestSessionManager:
    """Tests for the SessionManager class."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for session storage."""
        session_dir = tmp_path / "sessions"
        session_dir.mkdir(exist_ok=True)
        yield session_dir
        
        # Clean up
        if session_dir.exists():
            shutil.rmtree(session_dir)

    @pytest.fixture
    def session_manager(self, temp_dir):
        """Create a session manager with custom configuration."""
        # Reset the singleton instance
        SessionManager._instance = None

        # Create a new SessionManager with test config
        config = {
            "session_ttl_seconds": 60,  # Short TTL for testing
            "cleanup_interval_seconds": 10,
            "max_sessions": 5,
            "temp_dir": str(temp_dir)
        }
        manager = SessionManager(config)
        
        # Return the manager
        yield manager
        
        # Clean up task if needed
        if manager.cleanup_task and not manager.cleanup_task.done():
            manager.cleanup_task.cancel()

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test that SessionManager uses the singleton pattern."""
        # Create two instances
        manager1 = SessionManager({"temp_dir": "temp1"})
        manager2 = SessionManager({"temp_dir": "temp2"})
        
        # Verify they are the same instance
        assert manager1 is manager2
        
        # Verify config doesn't change once initialized
        assert manager1.temp_dir == manager2.temp_dir
        assert "temp1" in str(manager1.temp_dir)
        
        # Reset for other tests
        SessionManager._instance = None

    @pytest.mark.asyncio
    async def test_get_session_new(self, session_manager):
        """Test creating a new session."""
        # Generate a valid session ID
        session_id = str(uuid.uuid4())
        
        # Get or create session
        session = await session_manager.get_session(session_id)
        
        # Verify session was created
        assert session["session_id"] == session_id
        assert "created_at" in session
        assert "documents" in session
        assert "metadata" in session
        assert session_id in session_manager.sessions
        assert session_id in session_manager.session_times
        assert session_id in session_manager.temp_session_dirs
        
        # Verify temp directory was created
        session_dir = session_manager.temp_session_dirs[session_id]
        assert session_dir.exists()

    @pytest.mark.asyncio
    async def test_get_session_existing(self, session_manager):
        """Test getting an existing session."""
        # Create a session
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Record the initial access time
        initial_time = session_manager.session_times[session_id]
        
        # Wait a moment to ensure time difference
        await asyncio.sleep(0.1)
        
        # Get the same session again
        session = await session_manager.get_session(session_id)
        
        # Verify session was retrieved and time was updated
        assert session["session_id"] == session_id
        assert session_manager.session_times[session_id] > initial_time

    @pytest.mark.asyncio
    async def test_get_session_invalid_id(self, session_manager):
        """Test getting a session with an invalid ID format."""
        # Use an invalid session ID
        invalid_id = "not-a-uuid"
        
        # Get or create session
        session = await session_manager.get_session(invalid_id)
        
        # Verify a new session with valid UUID was created
        assert session["session_id"] != invalid_id
        assert "created_at" in session
        
        # Verify the ID is a valid UUID
        try:
            uuid.UUID(session["session_id"])
            is_valid = True
        except ValueError:
            is_valid = False
        
        assert is_valid

    @pytest.mark.asyncio
    async def test_get_session_not_creating(self, session_manager):
        """Test getting a non-existent session without creating it."""
        # Use a non-existent session ID
        session_id = str(uuid.uuid4())
        
        # Get session without creating
        session = await session_manager.get_session(session_id, create_if_missing=False)
        
        # Verify empty dictionary is returned
        assert session == {}
        assert session_id not in session_manager.sessions

    @pytest.mark.asyncio
    async def test_add_document(self, session_manager):
        """Test adding a document to a session."""
        # Create a session
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Add a document
        content = b"Test document content"
        metadata = {"filename": "test.txt", "content_type": "text/plain"}
        document_id = await session_manager.add_document(session_id, "doc1", content, metadata)
        
        # Verify document was added
        assert document_id == "doc1"
        
        # Check session data
        session = session_manager.sessions[session_id]
        assert document_id in session["documents"]
        assert session["documents"][document_id]["metadata"] == metadata
        assert session["documents"][document_id]["size"] == len(content)
        
        # Check document file
        document_path = Path(session["documents"][document_id]["file_path"])
        assert document_path.exists()
        with open(document_path, 'rb') as f:
            saved_content = f.read()
        assert saved_content == content

    @pytest.mark.asyncio
    async def test_add_document_auto_id(self, session_manager):
        """Test adding a document with auto-generated ID."""
        # Create a session
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Add a document with no ID
        content = b"Test document content"
        metadata = {"filename": "test.txt"}
        document_id = await session_manager.add_document(session_id, None, content, metadata)
        
        # Verify document was added with a generated ID
        assert document_id is not None
        assert document_id in session_manager.sessions[session_id]["documents"]
        
        # Verify the ID is a valid UUID
        try:
            uuid.UUID(document_id)
            is_valid = True
        except ValueError:
            is_valid = False
        
        assert is_valid

    @pytest.mark.asyncio
    async def test_get_document(self, session_manager):
        """Test getting a document from a session."""
        # Create a session and add a document
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        content = b"Test document content"
        metadata = {"filename": "test.txt"}
        document_id = await session_manager.add_document(session_id, "doc1", content, metadata)
        
        # Get the document
        document = await session_manager.get_document(session_id, document_id)
        
        # Verify document data
        assert document["document_id"] == document_id
        assert document["content"] == content
        assert document["metadata"] == metadata
        assert document["size"] == len(content)

    @pytest.mark.asyncio
    async def test_get_document_nonexistent_session(self, session_manager):
        """Test getting a document from a non-existent session."""
        # Use a non-existent session ID
        session_id = str(uuid.uuid4())
        
        # Try to get a document
        document = await session_manager.get_document(session_id, "doc1")
        
        # Verify None is returned
        assert document is None

    @pytest.mark.asyncio
    async def test_get_document_nonexistent_document(self, session_manager):
        """Test getting a non-existent document from a session."""
        # Create a session
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Try to get a non-existent document
        document = await session_manager.get_document(session_id, "nonexistent")
        
        # Verify None is returned
        assert document is None

    @pytest.mark.asyncio
    async def test_get_document_file_error(self, session_manager):
        """Test getting a document when the file is missing."""
        # Create a session and add a document
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        content = b"Test document content"
        metadata = {"filename": "test.txt"}
        document_id = await session_manager.add_document(session_id, "doc1", content, metadata)
        
        # Remove the document file but keep the metadata
        document_path = Path(session_manager.sessions[session_id]["documents"][document_id]["file_path"])
        os.remove(document_path)
        
        # Try to get the document
        document = await session_manager.get_document(session_id, document_id)
        
        # Verify None is returned due to file error
        assert document is None

    @pytest.mark.asyncio
    async def test_get_all_documents(self, session_manager):
        """Test getting all documents in a session."""
        # Create a session and add documents
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Add multiple documents
        await session_manager.add_document(session_id, "doc1", b"Content 1", {"filename": "file1.txt"})
        await session_manager.add_document(session_id, "doc2", b"Content 2", {"filename": "file2.txt"})
        
        # Get all documents
        documents = await session_manager.get_all_documents(session_id)
        
        # Verify documents
        assert len(documents) == 2
        doc_ids = [doc["document_id"] for doc in documents]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        
        # Verify no content is included
        for doc in documents:
            assert "content" not in doc

    @pytest.mark.asyncio
    async def test_get_all_documents_nonexistent_session(self, session_manager):
        """Test getting all documents from a non-existent session."""
        # Use a non-existent session ID
        session_id = str(uuid.uuid4())
        
        # Get all documents
        documents = await session_manager.get_all_documents(session_id)
        
        # Verify empty list is returned
        assert documents == []

    @pytest.mark.asyncio
    async def test_remove_document(self, session_manager):
        """Test removing a document from a session."""
        # Create a session and add a document
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        content = b"Test document content"
        metadata = {"filename": "test.txt"}
        document_id = await session_manager.add_document(session_id, "doc1", content, metadata)
        
        # Get the document file path
        document_path = Path(session_manager.sessions[session_id]["documents"][document_id]["file_path"])
        
        # Remove the document
        result = await session_manager.remove_document(session_id, document_id)
        
        # Verify document was removed
        assert result is True
        assert document_id not in session_manager.sessions[session_id]["documents"]
        assert not document_path.exists()

    @pytest.mark.asyncio
    async def test_remove_document_nonexistent_session(self, session_manager):
        """Test removing a document from a non-existent session."""
        # Use a non-existent session ID
        session_id = str(uuid.uuid4())
        
        # Try to remove a document
        result = await session_manager.remove_document(session_id, "doc1")
        
        # Verify removal failed
        assert result is False

    @pytest.mark.asyncio
    async def test_remove_document_nonexistent_document(self, session_manager):
        """Test removing a non-existent document from a session."""
        # Create a session
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Try to remove a non-existent document
        result = await session_manager.remove_document(session_id, "nonexistent")
        
        # Verify removal failed
        assert result is False

    @pytest.mark.asyncio
    async def test_remove_document_file_error(self, session_manager):
        """Test removing a document when the file is already missing."""
        # Create a session and add a document
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        content = b"Test document content"
        metadata = {"filename": "test.txt"}
        document_id = await session_manager.add_document(session_id, "doc1", content, metadata)
        
        # Remove the document file but keep the metadata
        document_path = Path(session_manager.sessions[session_id]["documents"][document_id]["file_path"])
        os.remove(document_path)
        
        # Try to remove the document
        with patch('app.services.storage.session_manager.logger.warning') as mock_warning:
            result = await session_manager.remove_document(session_id, document_id)
        
        # Verify removal succeeded despite file error
        assert result is True
        assert document_id not in session_manager.sessions[session_id]["documents"]
        assert mock_warning.called

    @pytest.mark.asyncio
    async def test_update_session_metadata(self, session_manager):
        """Test updating session metadata."""
        # Create a session
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Update metadata
        metadata = {"user_id": "123", "language": "en"}
        result = await session_manager.update_session_metadata(session_id, metadata)
        
        # Verify metadata was updated
        assert result is True
        assert session_manager.sessions[session_id]["metadata"] == metadata
        
        # Update with additional metadata
        more_metadata = {"theme": "dark"}
        result = await session_manager.update_session_metadata(session_id, more_metadata)
        
        # Verify metadata was merged
        assert result is True
        assert session_manager.sessions[session_id]["metadata"]["user_id"] == "123"
        assert session_manager.sessions[session_id]["metadata"]["language"] == "en"
        assert session_manager.sessions[session_id]["metadata"]["theme"] == "dark"

    @pytest.mark.asyncio
    async def test_update_session_metadata_nonexistent_session(self, session_manager):
        """Test updating metadata for a non-existent session."""
        # Use a non-existent session ID
        session_id = str(uuid.uuid4())
        
        # Try to update metadata
        result = await session_manager.update_session_metadata(session_id, {"key": "value"})
        
        # Verify update failed
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_session(self, session_manager):
        """Test deleting a session."""
        # Create a session and add a document
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        await session_manager.add_document(session_id, "doc1", b"content", {})
        
        # Get the session directory
        session_dir = session_manager.temp_session_dirs[session_id]
        
        # Delete the session
        result = await session_manager.delete_session(session_id)
        
        # Verify session was deleted
        assert result is True
        assert session_id not in session_manager.sessions
        assert session_id not in session_manager.session_times
        assert session_id not in session_manager.temp_session_dirs
        assert not session_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_session_nonexistent(self, session_manager):
        """Test deleting a non-existent session."""
        # Use a non-existent session ID
        session_id = str(uuid.uuid4())
        
        # Try to delete the session
        result = await session_manager.delete_session(session_id)
        
        # Verify deletion failed
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_session_directory_error(self, session_manager):
        """Test deleting a session when the directory removal fails."""
        # Create a session
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Mock shutil.rmtree to raise an exception
        with patch('shutil.rmtree', side_effect=Exception("Test error")), \
             patch('app.services.storage.session_manager.logger.warning') as mock_warning:
            
            # Delete the session
            result = await session_manager.delete_session(session_id)
            
            # Verify session was still deleted from memory despite directory error
            assert result is True
            assert session_id not in session_manager.sessions
            assert session_id not in session_manager.session_times
            assert session_id not in session_manager.temp_session_dirs
            assert mock_warning.called

    @pytest.mark.asyncio
    async def test_evict_oldest_session(self, session_manager):
        """Test evicting the oldest session when max sessions is reached."""
        # Create max_sessions sessions with incremental timestamps
        for i in range(session_manager.max_sessions):
            session_id = str(uuid.uuid4())
            await session_manager.get_session(session_id)
            # Set different times to ensure order
            session_manager.session_times[session_id] = time.time() - (session_manager.max_sessions - i)
        
        # Record the sessions before eviction
        original_sessions = set(session_manager.sessions.keys())
        
        # Add one more session to trigger eviction
        new_session_id = str(uuid.uuid4())
        await session_manager.get_session(new_session_id)
        
        # Verify total session count is still at max
        assert len(session_manager.sessions) == session_manager.max_sessions
        
        # Verify the new session was added
        assert new_session_id in session_manager.sessions
        
        # Verify the oldest session was evicted
        current_sessions = set(session_manager.sessions.keys())
        evicted_sessions = original_sessions - current_sessions
        assert len(evicted_sessions) == 1

    @pytest.mark.asyncio
    async def test_cleanup_task(self, session_manager):
        """Test the cleanup task for expired sessions."""
        # Create a session
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Add a document
        await session_manager.add_document(session_id, "doc1", b"content", {})
        
        # Get the session directory
        session_dir = session_manager.temp_session_dirs[session_id]
        
        # Set access time to ensure expiration
        session_manager.session_times[session_id] = time.time() - (session_manager.session_ttl + 10)
        
        # Run the cleanup task directly (just once)
        with patch('asyncio.sleep', AsyncMock()):
            cleanup_task = asyncio.create_task(session_manager._cleanup_task())
            
            # Give it a moment to run
            await asyncio.sleep(0.1)
            
            # Cancel the task
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Verify session was cleaned up
        assert session_id not in session_manager.sessions
        assert session_id not in session_manager.session_times
        assert session_id not in session_manager.temp_session_dirs
        assert not session_dir.exists()

    @pytest.mark.asyncio
    async def test_start_cleanup_task(self, session_manager):
        """Test starting the cleanup task."""
        # Ensure no task is running
        session_manager.cleanup_task = None
        
        # Start the cleanup task
        await session_manager.start_cleanup_task()
        
        # Verify task was created
        assert session_manager.cleanup_task is not None
        assert not session_manager.cleanup_task.done()
        
        # Cancel the task
        session_manager.cleanup_task.cancel()
        try:
            await session_manager.cleanup_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_cleanup(self, session_manager):
        """Test cleaning up the session manager."""
        # Create a session
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Add a document
        await session_manager.add_document(session_id, "doc1", b"content", {})
        
        # Start the cleanup task
        await session_manager.start_cleanup_task()
        
        # Clean up the session manager
        await session_manager.cleanup()
        
        # Verify cleanup task was cancelled
        assert session_manager.cleanup_task.done()
        
        # Verify all data was cleared
        assert len(session_manager.sessions) == 0
        assert len(session_manager.session_times) == 0
        assert len(session_manager.temp_session_dirs) == 0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])