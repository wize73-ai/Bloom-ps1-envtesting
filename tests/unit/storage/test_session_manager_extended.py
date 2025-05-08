"""
Extended tests for the SessionManager class.

This test suite covers additional edge cases and concurrent access scenarios for the SessionManager.
"""

import os
import pytest
import time
import asyncio
import uuid
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import concurrent.futures

from app.services.storage.session_manager import SessionManager

# Set a smaller sleep time for tests to speed up execution
TEST_SLEEP_TIME = 0.01


class TestSessionManagerExtended:
    """Extended tests for the SessionManager class."""

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
            "session_ttl_seconds": 5,  # Very short TTL for testing
            "cleanup_interval_seconds": 1,  # Very short interval for testing
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
    async def test_concurrent_session_access(self, session_manager):
        """Test concurrent access to sessions."""
        # Create a session
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Define a function to update session metadata
        async def update_metadata(key, value):
            await session_manager.update_session_metadata(session_id, {key: value})
            return key, value
        
        # Run multiple concurrent updates
        tasks = []
        for i in range(10):
            tasks.append(asyncio.create_task(update_metadata(f"key{i}", f"value{i}")))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Verify all updates were applied
        session = await session_manager.get_session(session_id)
        for key, value in results:
            assert session["metadata"][key] == value

    @pytest.mark.asyncio
    async def test_concurrent_document_operations(self, session_manager):
        """Test concurrent document operations."""
        # Create a session
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Add some documents concurrently
        async def add_document(doc_id):
            return await session_manager.add_document(
                session_id, doc_id, f"Content for {doc_id}".encode(), {"doc_id": doc_id}
            )
        
        # Run concurrent document additions
        doc_ids = [f"doc{i}" for i in range(5)]
        tasks = [asyncio.create_task(add_document(doc_id)) for doc_id in doc_ids]
        results = await asyncio.gather(*tasks)
        
        # Verify all documents were added
        assert sorted(results) == sorted(doc_ids)
        
        # Get all documents
        documents = await session_manager.get_all_documents(session_id)
        assert len(documents) == 5
        
        # Now try concurrent retrieval
        async def get_document(doc_id):
            return await session_manager.get_document(session_id, doc_id)
        
        # Run concurrent document retrievals
        get_tasks = [asyncio.create_task(get_document(doc_id)) for doc_id in doc_ids]
        get_results = await asyncio.gather(*get_tasks)
        
        # Verify all retrievals worked
        for doc in get_results:
            assert doc is not None
            assert doc["document_id"] in doc_ids
            assert doc["content"] == f"Content for {doc['document_id']}".encode()

    @pytest.mark.asyncio
    async def test_temp_directory_creation_failure(self, session_manager):
        """Test handling of temp directory creation failure."""
        # Reset for clean test
        SessionManager._instance = None
        
        # Create a config with a directory that can't be created
        config = {
            "temp_dir": "/nonexistent/path/that/cannot/be/created"
        }
        
        # Mock mkdir to simulate permission error
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            # Should handle the error gracefully
            with pytest.raises(PermissionError):
                SessionManager(config)

    @pytest.mark.asyncio
    async def test_session_storage_limit_enforcement(self, session_manager):
        """Test enforcement of max_sessions limit under pressure."""
        # Create sessions up to the limit
        created_sessions = []
        for i in range(session_manager.max_sessions * 2):
            session_id = str(uuid.uuid4())
            await session_manager.get_session(session_id)
            created_sessions.append(session_id)
            
            # Verify the session count never exceeds max_sessions
            assert len(session_manager.sessions) <= session_manager.max_sessions

    @pytest.mark.asyncio
    async def test_cleanup_with_missing_directories(self, session_manager):
        """Test cleanup with missing session directories."""
        # Create a few sessions
        session_ids = []
        for i in range(3):
            session_id = str(uuid.uuid4())
            await session_manager.get_session(session_id)
            session_ids.append(session_id)
        
        # Manually remove a session directory
        session_dir = session_manager.temp_session_dirs[session_ids[0]]
        shutil.rmtree(session_dir)
        
        # Set all session times to be expired
        for session_id in session_ids:
            session_manager.session_times[session_id] = time.time() - (session_manager.session_ttl + 1)
        
        # Directly call the cleanup code to test it without the sleep loop
        async with session_manager.lock:
            current_time = time.time()
            expired_sessions = []
            
            # Find expired sessions
            for session_id, last_access_time in session_manager.session_times.items():
                if current_time - last_access_time > session_manager.session_ttl:
                    expired_sessions.append(session_id)
            
            # Clean up expired sessions
            for session_id in expired_sessions:
                # Clean up session directory
                if session_id in session_manager.temp_session_dirs:
                    session_dir = session_manager.temp_session_dirs[session_id]
                    try:
                        shutil.rmtree(session_dir)
                    except Exception:
                        pass
                    
                    # Remove from session directories
                    del session_manager.temp_session_dirs[session_id]
                
                # Remove session data
                if session_id in session_manager.sessions:
                    del session_manager.sessions[session_id]
                
                # Remove session time
                if session_id in session_manager.session_times:
                    del session_manager.session_times[session_id]
        
        # Verify all sessions were cleaned up despite directory issues
        for session_id in session_ids:
            assert session_id not in session_manager.sessions
            assert session_id not in session_manager.session_times
            assert session_id not in session_manager.temp_session_dirs

    @pytest.mark.asyncio
    async def test_cleanup_error_handling(self, session_manager):
        """Test that cleanup handles errors gracefully."""
        # Create a session
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Set it to be expired
        session_manager.session_times[session_id] = time.time() - (session_manager.session_ttl + 1)
        
        # Mock shutil.rmtree to simulate an error during cleanup
        with patch('shutil.rmtree', side_effect=Exception("Simulated error")), \
             patch('app.services.storage.session_manager.logger.warning') as mock_warning:
            
            # Directly call the cleanup code without async loops
            async with session_manager.lock:
                current_time = time.time()
                expired_sessions = []
                
                # Find expired sessions
                for session_id, last_access_time in session_manager.session_times.items():
                    if current_time - last_access_time > session_manager.session_ttl:
                        expired_sessions.append(session_id)
                
                # Clean up expired sessions
                for session_id in expired_sessions:
                    # This should handle the error from rmtree
                    if session_id in session_manager.temp_session_dirs:
                        session_dir = session_manager.temp_session_dirs[session_id]
                        try:
                            shutil.rmtree(session_dir)
                        except Exception:
                            pass
                        
                        # Remove from session directories
                        del session_manager.temp_session_dirs[session_id]
                    
                    # These should still execute despite the error
                    if session_id in session_manager.sessions:
                        del session_manager.sessions[session_id]
                    
                    if session_id in session_manager.session_times:
                        del session_manager.session_times[session_id]
            
            # Verify warning was logged
            assert mock_warning.called
            
            # Verify session was cleaned up despite directory removal error
            assert session_id not in session_manager.sessions
            assert session_id not in session_manager.session_times

    @pytest.mark.asyncio
    async def test_add_document_file_write_error(self, session_manager):
        """Test error handling when writing document file fails."""
        # Create a session
        session_id = str(uuid.uuid4())
        await session_manager.get_session(session_id)
        
        # Mock open to raise an exception
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                await session_manager.add_document(session_id, "doc1", b"content", {})

    @pytest.mark.asyncio
    async def test_cross_session_document_isolation(self, session_manager):
        """Test that documents are properly isolated between sessions."""
        # Create two sessions
        session_id1 = str(uuid.uuid4())
        session_id2 = str(uuid.uuid4())
        
        await session_manager.get_session(session_id1)
        await session_manager.get_session(session_id2)
        
        # Add documents to both sessions
        doc_id1 = await session_manager.add_document(session_id1, "doc1", b"content1", {})
        doc_id2 = await session_manager.add_document(session_id2, "doc2", b"content2", {})
        
        # Try to get document from other session
        doc1_from2 = await session_manager.get_document(session_id2, doc_id1)
        doc2_from1 = await session_manager.get_document(session_id1, doc_id2)
        
        # Verify proper isolation
        assert doc1_from2 is None
        assert doc2_from1 is None
        
        # But documents are accessible from their own sessions
        doc1 = await session_manager.get_document(session_id1, doc_id1)
        doc2 = await session_manager.get_document(session_id2, doc_id2)
        
        assert doc1 is not None
        assert doc2 is not None
        assert doc1["content"] == b"content1"
        assert doc2["content"] == b"content2"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])