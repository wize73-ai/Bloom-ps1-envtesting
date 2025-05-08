"""
Tests for the ConversationMemory and ConversationMemoryManager classes.

This test suite covers the memory management components in the RAG system,
ensuring proper conversation history tracking and session management.
"""

import os
import time
import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open, ANY

from app.core.rag.memory import ConversationMemory, ConversationMemoryManager


class TestConversationMemory:
    """Tests for the ConversationMemory class."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        mock = MagicMock()
        mock.encode.return_value = list(range(10))  # Mock token IDs
        return mock

    @pytest.fixture
    def memory_with_tokenizer(self, mock_tokenizer):
        """Create a ConversationMemory instance with a mock tokenizer."""
        with patch('app.core.rag.memory.TokenizerPipeline') as mock_pipeline, \
             patch('app.core.rag.memory.ModelRegistry') as mock_registry:
            
            # Configure mocks
            mock_registry_instance = MagicMock()
            mock_registry_instance.get_model_and_tokenizer.return_value = (None, "test-tokenizer")
            mock_registry.return_value = mock_registry_instance
            
            mock_pipeline.return_value = mock_tokenizer
            
            memory = ConversationMemory(max_session_messages=5, max_token_limit=100)
            
            # Verify proper initialization
            assert memory.tokenizer is not None
            return memory

    @pytest.fixture
    def memory_without_tokenizer(self):
        """Create a ConversationMemory instance without a tokenizer."""
        with patch('app.core.rag.memory.TokenizerPipeline') as mock_pipeline, \
             patch('app.core.rag.memory.ModelRegistry') as mock_registry:
            
            # Make registry raise an exception
            mock_registry_instance = MagicMock()
            mock_registry_instance.get_model_and_tokenizer.side_effect = Exception("Registry error")
            mock_registry.return_value = mock_registry_instance
            
            memory = ConversationMemory(max_session_messages=5, max_token_limit=100)
            
            # Verify tokenizer is None
            assert memory.tokenizer is None
            return memory

    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary directory for storage."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    @pytest.fixture
    def memory_with_storage(self, temp_storage_dir, mock_tokenizer):
        """Create a ConversationMemory instance with storage."""
        with patch('app.core.rag.memory.TokenizerPipeline') as mock_pipeline, \
             patch('app.core.rag.memory.ModelRegistry') as mock_registry:
            
            # Configure mocks
            mock_registry_instance = MagicMock()
            mock_registry_instance.get_model_and_tokenizer.return_value = (None, "test-tokenizer")
            mock_registry.return_value = mock_registry_instance
            
            mock_pipeline.return_value = mock_tokenizer
            
            memory = ConversationMemory(
                max_session_messages=5,
                max_token_limit=100,
                storage_dir=temp_storage_dir
            )
            
            return memory

    def test_init(self, memory_with_tokenizer):
        """Test initialization."""
        memory = memory_with_tokenizer
        
        assert memory.max_session_messages == 5
        assert memory.max_token_limit == 100
        assert memory.session_ttl_hours == 24
        assert memory.sessions == {}
        assert memory.tokenizer is not None

    def test_init_with_storage(self, memory_with_storage, temp_storage_dir):
        """Test initialization with storage."""
        memory = memory_with_storage
        
        assert memory.storage_dir == Path(temp_storage_dir)
        assert memory.storage_dir.exists()

    def test_add_message_new_session(self, memory_with_tokenizer):
        """Test adding a message to a new session."""
        memory = memory_with_tokenizer
        
        # Add a message
        memory.add_message("test_session", "user", "Hello")
        
        # Verify session was created
        assert "test_session" in memory.sessions
        assert len(memory.sessions["test_session"]["messages"]) == 1
        assert memory.sessions["test_session"]["messages"][0]["role"] == "user"
        assert memory.sessions["test_session"]["messages"][0]["content"] == "Hello"
        assert "timestamp" in memory.sessions["test_session"]["messages"][0]
        assert "token_count" in memory.sessions["test_session"]["messages"][0]
        assert memory.sessions["test_session"]["total_tokens"] > 0

    def test_add_message_with_metadata(self, memory_with_tokenizer):
        """Test adding a message with metadata."""
        memory = memory_with_tokenizer
        
        # Add a message with metadata
        metadata = {"language": "en", "source": "user_input"}
        memory.add_message("test_session", "user", "Hello", metadata)
        
        # Verify metadata was included
        assert memory.sessions["test_session"]["messages"][0]["metadata"] == metadata

    def test_add_message_without_tokenizer(self, memory_without_tokenizer):
        """Test adding a message without a tokenizer."""
        memory = memory_without_tokenizer
        
        # Add a message
        memory.add_message("test_session", "user", "Hello")
        
        # Verify message was added without token count
        assert "test_session" in memory.sessions
        assert len(memory.sessions["test_session"]["messages"]) == 1
        assert "token_count" not in memory.sessions["test_session"]["messages"][0]
        assert memory.sessions["test_session"]["total_tokens"] == 0

    def test_get_history_existing_session(self, memory_with_tokenizer):
        """Test getting history from an existing session."""
        memory = memory_with_tokenizer
        
        # Add messages
        memory.add_message("test_session", "user", "Hello")
        memory.add_message("test_session", "assistant", "Hi there!")
        
        # Get history
        history = memory.get_history("test_session")
        
        # Verify history
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there!"

    def test_get_history_nonexistent_session(self, memory_with_tokenizer):
        """Test getting history from a non-existent session."""
        memory = memory_with_tokenizer
        
        # Patch the _load_session method to return False
        with patch.object(memory, '_load_session', return_value=False):
            history = memory.get_history("nonexistent_session")
            
            # Verify empty history
            assert history == []

    def test_get_history_loads_from_storage(self, memory_with_storage):
        """Test that get_history loads a session from storage if it exists."""
        memory = memory_with_storage
        
        # Patch _load_session and mock the session data
        with patch.object(memory, '_load_session', return_value=True) as mock_load:
            # We need to set up the session data after _load_session is called
            def side_effect(session_id):
                memory.sessions[session_id] = {
                    "messages": [],
                    "created_at": datetime.now(),
                    "last_modified": datetime.now(),
                    "last_accessed": datetime.now(),
                    "total_tokens": 0
                }
                return True
            
            mock_load.side_effect = side_effect
            
            memory.get_history("stored_session")
            
            # Verify _load_session was called
            mock_load.assert_called_once_with("stored_session")

    def test_get_last_message(self, memory_with_tokenizer):
        """Test getting the last message."""
        memory = memory_with_tokenizer
        
        # Add messages
        memory.add_message("test_session", "user", "Hello")
        memory.add_message("test_session", "assistant", "Hi there!")
        memory.add_message("test_session", "user", "How are you?")
        
        # Get last message
        last_message = memory.get_last_message("test_session")
        
        # Verify last message
        assert last_message["role"] == "user"
        assert last_message["content"] == "How are you?"

    def test_get_last_message_by_role(self, memory_with_tokenizer):
        """Test getting the last message filtered by role."""
        memory = memory_with_tokenizer
        
        # Add messages
        memory.add_message("test_session", "user", "Hello")
        memory.add_message("test_session", "assistant", "Hi there!")
        memory.add_message("test_session", "user", "How are you?")
        
        # Get last assistant message
        last_assistant_message = memory.get_last_message("test_session", role="assistant")
        
        # Verify message
        assert last_assistant_message["role"] == "assistant"
        assert last_assistant_message["content"] == "Hi there!"

    def test_get_last_message_empty_history(self, memory_with_tokenizer):
        """Test getting the last message with empty history."""
        memory = memory_with_tokenizer
        
        # Get last message from non-existent session
        with patch.object(memory, 'get_history', return_value=[]):
            last_message = memory.get_last_message("nonexistent_session")
            
            # Verify None is returned
            assert last_message is None

    def test_get_last_message_no_matching_role(self, memory_with_tokenizer):
        """Test getting the last message with no matching role."""
        memory = memory_with_tokenizer
        
        # Add message
        memory.add_message("test_session", "user", "Hello")
        
        # Get last system message (none exist)
        last_system_message = memory.get_last_message("test_session", role="system")
        
        # Verify None is returned
        assert last_system_message is None

    def test_clear_history(self, memory_with_tokenizer):
        """Test clearing session history."""
        memory = memory_with_tokenizer
        
        # Add message
        memory.add_message("test_session", "user", "Hello")
        
        # Verify session exists
        assert "test_session" in memory.sessions
        
        # Clear history
        memory.clear_history("test_session")
        
        # Verify session was removed
        assert "test_session" not in memory.sessions

    def test_clear_history_with_storage(self, memory_with_storage):
        """Test clearing session history with storage."""
        memory = memory_with_storage
        
        # Add message
        memory.add_message("test_session", "user", "Hello")
        
        # Mock storage path exists
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'unlink') as mock_unlink:
            
            # Clear history
            memory.clear_history("test_session")
            
            # Verify unlink was called
            mock_unlink.assert_called_once()

    def test_get_session_summary_existing(self, memory_with_tokenizer):
        """Test getting summary for an existing session."""
        memory = memory_with_tokenizer
        
        # Add messages
        memory.add_message("test_session", "user", "Hello")
        memory.add_message("test_session", "assistant", "Hi there!")
        
        # Get summary
        summary = memory.get_session_summary("test_session")
        
        # Verify summary
        assert summary["exists"] is True
        assert summary["message_count"] == 2
        assert summary["user_message_count"] == 1
        assert summary["assistant_message_count"] == 1
        assert "created_at" in summary
        assert "last_modified" in summary
        assert "last_accessed" in summary

    def test_get_session_summary_nonexistent(self, memory_with_tokenizer):
        """Test getting summary for a non-existent session."""
        memory = memory_with_tokenizer
        
        # Patch _load_session to return False
        with patch.object(memory, '_load_session', return_value=False):
            summary = memory.get_session_summary("nonexistent_session")
            
            # Verify summary indicates non-existence
            assert summary["exists"] is False

    def test_get_all_sessions(self, memory_with_tokenizer):
        """Test getting all session IDs."""
        memory = memory_with_tokenizer
        
        # Add messages to multiple sessions
        memory.add_message("session1", "user", "Hello")
        memory.add_message("session2", "user", "Hi")
        memory.add_message("session3", "user", "Hey")
        
        # Get all sessions
        sessions = memory.get_all_sessions()
        
        # Verify all sessions are returned
        assert len(sessions) == 3
        assert "session1" in sessions
        assert "session2" in sessions
        assert "session3" in sessions

    def test_cleanup_expired_sessions(self, memory_with_tokenizer):
        """Test cleaning up expired sessions."""
        memory = memory_with_tokenizer
        
        # Add messages to sessions
        memory.add_message("active_session", "user", "Hello")
        memory.add_message("expired_session", "user", "Hi")
        
        # Make expired_session appear old
        now = datetime.now()
        memory.sessions["expired_session"]["last_accessed"] = now - timedelta(hours=memory.session_ttl_hours + 1)
        
        # Run cleanup
        removed_count = memory.cleanup_expired_sessions()
        
        # Verify cleanup
        assert removed_count == 1
        assert "active_session" in memory.sessions
        assert "expired_session" not in memory.sessions

    def test_trim_history_by_max_messages(self, memory_with_tokenizer):
        """Test trimming history by maximum message count."""
        memory = memory_with_tokenizer
        memory.max_session_messages = 3
        
        # Add more messages than the limit
        for i in range(5):
            memory.add_message("test_session", "user", f"Message {i}")
        
        # Verify trimming
        assert len(memory.sessions["test_session"]["messages"]) == 3
        assert memory.sessions["test_session"]["messages"][0]["content"] == "Message 2"
        assert memory.sessions["test_session"]["messages"][1]["content"] == "Message 3"
        assert memory.sessions["test_session"]["messages"][2]["content"] == "Message 4"

    def test_trim_history_by_token_limit(self, memory_with_tokenizer, mock_tokenizer):
        """Test trimming history by token limit."""
        memory = memory_with_tokenizer
        memory.max_token_limit = 25  # 10 tokens per message * 3 = 30, so we'll keep only 2 messages
        
        # Add multiple messages
        for i in range(5):
            memory.add_message("test_session", "user", f"Message {i}")
        
        # Verify trimming
        assert len(memory.sessions["test_session"]["messages"]) <= 3  # Should be 2 or 3 depending on exact token count
        # The newest messages should be kept
        assert memory.sessions["test_session"]["messages"][-1]["content"] == "Message 4"

    def test_save_session(self, memory_with_storage, temp_storage_dir):
        """Test saving a session to storage."""
        memory = memory_with_storage
        
        # Add message
        memory.add_message("test_session", "user", "Hello")
        
        # Call _save_session directly
        with patch('builtins.open', mock_open()) as mock_file:
            result = memory._save_session("test_session")
            
            # Verify file was opened for writing
            mock_file.assert_called_once_with(Path(temp_storage_dir) / "test_session.json", 'w', encoding='utf-8')
            
            # Verify json.dump was called
            handle = mock_file()
            handle.write.assert_called()
            
            # Verify result
            assert result is True

    def test_save_session_no_storage(self, memory_with_tokenizer):
        """Test saving a session without storage configured."""
        memory = memory_with_tokenizer
        
        # Add message
        memory.add_message("test_session", "user", "Hello")
        
        # Call _save_session
        result = memory._save_session("test_session")
        
        # Verify result
        assert result is False

    def test_save_session_error(self, memory_with_storage):
        """Test handling errors when saving a session."""
        memory = memory_with_storage
        
        # Add message
        memory.add_message("test_session", "user", "Hello")
        
        # Mock open to raise exception
        with patch('builtins.open', side_effect=Exception("Test error")):
            result = memory._save_session("test_session")
            
            # Verify result
            assert result is False

    def test_load_session(self, memory_with_storage, temp_storage_dir):
        """Test loading a session from storage."""
        memory = memory_with_storage
        
        # Create a mock session data
        session_data = {
            "messages": [
                {"role": "user", "content": "Hello", "timestamp": datetime.now().isoformat()}
            ],
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "total_tokens": 10
        }
        
        # Mock Path.exists to return True
        with patch.object(Path, 'exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(session_data))):
            
            # Load session
            result = memory._load_session("test_session")
            
            # Verify result
            assert result is True
            assert "test_session" in memory.sessions
            assert memory.sessions["test_session"]["messages"][0]["content"] == "Hello"

    def test_load_session_no_storage(self, memory_with_tokenizer):
        """Test loading a session without storage configured."""
        memory = memory_with_tokenizer
        
        # Load session
        result = memory._load_session("test_session")
        
        # Verify result
        assert result is False

    def test_load_session_not_exists(self, memory_with_storage):
        """Test loading a non-existent session."""
        memory = memory_with_storage
        
        # Mock Path.exists to return False
        with patch.object(Path, 'exists', return_value=False):
            result = memory._load_session("nonexistent_session")
            
            # Verify result
            assert result is False

    def test_load_session_error(self, memory_with_storage):
        """Test handling errors when loading a session."""
        memory = memory_with_storage
        
        # Mock Path.exists to return True but open to raise exception
        with patch.object(Path, 'exists', return_value=True), \
             patch('builtins.open', side_effect=Exception("Test error")):
            
            result = memory._load_session("test_session")
            
            # Verify result
            assert result is False

    def test_get_session_context(self, memory_with_tokenizer):
        """Test getting formatted session context."""
        memory = memory_with_tokenizer
        
        # Add messages
        memory.add_message("test_session", "user", "Hello")
        memory.add_message("test_session", "assistant", "Hi there!")
        memory.add_message("test_session", "user", "How are you?")
        
        # Get context
        context = memory.get_session_context("test_session")
        
        # Verify context format
        assert "USER: Hello" in context
        assert "ASSISTANT: Hi there!" in context
        assert "USER: How are you?" in context

    def test_get_session_context_with_role_filter(self, memory_with_tokenizer):
        """Test getting context filtered by role."""
        memory = memory_with_tokenizer
        
        # Add messages
        memory.add_message("test_session", "user", "Hello")
        memory.add_message("test_session", "assistant", "Hi there!")
        memory.add_message("test_session", "system", "Language: English")
        
        # Get context with only user and assistant roles
        context = memory.get_session_context(
            "test_session", 
            include_roles=["user", "assistant"]
        )
        
        # Verify context
        assert "USER: Hello" in context
        assert "ASSISTANT: Hi there!" in context
        assert "SYSTEM: Language: English" not in context

    def test_get_session_context_with_token_limit(self, memory_with_tokenizer):
        """Test getting context with token limit."""
        memory = memory_with_tokenizer
        
        # Add messages (each with 10 tokens)
        for i in range(5):
            memory.add_message("test_session", "user", f"Message {i}")
        
        # Get context with token limit that will include only 2 messages
        context = memory.get_session_context("test_session", max_context_tokens=20)
        
        # Verify context contains only the most recent messages
        assert "USER: Message 3" in context
        assert "USER: Message 4" in context
        assert "USER: Message 0" not in context

    def test_get_session_context_empty_history(self, memory_with_tokenizer):
        """Test getting context with empty history."""
        memory = memory_with_tokenizer
        
        # Get context for non-existent session
        with patch.object(memory, 'get_history', return_value=[]):
            context = memory.get_session_context("nonexistent_session")
            
            # Verify empty context
            assert context == ""

    def test_repr(self, memory_with_tokenizer):
        """Test the __repr__ method."""
        memory = memory_with_tokenizer
        
        # Add some sessions
        memory.add_message("session1", "user", "Hello")
        memory.add_message("session2", "user", "Hi")
        
        # Check __repr__
        assert repr(memory) == "<ConversationMemory sessions=2>"


class TestConversationMemoryManager:
    """Tests for the ConversationMemoryManager class."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock ConversationMemory."""
        return MagicMock(spec=ConversationMemory)

    @pytest.fixture
    def manager(self, mock_memory):
        """Create a ConversationMemoryManager with a mock memory."""
        return ConversationMemoryManager(memory=mock_memory, cleanup_interval_hours=1)

    def test_init(self, manager, mock_memory):
        """Test initialization."""
        assert manager.memory == mock_memory
        assert manager.cleanup_interval_hours == 1
        assert manager.last_cleanup is not None

    def test_get_memory(self, manager, mock_memory):
        """Test getting the memory instance."""
        memory = manager.get_memory()
        assert memory == mock_memory

    def test_check_cleanup_not_needed(self, manager, mock_memory):
        """Test _check_cleanup when cleanup is not needed."""
        # Set last_cleanup to very recent
        manager.last_cleanup = datetime.now()
        
        # Call _check_cleanup
        manager._check_cleanup()
        
        # Verify cleanup was not called
        mock_memory.cleanup_expired_sessions.assert_not_called()

    def test_check_cleanup_needed(self, manager, mock_memory):
        """Test _check_cleanup when cleanup is needed."""
        # Set last_cleanup to be before cleanup interval
        manager.last_cleanup = datetime.now() - timedelta(hours=2)
        
        # Call _check_cleanup
        manager._check_cleanup()
        
        # Verify cleanup was called
        mock_memory.cleanup_expired_sessions.assert_called_once()
        
        # Verify last_cleanup was updated
        assert (datetime.now() - manager.last_cleanup).total_seconds() < 5  # Within 5 seconds


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])