"""
Tests for the Summarization Pipeline in CasaLingua.

This module tests the SummarizationPipeline class that provides text summarization
capabilities, turning long documents into concise summaries with controlled length and quality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from app.core.pipeline.summarizer import SummarizationPipeline


class TestSummarizationPipeline:
    """Test cases for the SummarizationPipeline class."""

    @pytest.fixture
    def model_manager_mock(self):
        """Create a mock model manager."""
        model_manager = AsyncMock()
        model_manager.load_model = AsyncMock()
        model_manager.run_model = AsyncMock()
        return model_manager

    @pytest.fixture
    def config_mock(self):
        """Create a mock configuration."""
        return {
            "max_length": 150,
            "min_length": 30,
            "temperature": 0.7,
            "model_id": "rag_generator"
        }

    @pytest.fixture
    def registry_config_mock(self):
        """Create a mock registry configuration."""
        return {
            "rag_generator": {
                "model_name": "google/mt5-base",
                "task": "rag_generation"
            }
        }

    @pytest.fixture
    def summarizer(self, model_manager_mock, config_mock, registry_config_mock):
        """Create a SummarizationPipeline instance with mocked dependencies."""
        return SummarizationPipeline(
            model_manager=model_manager_mock,
            config=config_mock,
            registry_config=registry_config_mock
        )

    @pytest.mark.asyncio
    async def test_initialization(self, summarizer, model_manager_mock):
        """Test successful initialization of the summarization pipeline."""
        # Verify initial state
        assert summarizer.initialized is False
        assert summarizer.model_manager is model_manager_mock
        assert summarizer.model_type == "rag_generator"

        # Setup mock return value for load_model
        model_manager_mock.load_model.return_value = {
            "model": MagicMock(),
            "tokenizer": MagicMock()
        }

        # Call initialize method
        await summarizer.initialize()

        # Verify that the model was loaded
        model_manager_mock.load_model.assert_called_once_with("rag_generator")
        assert summarizer.initialized is True

    @pytest.mark.asyncio
    async def test_initialization_error(self, summarizer, model_manager_mock):
        """Test error handling during initialization."""
        # Setup mock to raise exception
        model_manager_mock.load_model.side_effect = Exception("Model loading failed")

        # Call initialize method and verify exception is raised
        with pytest.raises(Exception) as excinfo:
            await summarizer.initialize()

        # Verify exception message
        assert "Model loading failed" in str(excinfo.value)
        assert summarizer.initialized is False

    @pytest.mark.asyncio
    async def test_initialization_empty_model_info(self, summarizer, model_manager_mock):
        """Test initialization with empty model info."""
        # Setup mock to return empty model info
        model_manager_mock.load_model.return_value = {}

        # Call initialize method and verify exception is raised
        with pytest.raises(ValueError) as excinfo:
            await summarizer.initialize()

        # Verify exception message
        assert "Failed to load rag_generator model" in str(excinfo.value)
        assert summarizer.initialized is False

    @pytest.mark.asyncio
    async def test_double_initialization(self, summarizer, model_manager_mock):
        """Test that initializing twice doesn't reload the model."""
        # Setup mock return value for load_model
        model_manager_mock.load_model.return_value = {
            "model": MagicMock(),
            "tokenizer": MagicMock()
        }

        # Call initialize method
        await summarizer.initialize()
        
        # Reset mock to verify it's not called again
        model_manager_mock.load_model.reset_mock()
        
        # Initialize again
        await summarizer.initialize()
        
        # Verify model_manager.load_model wasn't called again
        model_manager_mock.load_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize(self, summarizer, model_manager_mock):
        """Test the summarize method."""
        # Setup model manager mock
        model_manager_mock.load_model.return_value = {
            "model": MagicMock(),
            "tokenizer": MagicMock()
        }
        
        model_manager_mock.run_model.return_value = {
            "result": "This is a summarized text.",
            "metrics": {"tokens": 6, "time": 0.5},
            "metadata": {"model": "mt5"}
        }

        # Test text to summarize
        text = "This is a long text that needs to be summarized. It contains multiple sentences and should be reduced to a shorter version while maintaining the key information."
        
        # Call summarize method
        result = await summarizer.summarize(text, language="en", max_length=100, min_length=30)
        
        # Verify the model manager was called correctly
        model_manager_mock.run_model.assert_called_once()
        
        # Verify the input to run_model
        call_args = model_manager_mock.run_model.call_args[0]
        assert call_args[0] == "rag_generator"  # model_type
        assert call_args[1] == "process"  # method_name
        
        input_data = call_args[2]
        assert input_data["text"] == text
        assert input_data["source_language"] == "en"
        assert input_data["parameters"]["max_length"] == 100
        assert input_data["parameters"]["min_length"] == 30
        
        # Verify the result
        assert result["summary"] == "This is a summarized text."
        assert result["model_used"] == "rag_generator"
        assert result["language"] == "en"
        assert result["metrics"] == {"tokens": 6, "time": 0.5}
        assert result["metadata"] == {"model": "mt5"}

    @pytest.mark.asyncio
    async def test_summarize_with_custom_model_id(self, summarizer, model_manager_mock):
        """Test summarization with a custom model ID."""
        # Setup model manager mock
        model_manager_mock.load_model.return_value = {
            "model": MagicMock(),
            "tokenizer": MagicMock()
        }
        
        model_manager_mock.run_model.return_value = {
            "result": "This is a summarized text with a custom model.",
            "metrics": {"tokens": 9, "time": 0.7},
            "metadata": {"model": "custom_model"}
        }

        # Test text to summarize
        text = "This is a long text that needs to be summarized."
        
        # Call summarize method with custom model_id
        result = await summarizer.summarize(
            text, 
            language="en", 
            max_length=100, 
            min_length=30,
            model_id="custom_summarizer"
        )
        
        # Verify the model manager was called with the custom model ID
        model_manager_mock.run_model.assert_called_once()
        assert model_manager_mock.run_model.call_args[0][0] == "custom_summarizer"
        
        # Verify the result
        assert result["model_used"] == "custom_summarizer"
        assert "This is a summarized text with a custom model." in result["summary"]

    @pytest.mark.asyncio
    async def test_summarize_with_error(self, summarizer, model_manager_mock):
        """Test error handling during summarization."""
        # Setup model manager mock
        model_manager_mock.load_model.return_value = {
            "model": MagicMock(),
            "tokenizer": MagicMock()
        }
        
        # Setup run_model to raise an exception
        model_manager_mock.run_model.side_effect = Exception("Summarization failed")

        # Test text to summarize
        text = "This is a text that will cause an error during summarization."
        
        # Call summarize method
        result = await summarizer.summarize(text, language="en")
        
        # Verify the error is properly captured in the result
        assert "error" in result
        assert "Summarization failed" in result["error"]
        assert result["model_used"] == "rag_generator"
        assert result["language"] == "en"

    @pytest.mark.asyncio
    async def test_summarize_without_initialization(self, summarizer, model_manager_mock):
        """Test that summarize initializes the pipeline if not already initialized."""
        # Setup model manager mock
        model_manager_mock.load_model.return_value = {
            "model": MagicMock(),
            "tokenizer": MagicMock()
        }
        
        model_manager_mock.run_model.return_value = {
            "result": "Auto-initialized summarization.",
            "metrics": {},
            "metadata": {}
        }

        # Verify not initialized
        assert summarizer.initialized is False
        
        # Call summarize without explicit initialization
        await summarizer.summarize("Test text", language="en")
        
        # Verify that load_model was called (initialization happened)
        model_manager_mock.load_model.assert_called_once()
        assert summarizer.initialized is True

    @pytest.mark.asyncio
    async def test_summarize_with_non_dict_result(self, summarizer, model_manager_mock):
        """Test summarization with a non-dictionary result from the model."""
        # Setup model manager mock
        model_manager_mock.load_model.return_value = {
            "model": MagicMock(),
            "tokenizer": MagicMock()
        }
        
        # Return a non-dictionary result (string)
        model_manager_mock.run_model.return_value = "Plain text summary"

        # Call summarize method
        result = await summarizer.summarize("Test text", language="en")
        
        # Verify the result handling
        assert result["summary"] == "Plain text summary"
        assert result["model_used"] == "rag_generator"
        assert result["metrics"] == {}  # should have default empty dict
        assert result["metadata"] == {}  # should have default empty dict
    
    @pytest.mark.asyncio
    async def test_summarize_with_additional_parameters(self, summarizer, model_manager_mock):
        """Test summarization with additional parameters."""
        # Setup model manager mock
        model_manager_mock.load_model.return_value = {
            "model": MagicMock(),
            "tokenizer": MagicMock()
        }
        
        model_manager_mock.run_model.return_value = {
            "result": "Summarized with additional parameters.",
            "metrics": {"tokens": 5},
            "metadata": {"style": "concise"}
        }

        # Additional parameters
        user_id = "user123"
        request_id = "req456"
        additional_param = "extra_value"
        
        # Call summarize with additional parameters
        result = await summarizer.summarize(
            "Test text", 
            language="en",
            user_id=user_id,
            request_id=request_id,
            additional_param=additional_param
        )
        
        # Verify parameters were passed to the model
        input_data = model_manager_mock.run_model.call_args[0][2]
        assert input_data["parameters"]["user_id"] == user_id
        assert input_data["parameters"]["request_id"] == request_id
        assert input_data["parameters"]["additional_param"] == additional_param
        
        # Verify result
        assert result["summary"] == "Summarized with additional parameters."


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])