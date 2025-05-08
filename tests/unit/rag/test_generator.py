"""
Test suite for the RAG Generator component.

Tests generation capabilities including translation and chat using the
retrieval-augmented generation approach.
"""

import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import json
from datetime import datetime

from app.core.rag.generator import AugmentedGenerator, TranslationResult
from app.core.pipeline.tokenizer import TokenizerPipeline


class MockModelManager:
    """Mock model manager for testing"""
    
    def __init__(self):
        self.model = None
        
    def get_model(self, model_id=None, task=None, language=None):
        """Return a mock model."""
        self.model = MockModel()
        return self.model


class MockModel:
    """Mock model for testing"""
    
    def __init__(self):
        self.model_id = "test_model"
        
    async def translate(self, text, source_language, target_language):
        """Mock translation method."""
        # Simple mock translation by adding language tag
        result = MagicMock()
        result.text = f"[{target_language}] {text}"
        result.confidence = 0.85
        result.timestamp = datetime.now().isoformat()
        return result
        
    async def generate(self, prompt):
        """Mock text generation method."""
        # Simple mock response generation
        result = MagicMock()
        result.text = f"This is a mock response to: {prompt[:30]}..."
        return result


@pytest.fixture
def model_registry_mock():
    """Create a mock model registry."""
    with patch("app.services.models.loader.ModelRegistry") as mock:
        registry_instance = mock.return_value
        registry_instance.get_model_and_tokenizer.return_value = (None, "mock_tokenizer")
        yield mock


@pytest.fixture
def tokenizer_mock():
    """Create a mock tokenizer."""
    with patch("app.core.pipeline.tokenizer.TokenizerPipeline") as mock:
        tokenizer_instance = mock.return_value
        tokenizer_instance.encode.return_value = [101, 102, 103]  # Mock token IDs
        yield tokenizer_instance


@pytest.fixture
def generator(model_registry_mock, tokenizer_mock):
    """Create an AugmentedGenerator instance with mocked dependencies."""
    model_manager = MockModelManager()
    generator = AugmentedGenerator(model_manager)
    # Replace real tokenizer with our mock
    generator.tokenizer = tokenizer_mock
    return generator


@pytest.mark.asyncio
async def test_translate_without_context(generator):
    """Test basic translation without context."""
    result = await generator.translate(
        text="Hello world",
        source_language="en",
        target_language="es"
    )
    
    assert isinstance(result, TranslationResult)
    assert "[es]" in result.text
    assert "Hello world" in result.text
    assert result.confidence >= 0.8
    assert result.model_id == "test_model"


@pytest.mark.asyncio
async def test_translate_with_context(generator):
    """Test translation with context documents."""
    reference_docs = [
        MagicMock(content="This is a reference document."),
        MagicMock(content="This provides additional context.")
    ]
    
    result = await generator.translate(
        text="Hello world",
        source_language="en",
        target_language="fr",
        reference_documents=reference_docs
    )
    
    assert isinstance(result, TranslationResult)
    assert "[fr]" in result.text
    # Verify the context was included
    assert "reference document" in result.text or "additional context" in result.text
    assert result.confidence >= 0.8
    assert result.model_id == "test_model"


@pytest.mark.asyncio
async def test_translate_with_invalid_reference_docs(generator):
    """Test translation with malformed reference documents."""
    # Create reference docs without content attribute
    invalid_docs = [
        {"text": "This will cause an error because it lacks content attribute"}
    ]
    
    result = await generator.translate(
        text="Hello world",
        source_language="en",
        target_language="fr",
        reference_documents=invalid_docs
    )
    
    # Should succeed without context since error is handled
    assert isinstance(result, TranslationResult)
    assert "[fr]" in result.text
    assert "Hello world" in result.text


@pytest.mark.asyncio
async def test_translate_with_model_failure(generator):
    """Test handling of model failures during translation."""
    # Make the model translation method fail
    generator.model_manager.get_model = MagicMock(return_value=None)
    
    with pytest.raises(ValueError, match="Translation model could not be loaded"):
        await generator.translate(
            text="Hello world",
            source_language="en",
            target_language="es"
        )


@pytest.mark.asyncio
async def test_generate_chat_response(generator):
    """Test chat response generation."""
    conversation_history = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you. How can I help?"}
    ]
    
    response = await generator.generate_chat_response(
        message="Can you tell me about machine learning?",
        conversation_history=conversation_history,
        language="en"
    )
    
    assert response is not None
    assert isinstance(response, str)
    assert "mock response" in response.lower()


@pytest.mark.asyncio
async def test_generate_chat_with_reference_docs(generator):
    """Test chat response generation with reference documents."""
    conversation_history = [
        {"role": "user", "content": "What can you tell me about CasaLingua?"}
    ]
    
    reference_docs = [
        MagicMock(content="CasaLingua is a multilingual language processing system.")
    ]
    
    response = await generator.generate_chat_response(
        message="How does it work?",
        conversation_history=conversation_history,
        reference_documents=reference_docs,
        language="en"
    )
    
    assert response is not None
    assert isinstance(response, str)
    assert "mock response" in response.lower()


@pytest.mark.asyncio
async def test_generate_chat_with_model_failure(generator):
    """Test handling of model failures during chat generation."""
    conversation_history = [
        {"role": "user", "content": "Hello"}
    ]
    
    # Make the model fail
    generator.model_manager.get_model = MagicMock(return_value=None)
    
    with pytest.raises(ValueError, match="Chat model could not be loaded"):
        await generator.generate_chat_response(
            message="Hello",
            conversation_history=conversation_history,
            language="en"
        )


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])