"""
Tests for the indexer module within the RAG system.
"""

import os
import tempfile
import json
import pytest
import pickle
import numpy as np
from unittest.mock import patch, MagicMock, Mock, AsyncMock
from pathlib import Path
from datetime import datetime

from app.core.rag.indexer import (
    DocumentProcessor,
    TextProcessor,
    DocxProcessor,
    CSVProcessor,
    Indexer
)

# Test text content
TEST_TEXT = """This is a test document for the indexer module.
It contains multiple sentences and paragraphs.

This is the second paragraph with more text.
The indexer should break this into appropriate chunks."""

# Test text in multiple languages
TEST_MULTILINGUAL = {
    "en": "This is English text with some common words like the and of and to.",
    "es": "Este es un texto en español con palabras comunes como el y la y de.",
    "fr": "Ceci est un texte en français avec des mots courants comme le et la et de.",
    "de": "Dies ist ein deutscher Text mit einigen häufigen Wörtern wie der die das und."
}


class TestDocumentProcessor:
    """Tests for the base DocumentProcessor class."""
    
    def test_init(self):
        """Test initialization of DocumentProcessor."""
        processor = DocumentProcessor()
        assert isinstance(processor, DocumentProcessor)
    
    def test_process_not_implemented(self):
        """Test that process method must be implemented by subclasses."""
        processor = DocumentProcessor()
        with pytest.raises(NotImplementedError):
            processor.process("dummy_path.txt")
    
    def test_detect_language_english(self):
        """Test language detection for English text."""
        text = "This is a sample English text with common words like the and of."
        language = DocumentProcessor.detect_language(text)
        assert language == "en"
    
    def test_detect_language_spanish(self):
        """Test language detection for Spanish text."""
        text = "Este es un ejemplo de texto en español con palabras comunes como el y la."
        language = DocumentProcessor.detect_language(text)
        assert language == "es"
    
    def test_detect_language_french(self):
        """Test language detection for French text."""
        text = "Ceci est un exemple de texte en français avec des mots courants comme le et la."
        language = DocumentProcessor.detect_language(text)
        assert language == "fr"
    
    def test_detect_language_german(self):
        """Test language detection for German text."""
        text = "Dies ist ein Beispieltext auf Deutsch mit häufigen Wörtern wie der und die."
        language = DocumentProcessor.detect_language(text)
        assert language == "de"
    
    def test_detect_language_italian(self):
        """Test language detection for Italian text."""
        text = "Questo è un esempio di testo in italiano con parole comuni come il e la."
        language = DocumentProcessor.detect_language(text)
        assert language == "it"
    
    def test_detect_language_unknown(self):
        """Test language detection for unknown text."""
        text = "xyz123 abc456"  # No common words for any language
        language = DocumentProcessor.detect_language(text)
        assert language == "unknown"


class TestTextProcessor:
    """Tests for the TextProcessor class."""
    
    @pytest.fixture
    def text_processor(self):
        """Create a TextProcessor for testing."""
        return TextProcessor(chunk_size=100, chunk_overlap=20)
    
    @pytest.fixture
    def temp_text_file(self):
        """Create a temporary text file for testing."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as f:
            f.write(TEST_TEXT)
            filepath = f.name
        
        yield filepath
        
        # Cleanup
        if os.path.exists(filepath):
            os.unlink(filepath)
    
    def test_init(self):
        """Test initialization of TextProcessor."""
        processor = TextProcessor(chunk_size=200, chunk_overlap=50)
        assert processor.chunk_size == 200
        assert processor.chunk_overlap == 50
        assert processor.tokenizer is None
    
    def test_chunk_text(self, text_processor):
        """Test text chunking functionality."""
        chunks = text_processor._chunk_text(TEST_TEXT, "test_source")
        
        # Verify chunks were created
        assert len(chunks) > 0
        
        # Verify chunk structure
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "source" in chunk
            assert "metadata" in chunk
            assert "language" in chunk["metadata"]
            assert "start_char" in chunk["metadata"]
            assert "end_char" in chunk["metadata"]
            assert "character_count" in chunk["metadata"]
    
    def test_process(self, text_processor, temp_text_file):
        """Test processing a text file."""
        chunks = text_processor.process(temp_text_file)
        
        # Verify chunks were created
        assert len(chunks) > 0
        
        # Verify chunks contain the text content
        combined_text = " ".join([chunk["text"] for chunk in chunks])
        # Check if key phrases from the original text are present
        assert "test document" in combined_text
        assert "multiple sentences" in combined_text
        assert "second paragraph" in combined_text
    
    def test_process_with_tokenizer(self, temp_text_file):
        """Test processing with a tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = "MOCK_TOKENS"
        
        processor = TextProcessor(chunk_size=100, chunk_overlap=20, tokenizer=mock_tokenizer)
        chunks = processor.process(temp_text_file)
        
        # Verify tokenizer was used
        assert mock_tokenizer.encode.call_count > 0
        
        # Verify tokens were added to metadata
        for chunk in chunks:
            assert "tokens" in chunk["metadata"]
            assert chunk["metadata"]["tokens"] == "MOCK_TOKENS"
    
    def test_process_nonexistent_file(self, text_processor):
        """Test processing a non-existent file."""
        chunks = text_processor.process("/path/to/nonexistent/file.txt")
        assert chunks == []


class TestIndexer:
    """Tests for the Indexer class."""
    
    @pytest.fixture
    def mock_tokenizer_pipeline(self):
        """Create a mock for TokenizerPipeline."""
        with patch('app.core.rag.indexer.TokenizerPipeline') as mock:
            mock_instance = MagicMock()
            mock_instance.encode.return_value = "MOCK_TOKENS"
            mock.return_value = mock_instance
            yield mock
    
    @pytest.fixture
    def mock_model_registry(self):
        """Create a mock for ModelRegistry."""
        with patch('app.core.rag.indexer.ModelRegistry') as mock:
            mock_instance = MagicMock()
            mock_instance.get_model_and_tokenizer.return_value = (None, "mock-tokenizer")
            mock.return_value = mock_instance
            yield mock
    
    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock for SessionManager."""
        with patch('app.core.rag.indexer.SessionManager') as mock:
            mock_instance = MagicMock()
            mock_instance.get_document = AsyncMock()
            mock_instance.get_all_documents = AsyncMock()
            mock.return_value = mock_instance
            yield mock
    
    @pytest.fixture
    def indexer(self, mock_tokenizer_pipeline, mock_model_registry, mock_session_manager):
        """Create an Indexer instance for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            indexer = Indexer(output_dir=tmp_dir)
            yield indexer
    
    @pytest.fixture
    def temp_text_file(self):
        """Create a temporary text file for testing."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as f:
            f.write(TEST_TEXT)
            filepath = f.name
        
        yield filepath
        
        # Cleanup
        if os.path.exists(filepath):
            os.unlink(filepath)
    
    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create some text files in the directory
            for i in range(3):
                with open(os.path.join(tmp_dir, f"test_{i}.txt"), 'w') as f:
                    f.write(f"This is test file {i}.\n" * 5)
            
            # Create a subdirectory with a file
            subdir = os.path.join(tmp_dir, "subdir")
            os.makedirs(subdir)
            with open(os.path.join(subdir, "subdir_file.txt"), 'w') as f:
                f.write("This is a file in the subdirectory.\n" * 5)
            
            yield tmp_dir
    
    def test_init(self, mock_tokenizer_pipeline, mock_model_registry, mock_session_manager):
        """Test initialization of Indexer."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            indexer = Indexer(output_dir=tmp_dir, chunk_size=150, chunk_overlap=30)
            
            assert indexer.output_dir == tmp_dir
            assert indexer.chunk_size == 150
            assert indexer.chunk_overlap == 30
            assert ".txt" in indexer.processors
            assert ".docx" in indexer.processors
            assert ".csv" in indexer.processors
            assert os.path.exists(tmp_dir)
    
    def test_register_processor(self, indexer):
        """Test registering a custom processor."""
        custom_processor = MagicMock(spec=DocumentProcessor)
        indexer.register_processor(".pdf", custom_processor)
        
        assert ".pdf" in indexer.processors
        assert indexer.processors[".pdf"] == custom_processor
    
    def test_index_file(self, indexer, temp_text_file):
        """Test indexing a single file."""
        chunks = indexer.index_file(temp_text_file)
        
        # Verify chunks were created
        assert len(chunks) > 0
        
        # Verify indexing metadata was added
        for chunk in chunks:
            assert "metadata" in chunk
            assert "indexed_at" in chunk["metadata"]
            assert "file_path" in chunk["metadata"]
            assert "file_name" in chunk["metadata"]
            assert "file_extension" in chunk["metadata"]
            assert chunk["metadata"]["file_path"] == temp_text_file
            assert chunk["metadata"]["file_extension"] == ".txt"
    
    def test_index_nonexistent_file(self, indexer):
        """Test indexing a non-existent file."""
        with patch('app.core.rag.indexer.logger.warning') as mock_warning:
            chunks = indexer.index_file("/path/to/nonexistent/file.txt")
            assert chunks == []
            mock_warning.assert_called_once()
    
    def test_index_unsupported_extension(self, indexer, temp_text_file):
        """Test indexing a file with unsupported extension."""
        # Rename the file to have an unsupported extension
        unsupported_file = temp_text_file + ".xyz"
        os.rename(temp_text_file, unsupported_file)
        
        try:
            with patch('app.core.rag.indexer.logger.warning') as mock_warning:
                chunks = indexer.index_file(unsupported_file)
                assert chunks == []
                mock_warning.assert_called_once()
        finally:
            # Cleanup
            if os.path.exists(unsupported_file):
                os.unlink(unsupported_file)
    
    def test_index_directory(self, indexer, temp_directory):
        """Test indexing a directory."""
        chunks = indexer.index_directory(temp_directory)
        
        # Verify chunks were created for all files
        assert len(chunks) > 0
        
        # Verify chunks for main directory files
        main_dir_chunks = [c for c in chunks if "subdir" not in c["metadata"]["file_path"]]
        assert len(main_dir_chunks) > 0
        
        # Verify chunks for subdirectory files when recursive=True
        subdir_chunks = [c for c in chunks if "subdir" in c["metadata"]["file_path"]]
        assert len(subdir_chunks) > 0
    
    def test_index_directory_non_recursive(self, indexer, temp_directory):
        """Test indexing a directory non-recursively."""
        chunks = indexer.index_directory(temp_directory, recursive=False)
        
        # Verify chunks were created for main directory files
        assert len(chunks) > 0
        
        # Verify no chunks for subdirectory files when recursive=False
        subdir_chunks = [c for c in chunks if "subdir" in c["metadata"]["file_path"]]
        assert len(subdir_chunks) == 0
    
    def test_index_directory_with_extensions(self, indexer, temp_directory):
        """Test indexing a directory with specific extensions."""
        # Add a file with a different extension
        other_file = os.path.join(temp_directory, "other_file.dat")
        with open(other_file, 'w') as f:
            f.write("This is a file with a different extension.\n" * 5)
        
        chunks = indexer.index_directory(temp_directory, file_extensions=[".txt"])
        
        # Verify chunks were created for txt files
        assert len(chunks) > 0
        
        # Verify all chunks are from txt files
        for chunk in chunks:
            assert chunk["metadata"]["file_extension"] == ".txt"
    
    def test_save_index(self, indexer):
        """Test saving indexed chunks to a file."""
        chunks = [
            {
                "id": "test1",
                "text": "Test chunk 1",
                "source": "test_source",
                "metadata": {"language": "en"}
            },
            {
                "id": "test2",
                "text": "Test chunk 2",
                "source": "test_source",
                "metadata": {"language": "en"}
            }
        ]
        
        output_file = os.path.join(indexer.output_dir, "test_index.json")
        saved_path = indexer.save_index(chunks, output_file)
        
        # Verify file was created
        assert os.path.exists(saved_path)
        assert saved_path == output_file
        
        # Verify file contains the correct chunks
        with open(saved_path, 'r') as f:
            saved_chunks = json.load(f)
            assert len(saved_chunks) == 2
            assert saved_chunks[0]["id"] == "test1"
            assert saved_chunks[1]["id"] == "test2"
    
    def test_save_index_with_rag_expert(self, indexer):
        """Test saving indexed chunks with a RAG expert."""
        # Create mock RAG expert
        mock_expert = MagicMock()
        mock_expert.knowledge_base = []
        mock_expert._build_index = AsyncMock()
        indexer.rag_expert = mock_expert
        
        chunks = [
            {"id": "test1", "text": "Test chunk 1", "source": "test_source", "metadata": {"language": "en"}},
            {"id": "test2", "text": "Test chunk 2", "source": "test_source", "metadata": {"language": "en"}}
        ]
        
        # Import asyncio directly and patch it within the indexer module
        import asyncio
        with patch.object(asyncio, 'create_task') as mock_create_task:
            # Save the original method
            orig_method = indexer.save_index
            
            # Replace it with our instrumented version
            def instrumented_save_index(chunks, output_file=None):
                result = orig_method(chunks, output_file)
                # Check if _build_index was accessed
                if hasattr(mock_expert, '_build_index'):
                    # Verify it was called or accessed
                    assert mock_expert._build_index.called or mock_expert._build_index._mock_called
                return result
                
            indexer.save_index = instrumented_save_index
            
            try:
                saved_path = indexer.save_index(chunks)
                
                # Verify chunks were added to expert's knowledge base
                assert len(mock_expert.knowledge_base) == 2
                
            finally:
                # Restore original method
                indexer.save_index = orig_method
    
    def test_index_and_save(self, indexer, temp_text_file):
        """Test indexing and saving in one operation."""
        output_file = os.path.join(indexer.output_dir, "combined_index.json")
        result_path, chunk_count = indexer.index_and_save(temp_text_file, output_file)
        
        # Verify result
        assert os.path.exists(result_path)
        assert result_path == output_file
        assert chunk_count > 0
    
    @pytest.mark.asyncio
    async def test_index_document_content(self, indexer):
        """Test indexing document content directly from bytes."""
        document_content = TEST_TEXT.encode('utf-8')
        
        # Mock the processor if it exists
        if hasattr(indexer, 'processor') and indexer.processor:
            mock_processor = MagicMock()
            mock_processor.extract_document_text = AsyncMock(return_value={
                "text": TEST_TEXT,
                "metadata": {"page_count": 1}
            })
            indexer.processor = mock_processor
        
        chunks = await indexer.index_document_content(
            document_content=document_content,
            document_type="text/plain",
            filename="test.txt",
            metadata={"test_key": "test_value"}
        )
        
        # Verify chunks were created
        assert len(chunks) > 0
        
        # Verify metadata was added
        for chunk in chunks:
            assert "metadata" in chunk
            assert "indexed_at" in chunk["metadata"]
            assert "document_id" in chunk["metadata"]
            assert "file_name" in chunk["metadata"]
            assert "mime_type" in chunk["metadata"]
            assert "test_key" in chunk["metadata"]
            assert chunk["metadata"]["test_key"] == "test_value"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])