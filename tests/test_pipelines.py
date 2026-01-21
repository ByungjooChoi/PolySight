"""
PolySight Pipeline Tests
Tests for Visual Agent, Text Agent, and Ingestion pipelines.
"""
import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestVisualEngine:
    """Tests for Visual Agent components."""

    def test_pdf_processor_convert_to_images(self, tmp_path):
        """Test PDF to images conversion."""
        from backend.pipelines.visual_engine import PDFProcessor

        # Create a simple test PDF would require pypdfium2
        # This test verifies the class exists and has correct interface
        assert hasattr(PDFProcessor, 'convert_to_images')

    def test_process_uploaded_file_image(self, tmp_path):
        """Test processing uploaded image file."""
        from backend.pipelines.visual_engine import process_uploaded_file

        # Create test image
        img = Image.new('RGB', (100, 100), color='red')
        img_path = tmp_path / "test.png"
        img.save(img_path)

        # Process
        result = process_uploaded_file(str(img_path))

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Image.Image)

    def test_process_uploaded_file_unsupported(self, tmp_path):
        """Test handling of unsupported file types."""
        from backend.pipelines.visual_engine import process_uploaded_file

        # Create test file with unsupported extension
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            process_uploaded_file(str(test_file))

    def test_token_pooler_initialization(self):
        """Test TokenPooler initialization."""
        from backend.pipelines.visual_engine import TokenPooler

        pooler = TokenPooler(pool_factor=3)
        assert pooler.pool_factor == 3

    @patch('backend.pipelines.visual_engine.HierarchicalTokenPooler', create=True)
    def test_token_pooler_pool_vectors(self, mock_pooler_class):
        """Test token pooling functionality."""
        from backend.pipelines.visual_engine import TokenPooler
        import torch

        # Mock the pooler
        mock_pooler = MagicMock()
        mock_pooler.pool_embeddings.return_value = torch.randn(1, 43, 128)
        mock_pooler_class.return_value = mock_pooler

        pooler = TokenPooler(pool_factor=3)
        pooler._pooler = mock_pooler

        # Test input: 128 vectors of 128 dimensions
        embeddings = [[0.1] * 128] * 128

        result = pooler.pool_vectors(embeddings)

        # Should return pooled vectors
        assert isinstance(result, list)


class TestTextEngine:
    """Tests for Text Agent components."""

    def test_ocr_base_interface(self):
        """Test OCRBase abstract interface."""
        from backend.pipelines.text_engine import OCRBase

        # Should be abstract
        with pytest.raises(TypeError):
            OCRBase()

    def test_docling_ocr_initialization(self):
        """Test DoclingOCR initialization."""
        from backend.pipelines.text_engine import DoclingOCR

        ocr = DoclingOCR(languages=["en", "ko"])
        assert ocr.languages == ["en", "ko"]

    def test_simple_ocr_fallback(self):
        """Test SimpleOCR fallback."""
        from backend.pipelines.text_engine import SimpleOCR

        ocr = SimpleOCR()
        # Should not crash even without tesseract
        assert hasattr(ocr, 'extract_text_from_image')

    def test_get_ocr_engine(self):
        """Test OCR engine factory function."""
        from backend.pipelines.text_engine import get_ocr_engine, OCRBase

        engine = get_ocr_engine(prefer_docling=False)
        assert isinstance(engine, OCRBase)

    def test_text_agent_initialization(self):
        """Test TextAgent initialization."""
        from backend.pipelines.text_engine import TextAgent

        agent = TextAgent()
        assert hasattr(agent, 'extract_text')
        assert hasattr(agent, 'process_images')


class TestElasticClient:
    """Tests for Elasticsearch client."""

    def test_client_singleton(self):
        """Test ElasticClient singleton pattern."""
        from backend.utils.elastic_client import ElasticClient

        client1 = ElasticClient()
        client2 = ElasticClient()

        # Should be same instance
        assert client1 is client2

    def test_index_names(self):
        """Test index name constants."""
        from backend.utils.elastic_client import ElasticClient

        assert ElasticClient.VISUAL_INDEX == "polysight-visual"
        assert ElasticClient.TEXT_INDEX == "polysight-text"

    @patch('backend.utils.elastic_client.Elasticsearch')
    def test_ensure_indices(self, mock_es):
        """Test index creation."""
        from backend.utils.elastic_client import ElasticClient

        # Reset singleton
        ElasticClient._instance = None

        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False
        mock_es.return_value = mock_client

        # Set env vars
        with patch.dict(os.environ, {
            'ELASTIC_CLOUD_SERVERLESS_URL': 'http://test:9200',
            'ELASTIC_API_KEY': 'test-key'
        }):
            client = ElasticClient()
            client.ensure_indices()

        # Should have attempted to create indices
        assert mock_client.indices.create.called


class TestIngestion:
    """Tests for Ingestion pipeline."""

    def test_ingestion_manager_initialization(self):
        """Test IngestionManager initialization."""
        from backend.pipelines.ingestion import IngestionManager

        manager = IngestionManager()
        assert hasattr(manager, 'process_file')
        assert hasattr(manager, 'process_file_sync')

    def test_search_manager_initialization(self):
        """Test SearchManager initialization."""
        from backend.pipelines.ingestion import SearchManager

        manager = SearchManager()
        assert hasattr(manager, 'search_visual')
        assert hasattr(manager, 'search_text')
        assert hasattr(manager, 'search_both')


class TestMCPTools:
    """Tests for MCP Server tools."""

    @pytest.mark.asyncio
    async def test_compare_search_results_no_connection(self):
        """Test compare_search_results with no Elastic connection."""
        from backend.mcp_server.tools.comparison import compare_search_results

        with patch('backend.mcp_server.tools.comparison.ElasticClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.get_client.return_value = None
            mock_client.return_value = mock_instance

            result = await compare_search_results("test query")

            assert "Error" in result or "not connected" in result.lower()

    @pytest.mark.asyncio
    async def test_get_index_status(self):
        """Test get_index_status tool."""
        from backend.mcp_server.tools.comparison import get_index_status

        with patch('backend.mcp_server.tools.comparison.ElasticClient') as mock_client:
            mock_instance = MagicMock()
            mock_instance.get_client.return_value = MagicMock()
            mock_instance.get_index_count.return_value = 10
            mock_client.return_value = mock_instance

            result = await get_index_status()

            assert "Index Status" in result
            assert "10" in result


class TestViDoReLoader:
    """Tests for ViDoRe dataset loader."""

    def test_loader_initialization(self):
        """Test ViDoReLoader initialization."""
        from backend.data.vidore_loader import ViDoReLoader

        loader = ViDoReLoader()
        assert loader.DATASET_NAME == "vidore/vidore-benchmark-v3"


# Import check test
def test_imports():
    """Test all imports work correctly."""
    from backend.pipelines.visual_engine import (
        PDFProcessor,
        VisualEmbedder,
        TokenPooler,
        process_uploaded_file
    )
    from backend.pipelines.text_engine import (
        DoclingOCR,
        TextAgent,
        get_ocr_engine
    )
    from backend.pipelines.ingestion import (
        IngestionManager,
        SearchManager
    )
    from backend.utils.elastic_client import ElasticClient
    from backend.data.vidore_loader import ViDoReLoader

    print("All imports successful!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
