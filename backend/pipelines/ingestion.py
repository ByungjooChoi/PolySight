"""
Ingestion Pipeline for PolySight
Orchestrates Visual Agent and Text Agent processing and indexing.
"""
import os
import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from backend.pipelines.visual_engine import (
    PDFProcessor,
    VisualEmbedder,
    process_uploaded_file
)
from backend.pipelines.text_engine import TextAgent
from backend.utils.elastic_client import ElasticClient

logger = logging.getLogger(__name__)


class IngestionManager:
    """
    Manages ingestion of documents through both Visual and Text pipelines.
    """

    def __init__(
        self,
        visual_embedder: Optional[VisualEmbedder] = None,
        text_agent: Optional[TextAgent] = None,
        elastic_client: Optional[ElasticClient] = None
    ):
        """
        Initialize Ingestion Manager.

        Args:
            visual_embedder: Visual Agent embedder (lazy loaded if None)
            text_agent: Text Agent (lazy loaded if None)
            elastic_client: Elasticsearch client
        """
        self._visual_embedder = visual_embedder
        self._text_agent = text_agent
        self.elastic = elastic_client or ElasticClient()
        self.executor = ThreadPoolExecutor(max_workers=4)

    @property
    def visual_embedder(self) -> VisualEmbedder:
        """Lazy load Visual Embedder"""
        if self._visual_embedder is None:
            self._visual_embedder = VisualEmbedder()
        return self._visual_embedder

    @property
    def text_agent(self) -> TextAgent:
        """Lazy load Text Agent"""
        if self._text_agent is None:
            self._text_agent = TextAgent()
        return self._text_agent

    async def process_file(
        self,
        file_path: str,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a file through both Visual and Text pipelines.

        Args:
            file_path: Path to the file (PDF or image)
            doc_id: Optional document ID (auto-generated if None)

        Returns:
            Dict with processing stats
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = Path(file_path).name
        doc_id = doc_id or str(uuid.uuid4())

        logger.info(f"Processing file: {file_name} (doc_id: {doc_id})")

        # Ensure indices exist
        self.elastic.ensure_indices()

        # Convert file to images
        loop = asyncio.get_event_loop()
        images = await loop.run_in_executor(
            self.executor,
            process_uploaded_file,
            file_path
        )

        logger.info(f"Loaded {len(images)} images from {file_name}")

        # Process through both pipelines in parallel
        visual_task = self._process_visual_pipeline(
            images, doc_id, file_path, file_name
        )
        text_task = self._process_text_pipeline(
            images, doc_id, file_path, file_name
        )

        visual_count, text_count = await asyncio.gather(visual_task, text_task)

        return {
            "doc_id": doc_id,
            "file_name": file_name,
            "page_count": len(images),
            "visual_count": visual_count,
            "text_count": text_count
        }

    async def _process_visual_pipeline(
        self,
        images: List[Image.Image],
        doc_id: str,
        file_path: str,
        file_name: str
    ) -> int:
        """
        Process images through Visual Agent pipeline.

        Returns:
            Number of successfully indexed documents
        """
        loop = asyncio.get_event_loop()
        count = 0

        logger.info(f"Starting Visual Pipeline for {len(images)} images...")

        for page_num, image in enumerate(images):
            try:
                # Generate multi-vector embedding (CPU/GPU bound)
                multi_vectors = await loop.run_in_executor(
                    self.executor,
                    self.visual_embedder.embed_image,
                    image
                )

                # Index to Elasticsearch
                success = self.elastic.index_visual(
                    doc_id=doc_id,
                    visual_vectors=multi_vectors,
                    page_number=page_num,
                    file_path=file_path,
                    file_name=file_name
                )

                if success:
                    count += 1
                    logger.debug(f"Visual indexed: page {page_num}")

            except Exception as e:
                logger.error(f"Visual pipeline failed for page {page_num}: {e}")

        logger.info(f"Visual Pipeline complete: {count}/{len(images)} indexed")
        return count

    async def _process_text_pipeline(
        self,
        images: List[Image.Image],
        doc_id: str,
        file_path: str,
        file_name: str
    ) -> int:
        """
        Process images through Text Agent pipeline.

        Returns:
            Number of successfully indexed documents
        """
        loop = asyncio.get_event_loop()
        count = 0

        logger.info(f"Starting Text Pipeline for {len(images)} images...")

        for page_num, image in enumerate(images):
            try:
                # Extract text using OCR (CPU bound)
                text = await loop.run_in_executor(
                    self.executor,
                    self.text_agent.extract_text,
                    image
                )

                if not text.strip():
                    logger.warning(f"No text extracted from page {page_num}")
                    continue

                # Index to Elasticsearch
                success = self.elastic.index_text(
                    doc_id=doc_id,
                    ocr_text=text,
                    page_number=page_num,
                    file_path=file_path,
                    file_name=file_name
                )

                if success:
                    count += 1
                    logger.debug(f"Text indexed: page {page_num}")

            except Exception as e:
                logger.error(f"Text pipeline failed for page {page_num}: {e}")

        logger.info(f"Text Pipeline complete: {count}/{len(images)} indexed")
        return count

    def process_file_sync(
        self,
        file_path: str,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for process_file.
        """
        return asyncio.run(self.process_file(file_path, doc_id))

    async def process_image(
        self,
        image: Image.Image,
        doc_id: str,
        page_number: int = 0,
        file_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Process a single PIL Image through both pipelines.
        Used for ViDoRe samples which are already loaded as images.

        Args:
            image: PIL Image to process
            doc_id: Document ID
            page_number: Page number
            file_name: File name for metadata

        Returns:
            Dict with processing stats
        """
        # Ensure indices exist
        self.elastic.ensure_indices()

        loop = asyncio.get_event_loop()
        visual_success = False
        text_success = False

        # Visual Pipeline
        try:
            multi_vectors = await loop.run_in_executor(
                self.executor,
                self.visual_embedder.embed_image,
                image
            )
            visual_success = self.elastic.index_visual(
                doc_id=doc_id,
                visual_vectors=multi_vectors,
                page_number=page_number,
                file_path="vidore_sample",
                file_name=file_name
            )
        except Exception as e:
            logger.error(f"Visual pipeline failed for {doc_id}: {e}")

        # Text Pipeline
        try:
            text = await loop.run_in_executor(
                self.executor,
                self.text_agent.extract_text,
                image
            )
            if text.strip():
                text_success = self.elastic.index_text(
                    doc_id=doc_id,
                    ocr_text=text,
                    page_number=page_number,
                    file_path="vidore_sample",
                    file_name=file_name
                )
        except Exception as e:
            logger.error(f"Text pipeline failed for {doc_id}: {e}")

        return {
            "doc_id": doc_id,
            "visual_indexed": visual_success,
            "text_indexed": text_success
        }


class SearchManager:
    """
    Manages search through both Visual Agent and Text Agent.
    """

    def __init__(
        self,
        visual_embedder: Optional[VisualEmbedder] = None,
        elastic_client: Optional[ElasticClient] = None
    ):
        self._visual_embedder = visual_embedder
        self.elastic = elastic_client or ElasticClient()

    @property
    def visual_embedder(self) -> VisualEmbedder:
        """Lazy load Visual Embedder"""
        if self._visual_embedder is None:
            self._visual_embedder = VisualEmbedder()
        return self._visual_embedder

    def search_visual(
        self,
        query: str,
        size: int = 5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Search using Visual Agent (MaxSim).

        Args:
            query: Search query text
            size: Number of results

        Returns:
            Tuple of (results, latency_ms)
        """
        import time
        start = time.time()

        # Generate query multi-vectors
        query_vectors = self.visual_embedder.embed_query(query)

        # Search with MaxSim
        results = self.elastic.search_visual_maxsim(query_vectors, size)

        latency = (time.time() - start) * 1000
        return results, latency

    def search_text(
        self,
        query: str,
        size: int = 5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Search using Text Agent (BM25).

        Args:
            query: Search query text
            size: Number of results

        Returns:
            Tuple of (results, latency_ms)
        """
        import time
        start = time.time()

        # Search with BM25 (no embedding needed)
        results = self.elastic.search_text_bm25(query, size)

        latency = (time.time() - start) * 1000
        return results, latency

    def search_both(
        self,
        query: str,
        size: int = 5
    ) -> Dict[str, Any]:
        """
        Search using both agents and compare results.

        Args:
            query: Search query text
            size: Number of results per agent

        Returns:
            Dict with both results and comparison
        """
        visual_results, visual_latency = self.search_visual(query, size)
        text_results, text_latency = self.search_text(query, size)

        return {
            "query": query,
            "visual_agent": {
                "results": visual_results,
                "latency_ms": visual_latency,
                "count": len(visual_results)
            },
            "text_agent": {
                "results": text_results,
                "latency_ms": text_latency,
                "count": len(text_results)
            }
        }
