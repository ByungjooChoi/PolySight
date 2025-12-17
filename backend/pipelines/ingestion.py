import asyncio
import logging
from typing import Dict, Any, List
from backend.pipelines.text_engine import JinaReaderOCR, TextEmbedder
from backend.pipelines.visual_engine import PDFProcessor, VisualEmbedder
from backend.utils.elastic_client import ElasticClient
from elasticsearch.helpers import bulk

logger = logging.getLogger(__name__)

class IngestionManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IngestionManager, cls).__new__(cls)
            cls._instance._init_engines()
        return cls._instance

    def _init_engines(self):
        logger.info("Initializing Ingestion Engines...")
        try:
            self.text_embedder = TextEmbedder()
            self.visual_embedder = VisualEmbedder()
            self.ocr_engine = JinaReaderOCR()
            self.elastic_client = ElasticClient().get_client()
        except Exception as e:
            logger.error(f"Failed to initialize engines: {e}")
            raise

    async def process_pdf(self, file_path: str):
        """
        Processes a PDF file through both Visual and Text pipelines and indexes to Elastic.
        """
        logger.info(f"Starting processing for {file_path}")
        
        if not self.elastic_client:
            logger.error("Elastic Client not available. Skipping indexing.")
            return

        # Define Tasks
        async def run_text_pipeline():
            try:
                logger.info("Running Text Pipeline (OCR)...")
                text = await self.ocr_engine.extract_text(file_path)
                if not text:
                    logger.warning("OCR returned empty text.")
                    return None
                
                logger.info("Generating Text Embedding...")
                # embed is CPU bound, run in executor
                loop = asyncio.get_running_loop()
                embedding = await loop.run_in_executor(None, self.text_embedder.embed, text)
                return {"text": text, "embedding": embedding}
            except Exception as e:
                logger.error(f"Text pipeline failed: {e}")
                return None

        async def run_visual_pipeline():
            try:
                logger.info("Running Visual Pipeline (PDF -> Image -> Embed)...")
                loop = asyncio.get_running_loop()
                
                # PDF conversion (CPU bound)
                images = await loop.run_in_executor(None, PDFProcessor.convert_to_images, file_path)
                
                page_embeddings = []
                for idx, img in enumerate(images):
                    # Embedding (CPU/GPU bound)
                    emb = await loop.run_in_executor(None, self.visual_embedder.embed_image, img)
                    page_embeddings.append({
                        "page": idx + 1,
                        "embedding": emb
                    })
                return page_embeddings
            except Exception as e:
                logger.error(f"Visual pipeline failed: {e}")
                return None

        # Execute Pipelines Concurrently
        results = await asyncio.gather(run_text_pipeline(), run_visual_pipeline())
        text_data, visual_data = results

        # Prepare Bulk Actions
        actions = []
        
        # 1. Text Index Action
        if text_data:
            actions.append({
                "_index": "text-index",
                "_source": {
                    "file_path": file_path,
                    "content": text_data["text"],
                    "dense_vector": text_data["embedding"],
                    "pipeline": "text"
                }
            })
            
        # 2. Visual Index Actions
        if visual_data:
            for item in visual_data:
                actions.append({
                    "_index": "visual-index",
                    "_source": {
                        "file_path": file_path,
                        "page_number": item["page"],
                        "rank_vectors": item["embedding"], # Mapped to dense_vector
                        "pipeline": "visual"
                    }
                })

        # Bulk Indexing
        if actions:
            logger.info(f"Indexing {len(actions)} documents to Elastic...")
            try:
                # bulk is synchronous, run in executor
                loop = asyncio.get_running_loop()
                success, _ = await loop.run_in_executor(None, lambda: bulk(self.elastic_client, actions))
                logger.info(f"Successfully indexed {success} documents.")
                
                text_count = 1 if text_data else 0
                visual_count = len(visual_data) if visual_data else 0
                
                return {
                    "visual_count": visual_count,
                    "text_count": text_count,
                    "total_indexed": success
                }
            except Exception as e:
                logger.error(f"Indexing failed: {e}")
                raise
        else:
            logger.warning("No data generated to index.")
            return {
                "visual_count": 0,
                "text_count": 0,
                "total_indexed": 0
            }
