import os
import aiohttp
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from transformers import AutoModel

# Configure logger
logger = logging.getLogger(__name__)

class OCRBase(ABC):
    @abstractmethod
    async def extract_text(self, file_path: str) -> str:
        """Extract text from a local file asynchronously."""
        pass

class JinaReaderOCR(OCRBase):
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            logger.warning("JINA_API_KEY not found. JinaReaderOCR may fail.")
        self.api_url = "https://r.jina.ai/"

    async def extract_text(self, file_path: str) -> str:
        """
        Extracts text from a PDF using Jina Reader API.
        Assumes Jina Reader supports direct file upload via POST.
        """
        if not self.api_key:
            logger.error("Missing JINA_API_KEY")
            return ""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-With-Generated-Alt": "true" # Optional: Request captions for images
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                # Read file content
                with open(file_path, "rb") as f:
                    file_data = f.read()
                
                # Send POST request with file binary
                # Note: This usage pattern assumes Jina Reader accepts raw body or similar for local files.
                # If specific endpoint is needed (e.g., /v1/reader), update here.
                async with session.post(self.api_url, data=file_data, headers=headers) as response:
                    if response.status == 200:
                        text = await response.text()
                        return text
                    else:
                        error_msg = await response.text()
                        logger.error(f"Jina Reader API failed: {response.status} - {error_msg}")
                        return ""
            except Exception as e:
                logger.error(f"OCR failed for {file_path}: {e}")
                return ""

class TextEmbedder:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v4"):
        self.model_name = model_name
        self.model = None
        
        logger.info(f"Loading Text Embedding Model: {model_name}...")
        try:
            # Jina models require trust_remote_code=True
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}. Attempting fallback to v3.")
            try:
                self.model_name = "jinaai/jina-embeddings-v3"
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
                logger.info(f"Successfully loaded {self.model_name}")
            except Exception as ex:
                logger.error(f"Failed to load fallback model: {ex}")
                raise

    def embed(self, text: str) -> List[float]:
        """Generates dense vector embedding for the given text."""
        if not self.model:
            logger.error("Model not initialized.")
            return []

        try:
            # Jina embeddings models (v2/v3) expose an 'encode' method
            # task="retrieval.query" or "retrieval.passage" can be specified for v3
            # We use default or passage for ingestion.
            # v3 signature: model.encode(texts, task="retrieval.passage")
            
            # Check if encode method supports task argument (v3 specific)
            import inspect
            sig = inspect.signature(self.model.encode)
            
            if "task" in sig.parameters:
                embeddings = self.model.encode([text], task="retrieval.passage")
            else:
                embeddings = self.model.encode([text])
            
            return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
