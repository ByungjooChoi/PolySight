"""
Visual Agent Engine for PolySight
- Jina V4 Multi-vector Embedding (128 dim)
- Token Pooling (HierarchicalTokenPooler, pool_factor=3)
- Late Interaction (MaxSim) support
- Supports both Local model and Jina API
"""
import os
import io
import base64
import logging
import torch
import requests
from typing import List, Optional, Union
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium

# Configure logger
logger = logging.getLogger(__name__)

# Jina API endpoint
JINA_API_URL = "https://api.jina.ai/v1/embeddings"


class PDFProcessor:
    """PDF to Images converter using pypdfium2"""

    @staticmethod
    def convert_to_images(file_path: str, scale: float = 3.0) -> List[Image.Image]:
        """
        Converts a PDF file to a list of PIL Images (one per page).
        Renders at high resolution (scale=3.0) for better VLM performance.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            logger.info(f"Converting PDF: {file_path} with scale={scale}")

            pdf = pdfium.PdfDocument(file_path)
            images = []

            for i, page in enumerate(pdf):
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                images.append(pil_image)

            logger.info(f"Converted {len(images)} pages from PDF.")
            return images

        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise


def process_uploaded_file(file_path: str) -> List[Image.Image]:
    """
    Process uploaded file and return list of images.
    Supports PDF and common image formats.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return PDFProcessor.convert_to_images(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".gif"]:
        return [Image.open(file_path).convert("RGB")]
    else:
        raise ValueError(f"Unsupported file type: {ext}")


class TokenPooler:
    """
    Token Pooling using HierarchicalTokenPooler from colpali_engine.
    Reduces vector count by pool_factor while maintaining accuracy (~94% with factor=3).

    Reference: https://github.com/elastic/elasticsearch-labs/tree/main/supporting-blog-content/colpali
    """

    def __init__(self, pool_factor: int = 3):
        self.pool_factor = pool_factor
        self._pooler = None

    @property
    def pooler(self):
        """Lazy load pooler to avoid import errors if colpali_engine not installed"""
        if self._pooler is None:
            try:
                from colpali_engine.compression.token_pooling import HierarchicalTokenPooler
                self._pooler = HierarchicalTokenPooler(pool_factor=self.pool_factor)
                logger.info(f"TokenPooler initialized with pool_factor={self.pool_factor}")
            except ImportError:
                logger.warning("colpali_engine not installed. Token pooling disabled.")
                self._pooler = None
        return self._pooler

    def pool_vectors(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Apply hierarchical token pooling to reduce vector count.

        Args:
            embeddings: List of vectors [[dim], [dim], ...] or tensor

        Returns:
            Pooled vectors as List[List[float]]
        """
        if not self.pooler:
            logger.warning("Pooler not available, returning original embeddings")
            return embeddings

        try:
            # Convert to tensor: (1, num_tokens, dim)
            tensor = torch.tensor(embeddings).unsqueeze(0)

            # Apply pooling
            pooled = self.pooler.pool_embeddings(tensor)

            # Convert back to list: (num_pooled_tokens, dim)
            result = pooled.squeeze(0).tolist()

            logger.debug(f"Pooled {len(embeddings)} vectors to {len(result)} vectors")
            return result

        except Exception as e:
            logger.error(f"Token pooling failed: {e}")
            return embeddings


class JinaAPIClient:
    """
    Jina API Client for cloud-based embedding.
    Uses Jina AI's embedding API endpoint.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def embed_image(self, image: Image.Image) -> List[List[float]]:
        """
        Generate multi-vector embedding for image via API.
        """
        try:
            image_b64 = self._image_to_base64(image)

            payload = {
                "model": "jina-embeddings-v4",
                "task": "retrieval.passage",
                "input": [{"image": image_b64}],
                "embedding_type": "float",
                "return_multivector": True
            }

            response = requests.post(
                JINA_API_URL,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            data = response.json()
            embeddings = data["data"][0]["embedding"]

            logger.debug(f"API returned {len(embeddings)} vectors for image")
            return embeddings

        except requests.exceptions.RequestException as e:
            logger.error(f"Jina API request failed: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Jina API response parsing failed: {e}")
            raise

    def embed_query(self, query_text: str) -> List[List[float]]:
        """
        Generate multi-vector embedding for text query via API.
        """
        try:
            payload = {
                "model": "jina-embeddings-v4",
                "task": "retrieval.query",
                "input": [query_text],
                "embedding_type": "float",
                "return_multivector": True
            }

            response = requests.post(
                JINA_API_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            embeddings = data["data"][0]["embedding"]

            logger.debug(f"API returned {len(embeddings)} query vectors")
            return embeddings

        except requests.exceptions.RequestException as e:
            logger.error(f"Jina API request failed: {e}")
            raise


class VisualEmbedder:
    """
    Visual Embedder using Jina V4 for multi-vector embeddings.
    Supports both Local model and Jina API mode.
    Supports Late Interaction (MaxSim) with rank_vectors in Elasticsearch.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v4",
        pool_factor: int = 3,
        use_pooling: bool = True,
        mode: str = "local",  # "local" or "api"
        jina_api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.model = None
        self.pooler = TokenPooler(pool_factor=pool_factor) if use_pooling else None
        self.use_pooling = use_pooling
        self.mode = mode
        self.api_client = None

        # Initialize based on mode
        if mode == "api" and jina_api_key:
            self._init_api_client(jina_api_key)
        else:
            self._load_model()

    def _init_api_client(self, api_key: str):
        """Initialize Jina API client"""
        logger.info("Initializing Jina V4 in API mode...")
        self.api_client = JinaAPIClient(api_key)
        self.mode = "api"
        logger.info("Jina API client initialized successfully")

    def _load_model(self):
        """Load Jina V4 model locally"""
        logger.info(f"Loading Visual Embedding Model (Local): {self.model_name}...")

        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.mode = "local"
            logger.info(f"Successfully loaded {self.model_name} (Local mode)")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            raise

    def embed_image(self, image: Image.Image) -> List[List[float]]:
        """
        Generate multi-vector embedding for a single image.

        Returns:
            List of vectors (multi-vector) for Late Interaction.
            Each vector is 128-dim for Jina V4 multi-vector mode.
        """
        if self.mode == "api" and self.api_client:
            multi_vectors = self.api_client.embed_image(image)
        else:
            if not self.model:
                raise RuntimeError("Model not initialized")

            try:
                # Jina V4 multi-vector encoding
                # task="retrieval.passage" for document/image indexing
                embeddings = self.model.encode(
                    [image],
                    task="retrieval.passage",
                    return_multivector=True  # Enable multi-vector mode (128 dim)
                )

                # embeddings shape: (1, num_tokens, 128)
                # Convert to list of vectors
                if hasattr(embeddings, 'tolist'):
                    multi_vectors = embeddings[0].tolist()
                else:
                    multi_vectors = list(embeddings[0])

            except Exception as e:
                logger.error(f"Image embedding failed: {e}")
                raise

        # Apply token pooling if enabled (for both modes)
        if self.use_pooling and self.pooler:
            multi_vectors = self.pooler.pool_vectors(multi_vectors)

        logger.debug(f"Generated {len(multi_vectors)} vectors for image (mode={self.mode})")
        return multi_vectors

    def embed_query(self, query_text: str) -> List[List[float]]:
        """
        Generate multi-vector embedding for a text query.

        Returns:
            List of vectors (multi-vector) for MaxSim search.
        """
        if self.mode == "api" and self.api_client:
            multi_vectors = self.api_client.embed_query(query_text)
        else:
            if not self.model:
                raise RuntimeError("Model not initialized")

            try:
                # Jina V4 multi-vector encoding for query
                # task="retrieval.query" for search queries
                embeddings = self.model.encode(
                    [query_text],
                    task="retrieval.query",
                    return_multivector=True
                )

                if hasattr(embeddings, 'tolist'):
                    multi_vectors = embeddings[0].tolist()
                else:
                    multi_vectors = list(embeddings[0])

            except Exception as e:
                logger.error(f"Query embedding failed: {e}")
                raise

        # Note: Query vectors are NOT pooled (need full resolution for MaxSim)
        logger.debug(f"Generated {len(multi_vectors)} query vectors (mode={self.mode})")
        return multi_vectors

    def embed_images_batch(self, images: List[Image.Image]) -> List[List[List[float]]]:
        """
        Generate multi-vector embeddings for multiple images.

        Returns:
            List of multi-vectors for each image.
        """
        results = []
        for img in images:
            vectors = self.embed_image(img)
            results.append(vectors)
        return results

    @property
    def is_api_mode(self) -> bool:
        """Check if using API mode"""
        return self.mode == "api"

    def get_mode_info(self) -> str:
        """Get current mode information"""
        if self.mode == "api":
            return "Jina V4 (API Mode - Cloud)"
        else:
            return "Jina V4 (Local Mode - GPU)"
