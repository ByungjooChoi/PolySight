import os
import logging
import torch
from typing import List, Any, Union
from PIL import Image
import pypdfium2 as pdfium
from transformers import AutoModel
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)

class PDFProcessor:
    @staticmethod
    def convert_to_images(file_path: str, scale: float = 3.0) -> List[Image.Image]:
        """
        Converts a PDF file to a list of PIL Images (one per page) using pypdfium2.
        Renders at high resolution (scale=3.0) for better VLM/OCR performance.
        """
        try:
            # Verify file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            logger.info(f"Converting PDF: {file_path} with scale={scale}")
            
            # Load PDF
            pdf = pdfium.PdfDocument(file_path)
            images = []
            
            for i, page in enumerate(pdf):
                # Render page to bitmap with scaling
                # scale=3.0 ensures high resolution (~200-300 DPI equivalent depending on base)
                bitmap = page.render(scale=scale)
                
                # Convert to PIL Image
                pil_image = bitmap.to_pil()
                images.append(pil_image)
            
            logger.info(f"Converted {len(images)} pages.")
            return images
            
        except Exception as e:
            logger.error(f"PDF conversion failed with pypdfium2: {e}")
            raise e

def pool_tokens(tensor: torch.Tensor, ratio: float = 0.3) -> List[float]:
    """
    Pools tokens to reduce dimensionality or token count.
    
    If input is 1D (Single Vector), returns as List[float].
    If input is 2D (Tokens x Dim), performs pooling to return a single representative vector List[float]
    compatible with Elastic 'dense_vector'.
    
    Note: For true Late Interaction (ColPali), we would store multiple vectors.
    But for standard 'dense_vector', we must reduce to one.
    Here we implement Average Pooling as a standard approach.
    """
    if tensor is None:
        return []

    # Ensure tensor is on CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    if tensor.dim() == 1:
        return tensor.tolist()
    
    if tensor.dim() == 2:
        # Average Pooling: (Tokens, Dim) -> (Dim,)
        pooled = torch.mean(tensor, dim=0)
        return pooled.tolist()
    
    # Fallback
    return tensor.flatten().tolist()

class VisualEmbedder:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v4"):
        self.model_name = model_name
        self.model = None
        
        logger.info(f"Loading Visual Embedding Model: {model_name}...")
        try:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}. Falling back to v3.")
            try:
                self.model_name = "jinaai/jina-embeddings-v3"
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
                logger.info(f"Successfully loaded {self.model_name}")
            except Exception as ex:
                logger.error(f"Failed to load fallback model: {ex}")
                raise

    def embed_image(self, image: Image.Image) -> List[float]:
        """
        Generates embedding for a single image.
        Returns a list of floats compatible with Elastic dense_vector.
        """
        if not self.model:
            logger.error("Model not initialized.")
            return []

        try:
            # Jina v3 encode supports image input
            # Returns numpy array
            embeddings = self.model.encode([image])
            
            # Convert to torch tensor for pooling logic
            tensor = torch.tensor(embeddings[0])
            
            # Apply pooling (even if it's already 1D, to be safe and consistent)
            pooled_vector = pool_tokens(tensor)
            
            return pooled_vector
        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            return []
