"""
Text Agent Engine for PolySight
- Docling OCR for text extraction
- BM25 search (no embedding, pure text search)
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class OCRBase(ABC):
    """Abstract base class for OCR engines"""

    @abstractmethod
    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from a PIL Image"""
        pass

    @abstractmethod
    def extract_text_from_file(self, file_path: str) -> List[str]:
        """Extract text from a file (PDF or image), returns list of texts per page"""
        pass


def get_best_device() -> str:
    """
    Auto-detect the best available compute device.
    Cross-platform support: CUDA (NVIDIA), MPS (Apple Silicon), CPU (fallback)

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    try:
        import torch

        # Check CUDA first (Windows/Linux with NVIDIA GPU)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available: {device_name}")
            return "cuda"

        # Check MPS (macOS with Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS available (Apple Silicon)")
            return "mps"

        # Fallback to CPU
        logger.info("No GPU available, using CPU")
        return "cpu"

    except ImportError:
        logger.warning("PyTorch not available, using CPU")
        return "cpu"
    except Exception as e:
        logger.warning(f"Device detection failed: {e}, using CPU")
        return "cpu"


class DoclingOCR(OCRBase):
    """
    Docling OCR Engine for text extraction.
    Cross-platform GPU acceleration: CUDA (Windows/Linux), MPS (macOS), CPU (fallback)
    """

    def __init__(self, languages: List[str] = None, use_gpu: bool = True, device: str = None):
        """
        Initialize Docling OCR with hardware acceleration.

        Args:
            languages: List of language codes (default: ["en"] for speed)
            use_gpu: Whether to use GPU acceleration (default: True)
            device: Force specific device ('cuda', 'mps', 'cpu') or None for auto-detect
        """
        # Default to English only for faster OCR (V3 datasets are EN/FR)
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self.device = device or (get_best_device() if use_gpu else "cpu")
        self._converter = None

    @property
    def converter(self):
        """Lazy load Docling converter with cross-platform GPU acceleration"""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.datamodel.base_models import InputFormat

                # Configure pipeline options
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = True

                # Try to configure hardware acceleration
                accelerator_options = None
                try:
                    from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice

                    # Map device string to AcceleratorDevice
                    device_map = {
                        "cuda": AcceleratorDevice.CUDA,
                        "mps": AcceleratorDevice.MPS,
                        "cpu": AcceleratorDevice.CPU
                    }
                    accel_device = device_map.get(self.device, AcceleratorDevice.CPU)

                    # Get optimal thread count based on platform
                    import os
                    num_threads = min(os.cpu_count() or 4, 8)

                    accelerator_options = AcceleratorOptions(
                        device=accel_device,
                        num_threads=num_threads
                    )
                    # Apply accelerator to pipeline options
                    pipeline_options.accelerator_options = accelerator_options
                    logger.info(f"Accelerator: device={self.device}, threads={num_threads}")

                except ImportError:
                    logger.warning("AcceleratorOptions not available, using defaults")
                except AttributeError:
                    # Older docling version doesn't support accelerator_options on pipeline
                    logger.warning("Pipeline accelerator_options not supported, using defaults")
                except Exception as e:
                    logger.warning(f"Failed to configure accelerator: {e}")

                # Try RapidOCR with torch backend for better GPU support
                try:
                    from docling.datamodel.pipeline_options import RapidOcrOptions

                    # Use torch backend for GPU acceleration (CUDA/MPS)
                    backend = "torch" if self.device in ["cuda", "mps"] else "onnxruntime"
                    pipeline_options.ocr_options = RapidOcrOptions(
                        lang=self.languages,
                    )
                    logger.info(f"RapidOCR: languages={self.languages}")

                except ImportError:
                    logger.info("RapidOcrOptions not available, using defaults")
                except Exception as e:
                    logger.warning(f"Failed to configure OCR options: {e}")

                # Create converter - try different API styles
                try:
                    # Try format_options style (newer API)
                    from docling.datamodel.pipeline_options import ImagePipelineOptions
                    image_options = ImagePipelineOptions()
                    image_options.do_ocr = True

                    self._converter = DocumentConverter(
                        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE],
                        format_options={
                            InputFormat.PDF: pipeline_options,
                            InputFormat.IMAGE: image_options,
                        }
                    )
                    logger.info("Using DocumentConverter with format_options")
                except (TypeError, ImportError) as e:
                    # Fallback: try without format_options
                    try:
                        self._converter = DocumentConverter(
                            allowed_formats=[InputFormat.PDF, InputFormat.IMAGE]
                        )
                        logger.info("Using basic DocumentConverter")
                    except TypeError:
                        # Last resort: minimal converter
                        self._converter = DocumentConverter()
                        logger.info("Using default DocumentConverter")

                logger.info(f"Docling OCR initialized: device={self.device}")

            except ImportError as e:
                logger.error(f"Docling not installed: {e}")
                raise ImportError("Please install docling: pip install docling")

        return self._converter

    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from a PIL Image using Docling.

        Args:
            image: PIL Image object

        Returns:
            Extracted text string
        """
        try:
            import tempfile
            import os

            # Save image to temp file (Docling requires file path)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name

            try:
                # Convert using Docling
                result = self.converter.convert(tmp_path)

                # Extract text from result
                text = self._extract_text_from_result(result)
                return text.strip()

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Docling OCR failed for image: {e}")
            return ""

    def _extract_text_from_result(self, result) -> str:
        """Extract text from Docling conversion result."""
        if hasattr(result, 'document') and hasattr(result.document, 'export_to_markdown'):
            return result.document.export_to_markdown()
        elif hasattr(result, 'text'):
            return result.text
        else:
            return str(result)

    def extract_texts_from_images_batch(self, images: List[Image.Image]) -> List[str]:
        """
        Extract text from multiple PIL Images using Docling batch processing.
        More efficient than processing one by one.

        Args:
            images: List of PIL Image objects

        Returns:
            List of extracted text strings
        """
        if not images:
            return []

        import tempfile
        import os

        temp_dir = tempfile.mkdtemp()
        temp_paths = []
        texts = []

        try:
            # Save all images to temp files
            for i, image in enumerate(images):
                tmp_path = os.path.join(temp_dir, f"img_{i:04d}.png")
                image.save(tmp_path)
                temp_paths.append(tmp_path)

            # Batch convert using convert_all
            try:
                results = self.converter.convert_all(temp_paths, raises_on_error=False)

                for result in results:
                    try:
                        text = self._extract_text_from_result(result)
                        texts.append(text.strip())
                    except Exception as e:
                        logger.warning(f"Failed to extract text from result: {e}")
                        texts.append("")

            except AttributeError:
                # Fallback if convert_all not available
                logger.info("convert_all not available, falling back to sequential processing")
                for tmp_path in temp_paths:
                    try:
                        result = self.converter.convert(tmp_path)
                        text = self._extract_text_from_result(result)
                        texts.append(text.strip())
                    except Exception as e:
                        logger.warning(f"Failed to process {tmp_path}: {e}")
                        texts.append("")

        finally:
            # Clean up temp files
            for tmp_path in temp_paths:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

        return texts

    def extract_text_from_file(self, file_path: str) -> List[str]:
        """
        Extract text from a file (PDF or image).

        Args:
            file_path: Path to the file

        Returns:
            List of text strings (one per page for PDF, single item for image)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = Path(file_path).suffix.lower()

        try:
            if ext == ".pdf":
                return self._extract_from_pdf(file_path)
            elif ext in [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".gif"]:
                image = Image.open(file_path).convert("RGB")
                text = self.extract_text_from_image(image)
                return [text]
            else:
                raise ValueError(f"Unsupported file type: {ext}")

        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            return []

    def _extract_from_pdf(self, file_path: str) -> List[str]:
        """
        Extract text from PDF file page by page.

        Args:
            file_path: Path to PDF file

        Returns:
            List of text strings (one per page)
        """
        try:
            result = self.converter.convert(file_path)

            # Try to get page-by-page text
            texts = []

            if hasattr(result, 'document'):
                doc = result.document

                # Try to get text by page
                if hasattr(doc, 'pages'):
                    for page in doc.pages:
                        if hasattr(page, 'export_to_markdown'):
                            texts.append(page.export_to_markdown())
                        elif hasattr(page, 'text'):
                            texts.append(page.text)
                        else:
                            texts.append(str(page))

                # Fallback: get full document text
                if not texts:
                    if hasattr(doc, 'export_to_markdown'):
                        full_text = doc.export_to_markdown()
                        texts = [full_text]
                    elif hasattr(doc, 'text'):
                        texts = [doc.text]

            # Final fallback
            if not texts:
                texts = [str(result)]

            return texts

        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return []


class SimpleOCR(OCRBase):
    """
    Simple fallback OCR using pytesseract.
    Used when Docling is not available.
    """

    def __init__(self):
        self._tesseract_available = None

    @property
    def tesseract_available(self) -> bool:
        if self._tesseract_available is None:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self._tesseract_available = True
            except Exception:
                self._tesseract_available = False
        return self._tesseract_available

    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text using pytesseract"""
        if not self.tesseract_available:
            logger.warning("Tesseract not available")
            return ""

        try:
            import pytesseract
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""

    def extract_text_from_file(self, file_path: str) -> List[str]:
        """Extract text from file using pytesseract"""
        from backend.pipelines.visual_engine import process_uploaded_file

        try:
            images = process_uploaded_file(file_path)
            texts = []
            for img in images:
                text = self.extract_text_from_image(img)
                texts.append(text)
            return texts
        except Exception as e:
            logger.error(f"File extraction failed: {e}")
            return []


def get_ocr_engine(
    prefer_docling: bool = True,
    use_gpu: bool = True,
    device: str = None
) -> OCRBase:
    """
    Factory function to get the best available OCR engine.
    Cross-platform: auto-detects CUDA, MPS, or CPU.

    Args:
        prefer_docling: Whether to prefer Docling over SimpleOCR
        use_gpu: Whether to use GPU acceleration (CUDA/MPS)
        device: Force specific device ('cuda', 'mps', 'cpu') or None for auto

    Returns:
        OCR engine instance
    """
    if prefer_docling:
        try:
            return DoclingOCR(use_gpu=use_gpu, device=device)
        except ImportError:
            logger.warning("Docling not available, falling back to SimpleOCR")

    return SimpleOCR()


# ========== Text Agent Class ==========

class TextAgent:
    """
    Text Agent for PolySight.
    Extracts text using OCR and searches using BM25.
    """

    def __init__(self, ocr_engine: Optional[OCRBase] = None):
        """
        Initialize Text Agent.

        Args:
            ocr_engine: OCR engine to use (default: Docling)
        """
        self.ocr = ocr_engine or get_ocr_engine()

    def extract_text(self, image: Image.Image) -> str:
        """Extract text from an image"""
        return self.ocr.extract_text_from_image(image)

    def extract_texts_from_file(self, file_path: str) -> List[str]:
        """Extract texts from a file (one per page)"""
        return self.ocr.extract_text_from_file(file_path)

    def process_images(self, images: List[Image.Image], use_batch: bool = True) -> List[str]:
        """
        Process multiple images and extract text from each.
        Uses batch processing for better performance when available.

        Args:
            images: List of PIL Images
            use_batch: Whether to use batch processing (default: True)

        Returns:
            List of extracted texts
        """
        if not images:
            return []

        # Try batch processing for DoclingOCR
        if use_batch and hasattr(self.ocr, 'extract_texts_from_images_batch'):
            logger.info(f"Processing {len(images)} images in batch mode")
            texts = self.ocr.extract_texts_from_images_batch(images)
            for i, text in enumerate(texts):
                logger.debug(f"Extracted {len(text)} chars from image {i}")
            return texts

        # Fallback to sequential processing
        logger.info(f"Processing {len(images)} images sequentially")
        texts = []
        for i, img in enumerate(images):
            text = self.extract_text(img)
            logger.debug(f"Extracted {len(text)} chars from image {i}")
            texts.append(text)
        return texts
