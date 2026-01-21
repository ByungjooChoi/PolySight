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


class DoclingOCR(OCRBase):
    """
    Docling OCR Engine for text extraction.
    Uses EasyOCR backend with support for English and Korean.
    """

    def __init__(self, languages: List[str] = None):
        """
        Initialize Docling OCR.

        Args:
            languages: List of language codes (default: ["en", "ko"])
        """
        self.languages = languages or ["en", "ko"]
        self._converter = None

    @property
    def converter(self):
        """Lazy load Docling converter"""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.datamodel.base_models import InputFormat

                # Configure pipeline options
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = True
                pipeline_options.ocr_options = {
                    "lang": self.languages
                }

                self._converter = DocumentConverter(
                    allowed_formats=[InputFormat.PDF, InputFormat.IMAGE]
                )
                logger.info(f"Docling OCR initialized with languages: {self.languages}")

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
                if hasattr(result, 'document') and hasattr(result.document, 'export_to_markdown'):
                    text = result.document.export_to_markdown()
                elif hasattr(result, 'text'):
                    text = result.text
                else:
                    text = str(result)

                return text.strip()

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Docling OCR failed for image: {e}")
            return ""

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


def get_ocr_engine(prefer_docling: bool = True) -> OCRBase:
    """
    Factory function to get the best available OCR engine.

    Args:
        prefer_docling: Whether to prefer Docling over SimpleOCR

    Returns:
        OCR engine instance
    """
    if prefer_docling:
        try:
            return DoclingOCR()
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

    def process_images(self, images: List[Image.Image]) -> List[str]:
        """
        Process multiple images and extract text from each.

        Args:
            images: List of PIL Images

        Returns:
            List of extracted texts
        """
        texts = []
        for i, img in enumerate(images):
            text = self.extract_text(img)
            logger.debug(f"Extracted {len(text)} chars from image {i}")
            texts.append(text)
        return texts
