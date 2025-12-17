import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.pipelines.text_engine import JinaReaderOCR, TextEmbedder
    from backend.pipelines.visual_engine import PDFProcessor, VisualEmbedder
    from backend.pipelines.ingestion import IngestionManager
    
    print("Successfully imported all pipeline modules.")
    
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
