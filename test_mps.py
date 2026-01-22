#!/usr/bin/env python3
"""
Quick test to verify MPS/CUDA/CPU device detection and Docling OCR initialization.
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print("=" * 60)
print("PolySight Device Detection Test")
print("=" * 60)

# Test 1: PyTorch device detection
print("\n[1] PyTorch Device Detection:")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if hasattr(torch.backends, 'mps'):
        print(f"  MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print("  ✓ MPS (Apple Silicon) detected!")
    else:
        print("  MPS: Not supported in this PyTorch version")

except ImportError:
    print("  PyTorch not installed")

# Test 2: text_engine device detection
print("\n[2] Text Engine Device Detection:")
try:
    from backend.pipelines.text_engine import get_best_device, DoclingOCR

    device = get_best_device()
    print(f"  Best device detected: {device}")

    if device == "mps":
        print("  ✓ MPS will be used for OCR!")
    elif device == "cuda":
        print("  ✓ CUDA will be used for OCR!")
    else:
        print("  ⚠ CPU will be used (no GPU acceleration)")

except Exception as e:
    print(f"  Error: {e}")

# Test 3: Docling OCR initialization (this triggers lazy loading)
print("\n[3] Docling OCR Initialization:")
try:
    from backend.pipelines.text_engine import DoclingOCR

    ocr = DoclingOCR(use_gpu=True)
    print(f"  OCR device setting: {ocr.device}")

    # Force lazy loading by accessing converter
    print("  Initializing Docling converter (this may take a moment)...")
    try:
        _ = ocr.converter
        print("  ✓ Docling OCR initialized successfully!")
    except Exception as e:
        print(f"  ✗ Converter init failed: {e}")

    # Quick test with a simple image
    print("\n[4] Quick OCR Test:")
    try:
        from PIL import Image
        import io

        # Create a simple test image with text
        test_img = Image.new('RGB', (200, 50), color='white')
        text_result = ocr.extract_text_from_image(test_img)
        print(f"  Test image OCR result: '{text_result[:50]}...' (len={len(text_result)})")
        print("  ✓ OCR extraction working!")
    except Exception as e:
        print(f"  ✗ OCR test failed: {e}")

except ImportError as e:
    print(f"  Docling not installed: {e}")
except Exception as e:
    print(f"  Error initializing OCR: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
