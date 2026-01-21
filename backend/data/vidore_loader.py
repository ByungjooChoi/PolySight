"""
ViDoRe Benchmark v3 Dataset Loader for PolySight
Downloads and prepares the dataset for demo purposes.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class ViDoReLoader:
    """
    Loader for ViDoRe Benchmark datasets from HuggingFace.

    ViDoRe V3 is a collection of datasets, not a single dataset.
    We use docvqa_test_subsampled for demo purposes (lightweight, good quality).

    Reference: https://huggingface.co/collections/vidore/vidore-benchmark-v3
    """

    # Use docvqa_test_subsampled - lightweight and good for demos
    DATASET_NAME = "vidore/docvqa_test_subsampled"

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize ViDoRe loader.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "vidore_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self._dataset = None

    @property
    def dataset(self):
        """Lazy load dataset"""
        if self._dataset is None:
            self._dataset = self._load_dataset()
        return self._dataset

    def _load_dataset(self):
        """Load dataset from HuggingFace with streaming support"""
        try:
            from datasets import load_dataset

            logger.info(f"Loading {self.DATASET_NAME} from HuggingFace...")

            # Try streaming first (no full download needed)
            try:
                dataset = load_dataset(
                    self.DATASET_NAME,
                    streaming=False,  # Need False for .select() but uses cache
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                logger.info(f"Dataset loaded successfully")
                return dataset
            except Exception as e:
                logger.warning(f"Standard load failed, trying without cache: {e}")
                # Fallback without cache
                dataset = load_dataset(
                    self.DATASET_NAME,
                    trust_remote_code=True
                )
                return dataset

        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def get_samples(
        self,
        split: str = "test",
        num_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get samples from the dataset.

        Args:
            split: Dataset split (train, test, validation)
            num_samples: Number of samples to return (None for all)

        Returns:
            List of sample dictionaries with image and metadata
        """
        if split not in self.dataset:
            available = list(self.dataset.keys())
            raise ValueError(f"Split '{split}' not found. Available: {available}")

        data = self.dataset[split]

        if num_samples:
            data = data.select(range(min(num_samples, len(data))))

        samples = []
        for item in data:
            sample = {
                "image": item.get("image"),  # PIL Image
                "query": item.get("query", ""),
                "doc_id": item.get("doc_id", ""),
                "page_id": item.get("page_id", 0),
                "metadata": {
                    k: v for k, v in item.items()
                    if k not in ["image", "query", "doc_id", "page_id"]
                }
            }
            samples.append(sample)

        logger.info(f"Retrieved {len(samples)} samples from {split} split")
        return samples

    def iterate_samples(
        self,
        split: str = "test",
        batch_size: int = 10
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Iterate through samples in batches.

        Args:
            split: Dataset split
            batch_size: Number of samples per batch

        Yields:
            Batches of sample dictionaries
        """
        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not found")

        data = self.dataset[split]
        total = len(data)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = data.select(range(start, end))

            samples = []
            for item in batch:
                sample = {
                    "image": item.get("image"),
                    "query": item.get("query", ""),
                    "doc_id": item.get("doc_id", ""),
                    "page_id": item.get("page_id", 0),
                    "metadata": {
                        k: v for k, v in item.items()
                        if k not in ["image", "query", "doc_id", "page_id"]
                    }
                }
                samples.append(sample)

            yield samples

    def save_images_to_disk(
        self,
        output_dir: str,
        split: str = "test",
        num_samples: Optional[int] = None
    ) -> List[str]:
        """
        Save dataset images to disk.

        Args:
            output_dir: Directory to save images
            split: Dataset split
            num_samples: Number of samples to save

        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        samples = self.get_samples(split, num_samples)
        saved_paths = []

        for i, sample in enumerate(samples):
            image = sample.get("image")
            if image is None:
                continue

            doc_id = sample.get("doc_id", f"doc_{i}")
            page_id = sample.get("page_id", 0)

            filename = f"{doc_id}_page_{page_id}.png"
            filepath = os.path.join(output_dir, filename)

            try:
                if isinstance(image, Image.Image):
                    image.save(filepath)
                    saved_paths.append(filepath)
                    logger.debug(f"Saved: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save image {filename}: {e}")

        logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths

    def get_queries(self, split: str = "test") -> List[str]:
        """
        Get all unique queries from the dataset.

        Args:
            split: Dataset split

        Returns:
            List of unique query strings
        """
        samples = self.get_samples(split)
        queries = list(set(
            sample["query"] for sample in samples
            if sample.get("query")
        ))
        return queries

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        info = {
            "name": self.DATASET_NAME,
            "splits": {},
            "cache_dir": self.cache_dir
        }

        for split_name, split_data in self.dataset.items():
            info["splits"][split_name] = {
                "num_samples": len(split_data),
                "features": list(split_data.features.keys())
            }

        return info


def load_vidore_demo_data(
    num_samples: int = 20,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to load demo data from ViDoRe.

    Args:
        num_samples: Number of samples to load
        output_dir: Optional directory to save images

    Returns:
        Dict with samples and metadata
    """
    loader = ViDoReLoader()

    samples = loader.get_samples("test", num_samples)
    queries = loader.get_queries("test")[:10]  # Sample queries

    result = {
        "samples": samples,
        "sample_queries": queries,
        "info": loader.get_dataset_info()
    }

    if output_dir:
        result["saved_paths"] = loader.save_images_to_disk(
            output_dir, "test", num_samples
        )

    return result


if __name__ == "__main__":
    # Test loader
    logging.basicConfig(level=logging.INFO)

    print("Loading ViDoRe Benchmark v3...")
    loader = ViDoReLoader()

    info = loader.get_dataset_info()
    print(f"\nDataset Info:")
    print(f"  Name: {info['name']}")
    for split, details in info['splits'].items():
        print(f"  {split}: {details['num_samples']} samples")

    # Get sample queries
    queries = loader.get_queries("test")[:5]
    print(f"\nSample queries:")
    for q in queries:
        print(f"  - {q}")
