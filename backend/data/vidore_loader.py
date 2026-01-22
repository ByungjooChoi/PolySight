"""
ViDoRe Benchmark v3 Dataset Loader for PolySight
Downloads and prepares the dataset for demo purposes.

Loads 100 samples from each of the 8 public V3 datasets (800 total).
"""
import os
import logging
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class ViDoReLoader:
    """
    Loader for ViDoRe Benchmark V3 datasets from HuggingFace.

    ViDoRe V3 consists of 8 public datasets across enterprise domains.
    We sample 100 from each dataset for a balanced demo (800 total).

    Reference: https://huggingface.co/collections/vidore/vidore-benchmark-v3
    """

    # All 8 public V3 datasets
    V3_DATASETS = [
        "vidore/vidore_v3_hr",              # EU HR documents (EN)
        "vidore/vidore_v3_finance_en",       # Finance (EN)
        "vidore/vidore_v3_industrial",       # Aircraft tech docs (EN)
        "vidore/vidore_v3_pharmaceuticals",  # Pharma docs
        "vidore/vidore_v3_computer_science", # CS textbooks (EN)
        "vidore/vidore_v3_energy",           # Energy reports (FR)
        "vidore/vidore_v3_physics",          # Physics slides (FR)
        "vidore/vidore_v3_finance_fr",       # Finance (FR)
    ]

    # Samples per dataset
    SAMPLES_PER_DATASET = 100

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
        self._datasets = {}  # Cache loaded datasets
        self._combined_samples = None  # Cache combined samples

    def _load_single_dataset(self, dataset_name: str, config: str = "corpus"):
        """Load a single dataset from HuggingFace

        V3 datasets require config name: 'corpus', 'queries', 'qrels', etc.
        """
        from datasets import load_dataset

        cache_key = f"{dataset_name}_{config}"
        if cache_key in self._datasets:
            return self._datasets[cache_key]

        logger.info(f"Loading {dataset_name} config={config}...")
        try:
            # V3 datasets need config name, no trust_remote_code
            dataset = load_dataset(
                dataset_name,
                config,  # e.g., "corpus", "queries"
                cache_dir=self.cache_dir
            )
            self._datasets[cache_key] = dataset
            logger.info(f"Loaded {dataset_name}/{config} successfully")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}/{config}: {e}")
            raise

    def get_samples(
        self,
        split: str = "test",
        num_samples: Optional[int] = None,
        samples_per_dataset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get samples from all V3 datasets.

        Args:
            split: Dataset split (V3 corpus uses 'test')
            num_samples: Total number of samples (overrides samples_per_dataset)
            samples_per_dataset: Samples from each dataset (default: SAMPLES_PER_DATASET)

        Returns:
            List of sample dictionaries with image and metadata
        """
        if samples_per_dataset is None:
            samples_per_dataset = self.SAMPLES_PER_DATASET

        # If num_samples specified, calculate per-dataset amount
        if num_samples is not None:
            samples_per_dataset = max(1, num_samples // len(self.V3_DATASETS))

        all_samples = []

        for dataset_name in self.V3_DATASETS:
            try:
                # V3: load 'corpus' config, use 'test' split
                dataset = self._load_single_dataset(dataset_name, config="corpus")
                data = dataset["test"]  # V3 corpus uses 'test' split

                # Get domain name from dataset name (e.g., "vidore_v3_hr" -> "hr")
                domain = dataset_name.split("_")[-1]
                if dataset_name.endswith("_en") or dataset_name.endswith("_fr"):
                    domain = "_".join(dataset_name.split("_")[-2:])

                # Sample from this dataset
                n_samples = min(samples_per_dataset, len(data))
                subset = data.select(range(n_samples))

                for idx, item in enumerate(subset):
                    # V3 corpus fields: image, doc_id, page_num, markdown, corpus_id
                    image = item.get("image")
                    doc_id = item.get("doc_id", f"{domain}_{idx}")
                    page_num = item.get("page_num", 0)

                    sample = {
                        "image": image,
                        "query": "",  # Corpus doesn't have queries
                        "doc_id": f"v3_{domain}_{doc_id}",
                        "page_id": page_num,
                        "domain": domain,
                        "source_dataset": dataset_name,
                        "metadata": {
                            k: v for k, v in item.items()
                            if k not in ["image", "doc_id", "page_num"]
                        }
                    }
                    all_samples.append(sample)

                logger.info(f"Loaded {n_samples} samples from {dataset_name}")

            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
                continue

        logger.info(f"Total: {len(all_samples)} samples from {len(self.V3_DATASETS)} datasets")
        return all_samples

    def iterate_samples(
        self,
        split: str = "test",
        batch_size: int = 10,
        samples_per_dataset: Optional[int] = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Iterate through samples in batches.

        Args:
            split: Dataset split (ignored - auto-detected)
            batch_size: Number of samples per batch
            samples_per_dataset: Samples from each dataset

        Yields:
            Batches of sample dictionaries
        """
        # Get all samples first
        all_samples = self.get_samples(
            split=split,
            samples_per_dataset=samples_per_dataset
        )

        # Yield in batches
        for start in range(0, len(all_samples), batch_size):
            end = min(start + batch_size, len(all_samples))
            yield all_samples[start:end]

    def save_images_to_disk(
        self,
        output_dir: str,
        split: str = "test",
        samples_per_dataset: Optional[int] = None
    ) -> List[str]:
        """
        Save dataset images to disk.

        Args:
            output_dir: Directory to save images
            split: Dataset split
            samples_per_dataset: Samples from each dataset

        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        samples = self.get_samples(split, samples_per_dataset=samples_per_dataset)
        saved_paths = []

        for i, sample in enumerate(samples):
            image = sample.get("image")
            if image is None:
                continue

            doc_id = sample.get("doc_id", f"doc_{i}")
            page_id = sample.get("page_id", 0)

            # Sanitize doc_id for filename
            safe_doc_id = str(doc_id).replace("/", "_").replace("\\", "_")
            filename = f"{safe_doc_id}_page_{page_id}.png"
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

    def get_queries(self, split: str = "test", samples_per_dataset: int = 10) -> List[str]:
        """
        Get sample queries from the datasets.

        Args:
            split: Dataset split
            samples_per_dataset: How many samples to check per dataset

        Returns:
            List of unique query strings
        """
        samples = self.get_samples(split, samples_per_dataset=samples_per_dataset)
        queries = list(set(
            sample["query"] for sample in samples
            if sample.get("query")
        ))
        return queries

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the V3 datasets"""
        info = {
            "name": "ViDoRe Benchmark V3",
            "datasets": self.V3_DATASETS,
            "samples_per_dataset": self.SAMPLES_PER_DATASET,
            "total_datasets": len(self.V3_DATASETS),
            "cache_dir": self.cache_dir,
            "loaded": {}
        }

        for dataset_name in self.V3_DATASETS:
            cache_key = f"{dataset_name}_corpus"
            if cache_key in self._datasets:
                dataset = self._datasets[cache_key]
                info["loaded"][dataset_name] = {
                    "split": "test",
                    "num_samples": len(dataset["test"])
                }

        return info


def load_vidore_demo_data(
    samples_per_dataset: int = 100,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to load demo data from ViDoRe V3.

    Args:
        samples_per_dataset: Samples from each of 8 datasets (default: 100)
        output_dir: Optional directory to save images

    Returns:
        Dict with samples and metadata
    """
    loader = ViDoReLoader()

    samples = loader.get_samples(samples_per_dataset=samples_per_dataset)
    queries = loader.get_queries()[:10]  # Sample queries

    result = {
        "samples": samples,
        "sample_queries": queries,
        "info": loader.get_dataset_info()
    }

    if output_dir:
        result["saved_paths"] = loader.save_images_to_disk(
            output_dir, samples_per_dataset=samples_per_dataset
        )

    return result


if __name__ == "__main__":
    # Test loader
    logging.basicConfig(level=logging.INFO)

    print("Loading ViDoRe Benchmark V3...")
    print(f"8 datasets, {ViDoReLoader.SAMPLES_PER_DATASET} samples each\n")

    loader = ViDoReLoader()

    # Load small sample for testing
    samples = loader.get_samples(samples_per_dataset=5)
    print(f"\nLoaded {len(samples)} samples total")

    # Show samples per domain
    domains = {}
    for s in samples:
        domain = s.get("domain", "unknown")
        domains[domain] = domains.get(domain, 0) + 1

    print("\nSamples by domain:")
    for domain, count in sorted(domains.items()):
        print(f"  {domain}: {count}")

    # Get sample queries
    queries = [s["query"] for s in samples if s.get("query")][:5]
    print(f"\nSample queries:")
    for q in queries:
        print(f"  - {q[:80]}..." if len(q) > 80 else f"  - {q}")
