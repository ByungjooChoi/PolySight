"""
Elasticsearch Client for PolySight
- Elastic Cloud Serverless (9.2+)
- rank_vectors for Late Interaction (MaxSim)
- text field for BM25 search
"""
import os
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ElasticClient:
    """
    Singleton Elasticsearch client for PolySight.
    Supports both Visual Agent (MaxSim) and Text Agent (BM25).
    """
    _instance = None

    # Index names
    VISUAL_INDEX = "polysight-visual"
    TEXT_INDEX = "polysight-text"
    UNIFIED_INDEX = "polysight-unified"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ElasticClient, cls).__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        """Initialize Elasticsearch client (config.json > .env)"""
        # Try config.json first, then fall back to .env
        try:
            from backend.utils.config_manager import get_config
            config = get_config()
            url = config.elastic_url or os.getenv("ELASTIC_CLOUD_SERVERLESS_URL")
            api_key = config.elastic_api_key or os.getenv("ELASTIC_API_KEY")
        except Exception:
            url = os.getenv("ELASTIC_CLOUD_SERVERLESS_URL")
            api_key = os.getenv("ELASTIC_API_KEY")

        if not url or not api_key:
            logger.warning("ELASTIC_CLOUD_SERVERLESS_URL or ELASTIC_API_KEY not found.")
            self.client = None
            return

        try:
            self.client = Elasticsearch(url, api_key=api_key)
            info = self.client.info()
            logger.info(f"Connected to Elasticsearch {info['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch client: {e}")
            self.client = None

    def get_client(self) -> Optional[Elasticsearch]:
        return self.client

    def ensure_indices(self):
        """Create indices if they don't exist"""
        if not self.client:
            logger.error("Elasticsearch client not initialized")
            return

        # Visual Index: rank_vectors for Late Interaction (MaxSim)
        visual_mapping = {
            "mappings": {
                "properties": {
                    "visual_vectors": {
                        "type": "rank_vectors"  # Elastic 9.0+ for Late Interaction
                    },
                    "doc_id": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "file_path": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "image_path": {"type": "keyword"},  # Path to saved image for preview
                    "indexed_at": {"type": "date"}
                }
            }
        }

        # Text Index: text field for BM25 search
        text_mapping = {
            "mappings": {
                "properties": {
                    "ocr_text": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "doc_id": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "file_path": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "image_path": {"type": "keyword"},  # Path to saved image for preview
                    "indexed_at": {"type": "date"}
                }
            }
        }

        # Unified Index: visual + text in one document for RRF Hybrid Search
        unified_mapping = {
            "mappings": {
                "properties": {
                    "visual_vectors": {
                        "type": "rank_vectors"  # Jina V4 multi-vector (pooled)
                    },
                    "text_content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "page_image": {
                        "type": "binary"  # Base64-encoded page image
                    },
                    "doc_id": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "source_file": {"type": "keyword"},
                    "file_name": {"type": "keyword"},
                    "image_path": {"type": "keyword"},
                    "indexed_at": {"type": "date"}
                }
            }
        }

        indices = {
            self.VISUAL_INDEX: visual_mapping,
            self.TEXT_INDEX: text_mapping,
            self.UNIFIED_INDEX: unified_mapping
        }

        for index_name, mapping in indices.items():
            try:
                if not self.client.indices.exists(index=index_name):
                    self.client.indices.create(index=index_name, body=mapping)
                    logger.info(f"Created index: {index_name}")
                else:
                    logger.info(f"Index already exists: {index_name}")
            except Exception as e:
                logger.error(f"Error creating index {index_name}: {e}")

    # ========== Visual Agent (MaxSim) ==========

    def index_visual(
        self,
        doc_id: str,
        visual_vectors: List[List[float]],
        page_number: int,
        file_path: str,
        file_name: str,
        image_path: Optional[str] = None
    ) -> bool:
        """
        Index multi-vectors for Visual Agent (Late Interaction).

        Args:
            doc_id: Unique document ID
            visual_vectors: List of vectors (multi-vector embedding)
            page_number: Page number (0-indexed)
            file_path: Original file path
            file_name: Original file name
            image_path: Path to saved image file for preview
        """
        if not self.client:
            logger.error("Elasticsearch client not initialized")
            return False

        try:
            doc = {
                "visual_vectors": visual_vectors,
                "doc_id": doc_id,
                "page_number": page_number,
                "file_path": file_path,
                "file_name": file_name,
                "image_path": image_path,
                "indexed_at": datetime.now(timezone.utc).isoformat()
            }

            self.client.index(
                index=self.VISUAL_INDEX,
                id=f"{doc_id}_page_{page_number}",
                document=doc
            )
            logger.debug(f"Indexed visual: {doc_id} page {page_number}")
            return True

        except Exception as e:
            logger.error(f"Failed to index visual document: {e}")
            return False

    def search_visual_maxsim(
        self,
        query_vectors: List[List[float]],
        size: int = 5,
        normalize_scores: bool = True,
        min_score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using MaxSim (Late Interaction) on rank_vectors.

        Args:
            query_vectors: Multi-vector query embedding
            size: Number of results to return
            normalize_scores: If True, normalize scores by query token count (default: True)
            min_score_threshold: Minimum normalized score threshold (0-1). Results below this are filtered.
                                If None, no threshold is applied.

        Returns:
            List of search results with scores (normalized if normalize_scores=True)
        """
        if not self.client:
            logger.error("Elasticsearch client not initialized")
            return []

        try:
            # Request more results if we're filtering by threshold
            fetch_size = size * 3 if min_score_threshold else size

            query = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "maxSimDotProduct(params.query_vector, 'visual_vectors')",
                            "params": {
                                "query_vector": query_vectors
                            }
                        }
                    }
                },
                "size": fetch_size,
                "_source": ["doc_id", "page_number", "file_path", "file_name", "image_path"],
                "explain": True  # Get scoring explanation
            }

            response = self.client.search(index=self.VISUAL_INDEX, body=query)

            # Calculate normalization factor
            num_query_tokens = len(query_vectors)

            results = []
            for hit in response["hits"]["hits"]:
                raw_score = hit["_score"]

                # Normalize score by query token count
                if normalize_scores and num_query_tokens > 0:
                    normalized_score = raw_score / num_query_tokens
                else:
                    normalized_score = raw_score

                # Apply threshold filter
                if min_score_threshold is not None and normalized_score < min_score_threshold:
                    continue

                # Extract explanation if available
                explanation = hit.get("_explanation", {})

                results.append({
                    "doc_id": hit["_source"]["doc_id"],
                    "page_number": hit["_source"]["page_number"],
                    "file_path": hit["_source"]["file_path"],
                    "file_name": hit["_source"]["file_name"],
                    "image_path": hit["_source"].get("image_path"),
                    "score": normalized_score,  # Return normalized score
                    "raw_score": raw_score,     # Keep raw score for debugging
                    "explanation": explanation
                })

                # Stop once we have enough results
                if len(results) >= size:
                    break

            logger.info(f"MaxSim search returned {len(results)} results "
                       f"(normalized={normalize_scores}, threshold={min_score_threshold})")
            return results

        except Exception as e:
            logger.error(f"MaxSim search failed: {e}")
            return []

    # ========== Text Agent (BM25) ==========

    def index_text(
        self,
        doc_id: str,
        ocr_text: str,
        page_number: int,
        file_path: str,
        file_name: str,
        image_path: Optional[str] = None
    ) -> bool:
        """
        Index OCR text for Text Agent (BM25 search).

        Args:
            doc_id: Unique document ID
            ocr_text: Extracted text from OCR
            page_number: Page number (0-indexed)
            file_path: Original file path
            file_name: Original file name
            image_path: Path to saved image file for preview
        """
        if not self.client:
            logger.error("Elasticsearch client not initialized")
            return False

        try:
            doc = {
                "ocr_text": ocr_text,
                "doc_id": doc_id,
                "page_number": page_number,
                "file_path": file_path,
                "file_name": file_name,
                "image_path": image_path,
                "indexed_at": datetime.now(timezone.utc).isoformat()
            }

            self.client.index(
                index=self.TEXT_INDEX,
                id=f"{doc_id}_page_{page_number}",
                document=doc
            )
            logger.debug(f"Indexed text: {doc_id} page {page_number}")
            return True

        except Exception as e:
            logger.error(f"Failed to index text document: {e}")
            return False

    def search_text_bm25(
        self,
        query_text: str,
        size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search using BM25 on text field.

        Args:
            query_text: Search query string
            size: Number of results to return

        Returns:
            List of search results with scores
        """
        if not self.client:
            logger.error("Elasticsearch client not initialized")
            return []

        try:
            query = {
                "query": {
                    "match": {
                        "ocr_text": query_text
                    }
                },
                "size": size,
                "_source": ["doc_id", "page_number", "file_path", "file_name", "ocr_text", "image_path"],
                "highlight": {
                    "fields": {
                        "ocr_text": {
                            "fragment_size": 150,
                            "number_of_fragments": 2,
                            "pre_tags": ["<mark>"],
                            "post_tags": ["</mark>"]
                        }
                    }
                },
                "explain": True  # Get BM25 scoring explanation
            }

            response = self.client.search(index=self.TEXT_INDEX, body=query)

            results = []
            for hit in response["hits"]["hits"]:
                # Get highlighted text if available
                highlights = hit.get("highlight", {}).get("ocr_text", [])
                highlight_text = " ... ".join(highlights) if highlights else ""

                # Extract explanation if available
                explanation = hit.get("_explanation", {})

                results.append({
                    "doc_id": hit["_source"]["doc_id"],
                    "page_number": hit["_source"]["page_number"],
                    "file_path": hit["_source"]["file_path"],
                    "file_name": hit["_source"]["file_name"],
                    "image_path": hit["_source"].get("image_path"),
                    "ocr_text": hit["_source"].get("ocr_text", "")[:500],  # Truncate for display
                    "highlight": highlight_text,
                    "score": hit["_score"],
                    "explanation": explanation
                })

            logger.info(f"BM25 search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    # ========== Unified Index (Hybrid / RRF) ==========

    def index_unified(
        self,
        doc_id: str,
        visual_vectors: Optional[List[List[float]]],
        text_content: Optional[str],
        page_number: int,
        source_file: str,
        file_name: str,
        image_path: Optional[str] = None,
        page_image_base64: Optional[str] = None
    ) -> bool:
        """
        Index a page into the unified index with both visual vectors and text.
        Enables RRF Hybrid Search across visual and text modalities.

        Args:
            doc_id: Unique document ID
            visual_vectors: Multi-vector embedding from Jina V4 (pooled)
            text_content: OCR-extracted text from Docling
            page_number: Page number (0-indexed)
            source_file: Original file path
            file_name: Original file name
            image_path: Path to saved image file for preview
            page_image_base64: Base64-encoded page image
        """
        if not self.client:
            logger.error("Elasticsearch client not initialized")
            return False

        try:
            doc = {
                "doc_id": doc_id,
                "page_number": page_number,
                "source_file": source_file,
                "file_name": file_name,
                "image_path": image_path,
                "indexed_at": datetime.now(timezone.utc).isoformat()
            }

            # Only include non-empty fields
            if visual_vectors:
                doc["visual_vectors"] = visual_vectors
            if text_content and text_content.strip():
                doc["text_content"] = text_content
            if page_image_base64:
                doc["page_image"] = page_image_base64

            self.client.index(
                index=self.UNIFIED_INDEX,
                id=f"{doc_id}_page_{page_number}",
                document=doc
            )
            logger.debug(f"Indexed unified: {doc_id} page {page_number}")
            return True

        except Exception as e:
            logger.error(f"Failed to index unified document: {e}")
            return False

    def search_hybrid_rrf(
        self,
        query_vectors: Optional[List[List[float]]] = None,
        query_text: Optional[str] = None,
        size: int = 5,
        rank_window_size: int = 50,
        rank_constant: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search using RRF (Reciprocal Rank Fusion) on the unified index.
        Combines Visual (MaxSim) and Text (BM25) sub-retrievers.

        Args:
            query_vectors: Multi-vector query embedding for visual search
            query_text: Text query for BM25 search
            size: Number of results to return
            rank_window_size: Window size for RRF ranking (default 50)
            rank_constant: RRF constant k (default 60, higher = more equal weighting)

        Returns:
            List of search results with RRF rank scores
        """
        if not self.client:
            logger.error("Elasticsearch client not initialized")
            return []

        if not query_vectors and not query_text:
            logger.error("At least one of query_vectors or query_text is required")
            return []

        try:
            # Build sub-retrievers
            retrievers = []

            if query_vectors:
                # Sub-retriever 1: Visual (MaxSim via script_score)
                visual_retriever = {
                    "standard": {
                        "query": {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "maxSimDotProduct(params.query_vector, 'visual_vectors')",
                                    "params": {
                                        "query_vector": query_vectors
                                    }
                                }
                            }
                        }
                    }
                }
                retrievers.append(visual_retriever)

            if query_text:
                # Sub-retriever 2: Text (BM25)
                text_retriever = {
                    "standard": {
                        "query": {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["text_content"]
                            }
                        }
                    }
                }
                retrievers.append(text_retriever)

            # Build RRF query
            rrf_query = {
                "retriever": {
                    "rrf": {
                        "retrievers": retrievers,
                        "rank_window_size": rank_window_size,
                        "rank_constant": rank_constant
                    }
                },
                "size": size,
                "_source": [
                    "doc_id", "page_number", "source_file", "file_name",
                    "text_content", "image_path"
                ]
            }

            response = self.client.search(index=self.UNIFIED_INDEX, body=rrf_query)

            results = []
            for rank, hit in enumerate(response["hits"]["hits"], 1):
                source = hit["_source"]
                text_snippet = source.get("text_content", "")
                if text_snippet and len(text_snippet) > 500:
                    text_snippet = text_snippet[:500] + "..."

                results.append({
                    "doc_id": source.get("doc_id", ""),
                    "page_number": source.get("page_number", 0),
                    "file_path": source.get("source_file", ""),
                    "file_name": source.get("file_name", ""),
                    "image_path": source.get("image_path"),
                    "text_content": text_snippet,
                    "score": hit.get("_score", 0),
                    "rrf_rank": rank,
                    "explanation": {"type": "rrf", "sub_retrievers": len(retrievers)}
                })

            logger.info(f"RRF Hybrid search returned {len(results)} results "
                       f"(sub-retrievers={len(retrievers)}, k={rank_constant})")
            return results

        except Exception as e:
            logger.error(f"RRF Hybrid search failed: {e}")
            return []

    def get_rrf_query_json(
        self,
        query_vectors: Optional[List[List[float]]] = None,
        query_text: Optional[str] = None,
        size: int = 5,
        rank_window_size: int = 50,
        rank_constant: int = 60
    ) -> str:
        """
        Generate the RRF query JSON for DevTools copy/paste.
        Used in demos to show the raw Elasticsearch query.
        """
        import json

        retrievers = []

        if query_vectors:
            retrievers.append({
                "standard": {
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "maxSimDotProduct(params.query_vector, 'visual_vectors')",
                                "params": {"query_vector": "<<QUERY_VECTORS>>"}
                            }
                        }
                    }
                }
            })

        if query_text:
            retrievers.append({
                "standard": {
                    "query": {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["text_content"]
                        }
                    }
                }
            })

        rrf_body = {
            "retriever": {
                "rrf": {
                    "retrievers": retrievers,
                    "rank_window_size": rank_window_size,
                    "rank_constant": rank_constant
                }
            },
            "size": size
        }

        query_json = json.dumps(rrf_body, indent=2, ensure_ascii=False)
        # Replace placeholder with note about actual vectors
        if query_vectors:
            query_json = query_json.replace(
                '"<<QUERY_VECTORS>>"',
                '"[... Jina V4 multi-vector ...]"'
            )
        return f"GET /polysight-unified/_search\n{query_json}"

    # ========== Utility Methods ==========

    def get_index_count(self, index_name: str) -> int:
        """Get document count for an index"""
        if not self.client:
            return 0

        try:
            response = self.client.count(index=index_name)
            return response["count"]
        except Exception as e:
            logger.error(f"Failed to get count for {index_name}: {e}")
            return 0

    def delete_index(self, index_name: str) -> bool:
        """Delete an index"""
        if not self.client:
            return False

        try:
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
                logger.info(f"Deleted index: {index_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {e}")
            return False

    def clear_all_indices(self) -> bool:
        """Clear all PolySight indices"""
        success = True
        for index in [self.VISUAL_INDEX, self.TEXT_INDEX, self.UNIFIED_INDEX]:
            if not self.delete_index(index):
                success = False
        return success
