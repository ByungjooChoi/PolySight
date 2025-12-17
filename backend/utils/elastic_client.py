import os
import logging
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

# Load env vars
load_dotenv()

class ElasticClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ElasticClient, cls).__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        url = os.getenv("ELASTIC_CLOUD_SERVERLESS_URL")
        api_key = os.getenv("ELASTIC_API_KEY")

        if not url or not api_key:
            logging.warning("ELASTIC_CLOUD_SERVERLESS_URL or ELASTIC_API_KEY not found in environment.")
            self.client = None
            return

        try:
            self.client = Elasticsearch(
                url,
                api_key=api_key
            )
            logging.info("Elasticsearch client initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize Elasticsearch client: {e}")
            self.client = None

    def get_client(self):
        return self.client

    def ensure_indices(self):
        if not self.client:
            logging.error("Elasticsearch client is not initialized.")
            return

        indices = {
            "visual-index": {
                "mappings": {
                    "properties": {
                        "rank_vectors": {
                            "type": "dense_vector",
                            "dims": 1024, 
                            "index": True,
                            "similarity": "cosine"
                        },
                         "content": {"type": "text"},
                         "page_number": {"type": "integer"},
                         "file_path": {"type": "keyword"}
                    }
                }
            },
            "text-index": {
                "mappings": {
                    "properties": {
                        "dense_vector": {
                            "type": "dense_vector",
                            "dims": 1024,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "content": {"type": "text"},
                        "page_number": {"type": "integer"},
                        "file_path": {"type": "keyword"}
                    }
                }
            }
        }
        
        for index_name, body in indices.items():
            try:
                if not self.client.indices.exists(index=index_name):
                    self.client.indices.create(index=index_name, body=body)
                    logging.info(f"Created index: {index_name}")
                else:
                    logging.info(f"Index exists: {index_name}")
            except Exception as e:
                logging.error(f"Error checking/creating index {index_name}: {e}")
