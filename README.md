# PolySight 👁️

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Agent Battle: Visual Search (Late Interaction) vs Text Search (BM25)**

## Overview

PolySight demonstrates the power of **Late Interaction** search using Jina V4's multi-vector embeddings compared to traditional OCR-based text search.

```
┌─────────────────┐     ┌─────────────────┐
│  Visual Agent   │     │   Text Agent    │
│  (Jina V4)      │ vs  │   (Docling)     │
├─────────────────┤     ├─────────────────┤
│ 이미지 → 벡터   │     │ 이미지 → OCR    │
│ Multi-vector    │     │ 텍스트 추출     │
│ Token Pooling   │     │ BM25 검색       │
│ MaxSim 검색     │     │                 │
└─────────────────┘     └─────────────────┘
```

## Key Features

- **Visual Agent**: Jina V4 multi-vector (128 dim) + Token Pooling + MaxSim (Late Interaction)
- **Text Agent**: Docling OCR + BM25 keyword search
- **Side-by-Side Comparison**: Real-time latency and result comparison
- **Elastic Cloud Serverless**: Uses `rank_vectors` for Late Interaction support

## Technology Stack

| Component | Technology |
|-----------|------------|
| Visual Embedding | Jina V4 (jinaai/jina-embeddings-v4) |
| Token Pooling | colpali_engine (HierarchicalTokenPooler) |
| Visual Search | MaxSim on `rank_vectors` (Elastic 9.0+) |
| OCR Engine | Docling |
| Text Search | BM25 |
| Frontend | Gradio |
| Backend | Elastic Cloud Serverless |

## Installation

### Prerequisites

- Python 3.10+
- Elastic Cloud Serverless Account (9.2+)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ByungjooChoi/PolySight.git
   cd PolySight
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   # Create virtual environment
   python3 -m venv venv

   # Activate virtual environment
   source venv/bin/activate  # macOS/Linux
   # venv\Scripts\activate   # Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your credentials:
   ```env
   ELASTIC_CLOUD_SERVERLESS_URL=https://your-deployment.es.region.aws.elastic.cloud
   ELASTIC_API_KEY=your-api-key
   ```

## Running PolySight

### 1. Start the Gradio UI

```bash
# Make sure venv is activated first
source venv/bin/activate  # macOS/Linux

# Run the app
python frontend/app.py
```

Open http://localhost:7860 in your browser.

**Features:**
- 🎯 **Search Battle**: Compare Visual vs Text search side-by-side
- 📤 **Ingest Documents**: Upload PDF or images to index
- ⚙️ **Settings**: View index stats, clear indices

### 2. Start the MCP Server (for Kibana Agent Builder)

```bash
python backend/mcp_server/server.py
```

**Available MCP Tools:**
- `compare_search_results(query)`: Compare both agents
- `search_visual_only(query)`: Visual Agent only
- `search_text_only(query)`: Text Agent only
- `get_index_status()`: Get index statistics

## How It Works

### Visual Agent Pipeline

```
Image → Jina V4 Encode → Multi-vector (128 dim)
                              ↓
                        Token Pooling (pool_factor=3)
                              ↓
                        Elastic (rank_vectors)
                              ↓
                        MaxSim Search (Late Interaction)
```

### Text Agent Pipeline

```
Image → Docling OCR → Text Extraction
                           ↓
                     Elastic (text field)
                           ↓
                     BM25 Search
```

## Elasticsearch Index Mappings

### Visual Index (`polysight-visual`)

```json
{
  "mappings": {
    "properties": {
      "visual_vectors": { "type": "rank_vectors" },
      "doc_id": { "type": "keyword" },
      "page_number": { "type": "integer" },
      "file_path": { "type": "keyword" }
    }
  }
}
```

### Text Index (`polysight-text`)

```json
{
  "mappings": {
    "properties": {
      "ocr_text": { "type": "text", "analyzer": "standard" },
      "doc_id": { "type": "keyword" },
      "page_number": { "type": "integer" },
      "file_path": { "type": "keyword" }
    }
  }
}
```

## Supported File Formats

| Format | Visual Agent | Text Agent |
|--------|--------------|------------|
| PDF | ✅ (pypdfium2) | ✅ (Docling) |
| PNG | ✅ | ✅ |
| JPG/JPEG | ✅ | ✅ |
| WEBP | ✅ | ✅ |
| TIFF | ✅ | ✅ |

## Demo Dataset

PolySight includes a loader for **ViDoRe Benchmark v3** from HuggingFace:

```python
from backend.data.vidore_loader import ViDoReLoader

loader = ViDoReLoader()
samples = loader.get_samples("test", num_samples=20)
```

## Project Structure

```
PolySight/
├── backend/
│   ├── pipelines/
│   │   ├── visual_engine.py    # Jina V4 + Token Pooling + MaxSim
│   │   ├── text_engine.py      # Docling OCR + BM25
│   │   └── ingestion.py        # Pipeline orchestration
│   ├── utils/
│   │   └── elastic_client.py   # Elastic Serverless client
│   ├── data/
│   │   └── vidore_loader.py    # ViDoRe dataset loader
│   └── mcp_server/
│       ├── server.py           # MCP Server entry point
│       └── tools/
│           └── comparison.py   # MCP tools
├── frontend/
│   └── app.py                  # Gradio UI
├── tests/
│   └── test_pipelines.py       # Unit tests
├── PRD.md                      # Task checklist
├── architecture.md             # System architecture
└── .env.example                # Environment template
```

## Testing

```bash
# Make sure venv is activated
source venv/bin/activate

# Run tests
pytest tests/ -v
```

## References

- [Elastic Labs - ColPali Token Pooling](https://github.com/elastic/elasticsearch-labs/tree/main/supporting-blog-content/colpali)
- [Elastic Blog - Late Interaction with ColPali](https://www.elastic.co/search-labs/blog/late-interaction-model-colpali-scale)
- [Jina V4 Documentation](https://jina.ai/embeddings/)

## License

Copyright 2025 Byungjoo Choi.
Licensed under the Apache License, Version 2.0.
