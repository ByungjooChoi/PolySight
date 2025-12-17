# Elastic PDF Visual Search Comparison Architecture

## 1. Project Overview
This project is an **Elastic Cloud Serverless** native solution designed to demonstrate the difference between **Visual Language Model (VLM)** search and traditional **OCR + Text Embedding** search.

Users can upload PDF documents, which are processed through two distinct pipelines. The results are stored in Elastic Serverless and compared via the **Elastic Agent Builder** in Kibana.

**Key Features:**
* **Dual Pipeline:** Parallel processing of PDFs via Visual (Image-based) and Text (OCR-based) paths.
* **Independent Configuration:** Users can select different models for each pipeline via UI.
* **Optimization:** Configurable Token Pooling for VLM to manage vector scale.

## 2. System Architecture

### 2.1 High-Level Component Diagram
```text
[ Admin/Setup UI (Streamlit) ]      [ Kibana Agent Builder (User Search) ]
          | (Config & Ingest)                  | (Search Query)
          v                                    v
[ Backend Server (Python) & Custom MCP Server ]
          |
          +---> [ Pipeline A: Visual Search ] --------------------+
          |       - Input: Page Screenshots                       |
          |       - Model: Jina v4 / ColPali (Configurable)       |
          |       - Optimization: Token Pooling (Slider UI)       |
          |       - Output: rank_vectors (Elastic Visual Index)   |
          |                                                       |
          +---> [ Pipeline B: Text Search ] ----------------------+
                  - Input: OCR Text (Jina/Reducto/Unstructured)
                  - Model: Jina v4 / OpenAI (Configurable)
                  - Output: dense_vector (Elastic Text Index)
```
### 2.2 Core Components

#### A. Setup & Ingestion UI (`frontend/`)
* **Framework:** Streamlit.
* **Capabilities:**
    * **Pipeline A Config:** Select VLM Model (default: `jina-embeddings-v4`), Set Token Pooling Ratio (Slider 0-100%).
    * **Pipeline B Config:** Select OCR Engine (Dropdown), Select Text Embedding Model.
    * **Execution:** Upload PDF and trigger async ingestion.

#### B. MCP Server (`backend/mcp_server/`)
* **Base:** Fork of `elastic/mcp-server-elasticsearch`.
* **Role:** Bridges Elastic Agent Builder with the custom search logic.
* **Tools:**
    * `compare_search_results(query)`: Executes parallel searches on both indices and returns a comparative summary (Markdown table).

#### C. Data Store (Elastic Serverless)
* **Platform:** Elastic Cloud Serverless.
* **Indices:**
    * `visual-index`: Uses `rank_vectors` mapping for late-interaction (VLM).
    * `text-index`: Uses `dense_vector` mapping for KNN (Text).

## 3. Directory Structure
```text
elastic-visual-comparison/
├── backend/
│   ├── mcp_server/                 # [Forked] Based on elastic/mcp-server-elasticsearch
│   │   ├── tools/                  # Custom tools implementation
│   │   │   └── comparison.py       # compare_search_results tool
│   │   └── server.py               # Server entrypoint
│   ├── pipelines/
│   │   ├── visual_engine.py        # VLM inference & Token Pooling logic
│   │   ├── text_engine.py          # OCR adapters & Text embedding
│   │   └── ingestion.py            # Async orchestrator
│   └── utils/
│       └── elastic_client.py       # Elastic Serverless Client
├── frontend/
│   └── app.py                      # Streamlit Config Dashboard
├── config/
│   ├── config.yaml                 # Persisted settings from UI
│   └── mappings/                   # Index templates
├── .env                            # API Keys (masked in UI)
└── architecture.md
```