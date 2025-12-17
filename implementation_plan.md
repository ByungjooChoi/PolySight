# Implementation Plan: Elastic PDF Visual Search Comparison

This document outlines the step-by-step plan to build the **Elastic PDF Visual Search Comparison** tool.
The goal is to demonstrate the difference between **Visual Search (VLM)** and **Text Search (OCR)** using Elastic Cloud Serverless.

## Phase 1: Environment & Project Initialization

### 1.1 Git Safety & Project Structure (CRITICAL)
* **Action:** Initialize the project and secure secrets.
* **Task:**
    1. Initialize `poetry` project.
    2. **Create `.gitignore` immediately:**
       - Must include: `.env`, `__pycache__`, `.DS_Store`, `venv/`, `.vscode/`.
    3. **Create `.env`:** This file will hold your REAL API keys.
       - `ELASTIC_CLOUD_SERVERLESS_URL=...`
       - `ELASTIC_API_KEY=...`
       - `JINA_API_KEY=...`
       - `GH_TOKEN=...` (if needed for internal tools)
    4. **Create `.env.example`:** Copy keys from `.env` but remove values (e.g., `JINA_API_KEY=your_key_here`). This file will be committed to Git.
    5. Create the directory structure defined in `architecture.md`.

### 1.2 Dependencies & Config
* **Action:** Install packages and set defaults.
* **Dependencies:**
    * **Core:** `elasticsearch` (latest), `python-dotenv`, `pyyaml`, `pydantic`.
    * **MCP:** `mcp`.
    * **AI/ML:** `torch`, `transformers`, `pillow`, `pdf2image`, `jina`, `streamlit`.
* **Config:** Create `config/config.yaml` with default settings:
    * `visual_model`: "jinaai/jina-embeddings-v4"
    * `token_pooling_ratio`: 0.3

### 1.3 Elastic Serverless Client Wrapper
* **File:** `backend/utils/elastic_client.py`
* **Task:**
    * Implement a singleton class.
    * Load secrets from `os.environ` (loaded via `python-dotenv`).
    * Implement `ensure_indices()`: Check/Create `visual-index` (`rank_vectors`) and `text-index` (`dense_vector`).

---

## Phase 2: MCP Server Development (The Bridge)

### 2.1 Fork & Clean Strategy
* **Action:** Replicate the core structure of `elastic/mcp-server-elasticsearch`.
* **Location:** `backend/mcp_server/`
* **Task:**
    * Copy SDK/Server logic.
    * **Remove** default tools to keep the project clean.

### 2.2 Tool Definition: Comparison Logic
* **File:** `backend/mcp_server/tools/comparison.py`
* **Tool Name:** `compare_search_results`
* **Logic:**
    1. Receive user query.
    2. **Visual Search:** Search `visual-index`.
    3. **Text Search:** Search `text-index`.
    4. **Compare:** Format top 3 hits from both into a Markdown table.

---

## Phase 3: Data Pipelines (The Engine)

### 3.1 Text Pipeline (OCR Adapter)
* **File:** `backend/pipelines/text_engine.py`
* **Task:**
    * Define `OCRBase` abstract class.
    * Implement `JinaReaderOCR` and `TextEmbedder`.

### 3.2 Visual Pipeline (VLM + Optimization)
* **File:** `backend/pipelines/visual_engine.py`
* **Task:**
    * **PDF to Image:** Use `pdf2image`.
    * **Visual Embedding:** Load Jina v4 model.
    * **Token Pooling:** Implement logic to keep top-k% tokens (using `torch`).

### 3.3 Ingestion Orchestrator
* **File:** `backend/pipelines/ingestion.py`
* **Task:**
    * Function `process_pdf(file_path)`: Run pipelines in parallel and Bulk Index.

---

## Phase 4: Frontend & UI (Streamlit)

### 4.1 Configuration Dashboard
* **File:** `frontend/app.py`
* **Features:**
    * **Sidebar:** API Key inputs (masked, updates `.env` in memory or session state).
    * **Settings:** Token Pooling Slider, Model Selection.
    * **Ingestion:** File Uploader -> Trigger `process_pdf`.

---

## Phase 5: Documentation & Guide Generation

### 5.1 Generate Kibana Setup Guide
* **Action:** Since the Kibana Agent is created manually in the UI, we need a guide.
* **File:** `docs/KIBANA_SETUP.md`
* **Task:** Create a detailed markdown guide containing:
    1. **Prerequisites:** Setup `ngrok` (if local) or deploy MCP server.
    2. **Agent Builder Steps:** Screenshots description or button-click path.
    3. **System Prompt Template:** (Provide the exact prompt text to copy-paste).
       > "You are a Visual Search Analyst. ALWAYS use `compare_search_results`..."
    4. **Connection Settings:** How to add the MCP Server URL.

### 5.2 README & Final Polish
* **File:** `README.md`
* **Task:** Write instructions on how to install, set `.env` keys, run the Streamlit app, and connect Kibana.