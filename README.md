# PolySight 👁️

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Enterprise Document Intelligence: Comparing OCR vs. Vision Language Models (VLM)**

## The Why

Traditional OCR often fails to capture the full context of complex documents. Charts, diagrams, and layout-dependent information are lost when converted to plain text. 

**PolySight** solves this by leveraging **Jina V4 (Vision Language Model)** to "see" the document. It enables a visual-first search experience that understands charts and layouts, providing a strategic advantage over text-only search.

## How It Works

PolySight processes PDFs through two parallel pipelines to demonstrate the difference:

1.  **Pipeline A (Visual):**
    *   Renders PDF pages as high-resolution images using `pypdfium2`.
    *   Generates visual embeddings using **Jina V4 Visual Encoder**.
    *   Stores results in Elastic Cloud Serverless (`rank_vectors`).

2.  **Pipeline B (Text):**
    *   Extracts text using **Jina Reader API**.
    *   Generates text embeddings using **Jina V4 Text Encoder**.
    *   Stores results in Elastic Cloud Serverless (`dense_vector`).

The **Comparison Agent** (MCP Server) then queries both indices and synthesizes a comparative analysis for the user.

## Installation & Usage

### Prerequisites
*   Python 3.10+
*   [Poetry](https://python-poetry.org/)
*   Elastic Cloud Serverless Account
*   Jina AI API Key

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ByungjooChoi/PolySight.git
    cd PolySight
    ```

2.  **Install dependencies:**
    ```bash
    poetry install
    ```

3.  **Configure Environment:**
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and add your API keys.

### Running PolySight

1.  **Start the Dashboard (Streamlit):**
    Upload PDFs and manage ingestion.
    ```bash
    poetry run streamlit run frontend/app.py
    ```

2.  **Start the Comparison Agent (MCP Server):**
    Connects to Kibana Agent Builder.
    ```bash
    poetry run python backend/mcp_server/run.py
    ```

## Roadmap

*   [ ] **Universal Format Support:** Support for Docx, PPTX, and Google Workspace files.
*   [ ] **Local Repository Crawling:** Automatically ingest files from local folders.
*   [ ] **Native Image Support:** Direct ingestion of JPG/PNG files.

## License

Copyright 2025 Byungjoo Choi.
Licensed under the Apache License, Version 2.0.
