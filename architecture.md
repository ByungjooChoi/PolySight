# PolySight Architecture

## 1. Project Overview

**PolySight**는 Visual Agent vs Text Agent의 "Agent Battle" 데모입니다.
동일한 쿼리에 대해 두 에이전트가 경쟁하여 결과를 비교합니다.

**핵심 비교:**
- **Visual Agent (Jina V4)**: 이미지를 직접 벡터화하여 Late Interaction 검색
- **Text Agent (Docling)**: OCR로 텍스트 추출 후 BM25 검색

**목표:** 문서 검색에서 VLM 기반 Visual Search가 전통적인 OCR + Text Search보다 얼마나 효과적인지 시각적으로 비교

## 2. System Architecture

### 2.1 Agent Battle 구조
```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio UI                                │
│  ┌─────────────┐  ┌─────────────────────────────────────────┐  │
│  │ Query Input │  │        Side-by-Side Results             │  │
│  └─────────────┘  │  ┌───────────┐    ┌───────────┐        │  │
│                   │  │  Visual   │    │   Text    │        │  │
│                   │  │  Agent    │ vs │  Agent    │        │  │
│                   │  │  Results  │    │  Results  │        │  │
│                   │  └───────────┘    └───────────┘        │  │
│                   └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend Server                             │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │     Visual Agent        │  │       Text Agent            │  │
│  │  ┌───────────────────┐  │  │  ┌───────────────────────┐  │  │
│  │  │ Jina V4 Embedding │  │  │  │ Docling OCR           │  │  │
│  │  │ (Multi-vector)    │  │  │  │ (텍스트 추출)         │  │  │
│  │  └─────────┬─────────┘  │  │  └───────────┬───────────┘  │  │
│  │            ▼            │  │              ▼              │  │
│  │  ┌───────────────────┐  │  │  ┌───────────────────────┐  │  │
│  │  │ Token Pooling     │  │  │  │ Text Indexing         │  │  │
│  │  │ (pool_factor=3)   │  │  │  │                       │  │  │
│  │  └─────────┬─────────┘  │  │  └───────────┬───────────┘  │  │
│  │            ▼            │  │              ▼              │  │
│  │  ┌───────────────────┐  │  │  ┌───────────────────────┐  │  │
│  │  │ MaxSim Search     │  │  │  │ BM25 Search           │  │  │
│  │  │ (rank_vectors)    │  │  │  │ (text field)          │  │  │
│  │  └───────────────────┘  │  │  └───────────────────────┘  │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Elastic Cloud Serverless (9.2+)                    │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │    visual-index         │  │       text-index            │  │
│  │  ┌───────────────────┐  │  │  ┌───────────────────────┐  │  │
│  │  │ visual_vectors    │  │  │  │ ocr_text              │  │  │
│  │  │ (rank_vectors)    │  │  │  │ (text)                │  │  │
│  │  └───────────────────┘  │  │  └───────────────────────┘  │  │
│  │  ┌───────────────────┐  │  │  ┌───────────────────────┐  │  │
│  │  │ image_path        │  │  │  │ image_path            │  │  │
│  │  └───────────────────┘  │  │  └───────────────────────┘  │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

#### A. Gradio UI (`frontend/app.py`)
- **Framework:** Gradio
- **Layout:** Side-by-side 비교 (`gr.Row()`, `gr.Column()`)
- **Features:**
  - 쿼리 입력
  - Visual Agent / Text Agent 결과 동시 표시
  - 검색 시간 (latency) 표시
  - 결과 이미지 + 스코어 표시

#### B. Visual Agent (`backend/pipelines/visual_engine.py`)
- **Embedding Model:** Jina V4 (Multi-vector, 128 dim)
- **Optimization:** Token Pooling (HierarchicalTokenPooler, pool_factor=3)
- **Search:** MaxSim (maxSimDotProduct)
- **Index Field:** `rank_vectors` 타입

#### C. Text Agent (`backend/pipelines/text_engine.py`)
- **OCR Engine:** Docling (EasyOCR 백엔드)
- **Search:** BM25 (Elasticsearch 기본 텍스트 검색)
- **Index Field:** `text` 타입

#### D. MCP Server (`backend/mcp_server/`)
- **Base:** Fork of `elastic/mcp-server-elasticsearch`
- **Role:** Kibana Agent Builder 연동
- **Tools:**
  - `compare_search_results(query)`: 양쪽 에이전트 검색 결과 비교

#### E. Data Store (Elastic Cloud Serverless)
- **Platform:** Elastic Cloud Serverless (9.2+)
- **Indices:**
  - `visual-index`: `rank_vectors` 매핑 (Late Interaction)
  - `text-index`: `text` 매핑 (BM25)

## 3. Directory Structure
```
PolySight/
├── backend/
│   ├── mcp_server/
│   │   ├── tools/
│   │   │   └── comparison.py       # compare_search_results tool
│   │   └── server.py
│   ├── pipelines/
│   │   ├── visual_engine.py        # Jina V4 + Token Pooling + MaxSim
│   │   ├── text_engine.py          # Docling OCR + BM25
│   │   └── ingestion.py            # 인덱싱 오케스트레이터
│   └── utils/
│       └── elastic_client.py       # Elastic Serverless 클라이언트
├── frontend/
│   └── app.py                      # Gradio UI
├── config/
│   ├── config.yaml
│   └── mappings/
│       ├── visual_index.json       # rank_vectors 매핑
│       └── text_index.json         # text 매핑
├── data/
│   └── vidore_v3/                  # ViDoRe Benchmark v3 데이터
├── .env                            # API Keys
├── PRD.md                          # 태스크 목록 (Ralph v3)
├── progress.md                     # 진행 로그
└── architecture.md                 # 이 문서
```

## 4. Index Mappings

### 4.1 Visual Index
```json
{
  "mappings": {
    "properties": {
      "visual_vectors": {
        "type": "rank_vectors"
      },
      "image_path": {
        "type": "keyword"
      },
      "doc_id": {
        "type": "keyword"
      }
    }
  }
}
```

### 4.2 Text Index
```json
{
  "mappings": {
    "properties": {
      "ocr_text": {
        "type": "text",
        "analyzer": "standard"
      },
      "image_path": {
        "type": "keyword"
      },
      "doc_id": {
        "type": "keyword"
      }
    }
  }
}
```

## 5. Search Queries

### 5.1 Visual Agent (MaxSim)
```python
es_query = {
    "query": {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "maxSimDotProduct(params.query_vector, 'visual_vectors')",
                "params": {"query_vector": query_multi_vectors}
            }
        }
    },
    "size": 5
}
```

### 5.2 Text Agent (BM25)
```python
es_query = {
    "query": {
        "match": {
            "ocr_text": query_text
        }
    },
    "size": 5
}
```

## 6. Data Pipeline

### 6.1 데이터 소스
```
┌─────────────────────────────────────────────────────────────┐
│                      Data Sources                           │
├─────────────────────────────────────────────────────────────┤
│  1. ViDoRe v3 Dataset (데모용 샘플)                         │
│     - HuggingFace에서 로드                                  │
│     - 벤치마크 데이터셋                                     │
│                                                             │
│  2. 사용자 파일 업로드 (커스텀 데이터)                      │
│     - PDF: pypdfium2로 페이지별 이미지 변환                 │
│     - 이미지: PNG, JPG, JPEG, WEBP, TIFF 직접 처리          │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 File Upload Processing
```python
def process_uploaded_file(file_path: str) -> List[Image.Image]:
    """파일 타입에 따라 이미지 리스트 반환"""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        # PDF → 페이지별 이미지
        return PDFProcessor.convert_to_images(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".tiff"]:
        # 이미지 → 단일 이미지 리스트
        return [Image.open(file_path)]
    else:
        raise ValueError(f"Unsupported file type: {ext}")
```

### 6.3 Indexing Flow
```
Data Source (ViDoRe v3 or User Upload)
            │
            ▼
    ┌───────────────┐
    │  Load Images  │
    └───────┬───────┘
            │
    ┌───────┴───────┐
    ▼               ▼
┌───────────┐  ┌───────────┐
│  Visual   │  │   Text    │
│  Pipeline │  │  Pipeline │
└─────┬─────┘  └─────┬─────┘
      │              │
      ▼              ▼
┌───────────┐  ┌───────────┐
│ Jina V4   │  │ Docling   │
│ Embedding │  │ OCR       │
└─────┬─────┘  └─────┬─────┘
      │              │
      ▼              │
┌───────────┐        │
│ Token     │        │
│ Pooling   │        │
└─────┬─────┘        │
      │              │
      ▼              ▼
┌───────────┐  ┌───────────┐
│ visual-   │  │ text-     │
│ index     │  │ index     │
└───────────┘  └───────────┘
```

### 6.4 Search Flow
```
User Query
    │
    ├────────────────────┐
    ▼                    ▼
┌───────────┐      ┌───────────┐
│ Jina V4   │      │ Query     │
│ Query     │      │ Text      │
│ Embedding │      │           │
└─────┬─────┘      └─────┬─────┘
      │                  │
      ▼                  ▼
┌───────────┐      ┌───────────┐
│ MaxSim    │      │ BM25      │
│ Search    │      │ Search    │
└─────┬─────┘      └─────┬─────┘
      │                  │
      └────────┬─────────┘
               ▼
      ┌─────────────────┐
      │ Compare Results │
      │ (Side-by-Side)  │
      └─────────────────┘
```

## 7. Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Gradio |
| Backend | Python |
| Visual Embedding | Jina V4 (jinaai/jina-embeddings-v4) |
| OCR | Docling (EasyOCR backend) |
| Vector Optimization | Token Pooling (colpali_engine) |
| Search Backend | Elastic Cloud Serverless |
| Visual Search | MaxSim (rank_vectors) |
| Text Search | BM25 |
| Dataset | ViDoRe Benchmark v3 |
| MCP Integration | Kibana Agent Builder |
