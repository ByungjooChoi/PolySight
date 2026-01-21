# PolySight PRD (Product Requirements Document)

> 이 문서는 PolySight의 기능 요구사항 및 진행 상태를 추적합니다.
>
> - `[ ]` : 미완료
> - `[x]` : 완료
> - `[!]` : 차단됨 (수동 해결 필요)

---

## 프로젝트 개요

**PolySight**는 Visual Agent vs Text Agent의 "Agent Battle" 데모입니다.
동일한 쿼리에 대해 두 에이전트가 경쟁하여 결과를 비교합니다.

### 핵심 기술 스택
- **Visual Agent**: Jina V4 Multi-vector (128 dim) + Late Interaction (MaxSim)
- **Text Agent**: Docling OCR → 텍스트 추출 → BM25/텍스트 검색
- **Backend**: Elastic Cloud Serverless (9.2+, `rank_vectors` 지원)
- **Frontend**: Gradio
- **최적화**: Token Pooling (pool_factor=3, float 유지)
- **데이터셋**: ViDoRe Benchmark v3

### Agent Battle 컨셉
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

### 참조 코드
- [elastic/elasticsearch-labs/colpali](https://github.com/elastic/elasticsearch-labs/tree/main/supporting-blog-content/colpali) - Token Pooling, MaxSim 구현
- [ByungjooChoi/colpali](https://github.com/ByungjooChoi/colpali) - MaxSim + Elastic 쿼리 패턴

---

## Phase 1: Text Agent (Docling OCR + BM25) ✅

### 1.1 Docling 통합
- [x] `pyproject.toml`에 `docling` 의존성 추가
- [x] `backend/pipelines/text_engine.py`에 `DoclingOCR` 클래스 구현
- [x] `OCRBase` 인터페이스에 맞춰 `extract_text()` 메서드 구현
- [x] Docling OCR 옵션 설정 (EasyOCR 백엔드, 언어: 영어+한국어)

### 1.2 Text 인덱싱 및 검색
- [x] Elastic 인덱스 매핑: `text` 타입 필드
- [x] OCR 추출 텍스트 인덱싱
- [x] BM25 텍스트 검색 쿼리 구현

---

## Phase 2: Visual Agent (Jina V4 Multi-vector + Late Interaction) ✅

### 2.1 Token Pooling 구현
- [x] `colpali_engine.compression.token_pooling.HierarchicalTokenPooler` 사용
- [x] `pool_factor=3` 설정 (벡터 수 1/3 감소)
- [x] `pool_vectors()` 함수 구현 (Elastic 공식 코드 참조)

### 2.2 Elastic 인덱스 설정 (rank_vectors)
- [x] Visual 인덱스 매핑 생성 (`rank_vectors` 타입)
- [x] Pooled multi-vector 인덱싱

### 2.3 MaxSim 검색 쿼리 구현
- [x] `maxSimDotProduct` 스크립트 쿼리 구현

---

## Phase 3: Agent Battle UI (Gradio) ✅

### 3.1 Gradio 앱 기본 구조
- [x] `frontend/app.py`를 Gradio로 전면 교체
- [x] Side-by-side 레이아웃 (`gr.Row()`, `gr.Column()`)
- [x] 쿼리 입력 컴포넌트

### 3.2 동시 검색 및 결과 비교
- [x] 동일 쿼리로 Visual Agent / Text Agent 동시 호출
- [x] 검색 결과 나란히 표시 (이미지 + 스코어)
- [x] 검색 시간 (latency) 표시
- [x] 결과 랭킹 비교 시각화

### 3.3 Elastic Cloud Serverless 연결
- [x] Cloud ID / API Key 인증 설정
- [x] 환경변수: `ELASTIC_CLOUD_SERVERLESS_URL`, `ELASTIC_API_KEY`

---

## Phase 4: 데이터 준비 (ViDoRe Benchmark v3 + 파일 업로드) ✅

### 4.1 데이터셋 다운로드 (데모용 샘플)
- [x] HuggingFace에서 ViDoRe v3 데이터셋 로드
- [x] `backend/data/vidore_loader.py` 구현
- [x] **UI에서 ViDoRe 샘플 로드 버튼** 추가
- [x] **중복 로드 방지 로직** 구현

### 4.2 파일 업로드 기능 (사용자 커스텀 데이터)
- [x] PDF 업로드: pypdfium2로 페이지별 이미지 변환
- [x] 이미지 직접 업로드: PNG, JPG, JPEG, WEBP, TIFF 지원
- [x] Gradio `gr.File()` 컴포넌트

### 4.3 인덱싱 파이프라인
- [x] Visual Agent: 이미지 → Jina V4 multi-vector → Token Pooling → Elastic (rank_vectors)
- [x] Text Agent: 이미지 → Docling OCR → 텍스트 → Elastic (text 필드)

---

## Phase 5: MCP Server (Kibana Agent Builder 연동) ✅

### 5.1 실제 Elastic 쿼리 구현
- [x] `backend/mcp_server/tools/comparison.py`의 Mock 데이터 제거
- [x] Visual Agent: MaxSim 쿼리 연동
- [x] Text Agent: BM25 쿼리 연동
- [x] 검색 결과를 Markdown 테이블로 포맷팅
- [x] 추가 도구: `get_index_status`, `search_visual_only`, `search_text_only`

### 5.2 에러 핸들링
- [x] Elastic 연결 실패 시 graceful 에러 메시지
- [x] 인덱스가 비어있을 때 처리

---

## Phase 6: 테스트 & 문서화 ✅

### 6.1 테스트
- [x] Token Pooling 단위 테스트
- [x] MaxSim 쿼리 테스트
- [x] MCP Tools 테스트
- [x] Import 테스트

### 6.2 문서화
- [x] README 업데이트 (설치, 환경변수, 실행 방법)
- [x] Elastic Cloud Serverless 설정 가이드
- [x] 프로젝트 구조 문서화

---

## Phase 7: 설정 UI 및 Jina 모드 선택 ✅ (신규)

### 7.1 Settings UI
- [x] Elasticsearch URL/API Key 입력 폼
- [x] Jina API Key 입력 폼 (선택)
- [x] HuggingFace Token 입력 폼 (선택)
- [x] 연결 테스트 버튼 (Elastic, Jina)
- [x] 설정 저장 버튼 (`config.json` 저장)

### 7.2 Jina V4 로컬/API 모드 선택
- [x] **로컬 모드 (기본)**: GPU 권장, 무료
- [x] **API 모드**: Jina API Key 입력 시 자동 전환, GPU 불필요
- [x] `JinaAPIClient` 클래스 구현
- [x] `VisualEmbedder`에 모드 선택 로직 추가

### 7.3 설정 관리 시스템
- [x] `backend/utils/config_manager.py` 구현
- [x] 우선순위: `config.json` > `.env` > 기본값
- [x] 앱 시작 시 환경 설정 배너 표시 (미설정 시)
- [x] 친절한 에러 메시지 (설정 누락 시)

---

## 완료 상태 요약

| Phase | 설명 | 상태 |
|-------|------|------|
| Phase 1 | Text Agent (Docling + BM25) | ✅ 완료 |
| Phase 2 | Visual Agent (Jina V4 + MaxSim) | ✅ 완료 |
| Phase 3 | Agent Battle UI (Gradio) | ✅ 완료 |
| Phase 4 | 데이터 준비 (ViDoRe + 업로드) | ✅ 완료 |
| Phase 5 | MCP Server | ✅ 완료 |
| Phase 6 | 테스트 & 문서화 | ✅ 완료 |
| Phase 7 | Settings UI & Jina 모드 | ✅ 완료 |

---

## 다음 단계 (Future Work)

- [ ] 검색 결과에 이미지 썸네일 표시
- [ ] 검색 품질 메트릭 (MRR, NDCG) 표시
- [ ] 다국어 쿼리 지원 개선
- [ ] 배치 인덱싱 성능 최적화
