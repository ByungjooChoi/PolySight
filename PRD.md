# PolySight PRD (Product Requirements Document)

> 이 문서는 PolySight의 기능 요구사항 및 진행 상태를 추적합니다.
>
> - `[ ]` : 미완료
> - `[x]` : 완료
> - `[!]` : 차단됨 (수동 해결 필요)

---

## 프로젝트 개요

**PolySight**는 두 가지 차원의 프로젝트입니다:
1. **ColPali vs OCR 비교 (Agent Battle)** — Visual Agent vs Text Agent의 검색 결과 비교 (v1 완료)
2. **Morphik의 ES 버전** — Morphik(PostgreSQL+pgvector)이 하는 것을 Elasticsearch로 구현 (v2 진행 중)

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

## v1 완료 상태 요약

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

# v2: Morphik의 ES 버전 + Agent Battle 확장

## 프로젝트 비전

PolySight는 두 가지 차원으로 구성된다:
1. **ColPali vs OCR 비교 (Agent Battle)** — PolySight 고유의 차별점 (v1 완료)
2. **Morphik의 ES 버전** — Morphik이 PostgreSQL+pgvector로 하는 것을 Elasticsearch로 구현

## 기술 제약 사항 (확인 완료)

| 항목 | 결론 |
|------|------|
| `semantic_text` 필드 | sparse(ELSER) + dense 벡터만 지원. **rank_vectors 불가** |
| `rank_vectors` + RRF | `script_score`(maxSimDotProduct)를 `standard` retriever로 감싸면 RRF sub-retriever로 **사용 가능** |
| RRF sub-retriever 종류 | standard, kNN, sparse_vector, text_similarity_reranker |
| RRF 제약 | 최소 2개 sub-retriever 필요, search_after 불가, 커스텀 sort 불가 |

---

## Phase 8: Hybrid Search (RRF) — 최우선

### 목표
기존 2분할(Visual vs Text)에 **Hybrid(RRF 블렌딩)** 결과를 추가.
DevTools에서 직접 쿼리를 보여준 후 → UI에서 비교 데모.

### 8.1 RRF 쿼리 구현
- [x] `backend/utils/elastic_client.py`에 `search_hybrid_rrf()` 메서드 추가
- [x] RRF retriever 구조: `script_score`(MaxSim) + `multi_match`(BM25) sub-retrievers
- [x] 현재 인덱스가 분리(visual_index / text_index)되어 있으므로 **통합 인덱스 생성** 완료
  - ES RRF retriever는 하나의 인덱스 안에서 동작하므로 통합이 필수

### 8.2 통합 인덱스 설계
- [x] `polysight-unified` 인덱스 매핑: visual_vectors(rank_vectors) + text_content(text) + metadata
- [x] 인덱싱 파이프라인 수정: `process_image()`, `process_images_batch()`에 unified indexing 추가
- [x] 기존 분리 인덱스는 Agent Battle 비교용으로 유지

### 8.3 2-way RRF 쿼리
- [x] Sub-retriever 1: `script_score` + maxSimDotProduct (Visual)
- [x] Sub-retriever 2: `multi_match` BM25 (Text/Keyword)
- [x] `get_rrf_query_json()` — DevTools에서 복사/실행 가능한 JSON 생성
- [ ] (향후 필요 시 semantic_text 필드 추가 + reindex → 3-way 확장 가능)

### 8.4 UI 확장
- [x] Agent Battle 탭에 Hybrid 결과 패널 추가 (3분할 레이아웃)
- [x] 각 결과의 RRF 순위/스코어 표시
- [x] DevTools용 쿼리 복사 (Accordion + gr.Code)
- [x] RRF Rank Constant (k) 슬라이더 추가

### 8.5 데모 시나리오
- [ ] DevTools에서 RRF 쿼리 직접 실행 → "이게 ES의 Hybrid Search입니다"
- [ ] PolySight UI에서 같은 쿼리를 시각적으로 → "이걸 UI로 감싸면 이렇게 됩니다"
- [ ] MSG FAQ 데모처럼 단계별 진행: BM25만 → Visual만 → Hybrid RRF

---

## Phase 9: 다양한 문서 포맷 지원

### 목표
Morphik 수준의 문서 지원 (PDF 외 DOCX, PPTX, XLSX 등)

### 9.1 Office 문서 처리
- [x] python-docx → 텍스트 추출 + 페이지 이미지 변환 (DocxProcessor)
- [x] python-pptx → 슬라이드 이미지 변환 (PptxProcessor)
- [x] openpyxl → 시트 테이블 이미지 변환 (XlsxProcessor)
- [x] 공통 인터페이스: `DocumentProcessor.to_pages(file) → List[PIL.Image]`
- [x] `DocumentProcessor.extract_text(file)` — OCR 없이 직접 텍스트 추출

### 9.2 텍스트/데이터 파일
- [x] TXT, MD, HTML, XML, YAML, code files → TextFileProcessor
- [x] CSV/TSV → CsvProcessor (테이블 시각화 이미지)
- [x] JSON → JsonProcessor (list of dicts → 테이블, 그 외 → 포맷된 텍스트)

### 9.3 업로드 UI 확장
- [x] Gradio 파일 업로드에 36개 포맷 지원 추가
- [x] process_uploaded_file() → DocumentProcessor 자동 라우팅
- [x] Text Pipeline: Office/텍스트 문서는 OCR 대신 직접 추출 (_process_text_native)

---

## Phase 10: 비디오 처리

### 목표
Morphik의 비디오 검색 기능을 ES 버전으로 구현

### 10.1 비디오 → 프레임 추출
- [ ] ffmpeg로 N초 간격 프레임 추출
- [ ] 각 프레임을 PageImage로 변환 → Visual 임베딩

### 10.2 오디오 → 텍스트 전사
- [ ] Whisper API (또는 로컬 Whisper) 로 오디오 트랙 전사
- [ ] 타임스탬프 포함 텍스트 → Text 인덱싱

### 10.3 비디오 검색 UI
- [ ] "이 장면 찾아줘" → 타임스탬프 + 프레임 썸네일 반환
- [ ] 별도 "Video Search" 탭

---

## Phase 11: Agentic Search

### 목표
Gemini 대화에서 논의한 "Agent Battle Arena" — 두 Agent의 추론 과정 비교

### 11.1 LLM Tool Use 구현
- [x] Anthropic Claude API tool_use로 에이전트 루프 구현 (HTTP requests 기반)
- [x] Visual Agent: visual_search tool만 사용 가능
- [x] Text Agent: text_search tool만 사용 가능
- [x] Hybrid Agent: hybrid_search tool 사용
- [x] AgentBattleArena — 다중 에이전트 동시 실행 및 비교

### 11.2 추론 과정 스트리밍
- [x] 에이전트의 사고 과정(Thought Log)을 HTML 타임라인으로 표시
- [x] ThoughtStep 데이터클래스 (thinking/tool_call/tool_result/answer)
- [x] Generator 기반 run() + 동기 run_sync() 지원

### 11.3 Agent Battle 2.0
- [x] 3분할: Visual Agent vs Text Agent vs Hybrid Agent 추론 과정 비교
- [x] 검색 결과뿐 아니라 "왜 이 결과를 골랐는지" 사고 로그 표시
- [x] Agent Arena 탭 (Gradio UI) — 에이전트 선택 체크박스, 토큰/시간 통계

---

## Phase 12: 웹 데모 배포

### 목표
URL 부여해서 지속적으로 접근 가능한 데모 사이트

### 12.1 배포
- [x] GCP GPU VM 생성 (g2-standard-8, NVIDIA L4, us-west1-a, polysight-gpu)
- [x] 방화벽 규칙 설정 (TCP 7860 for Gradio, 태그: polysight)
- [x] API Key 보안 패턴 정립 (.env > config.json, .gitignore 적용)
- [x] GPU 서버 셋업 가이드 작성 (GPU_SERVER_SETUP.md)
- [ ] VM SSH 접속 후 PolySight 설치 및 실행
- [ ] ES Cloud Serverless 외부 연결 검증

### 12.2 인터랙티브 프레젠테이션 (별도 세션)
- [ ] Elastic 브랜드 가이드라인 적용한 소개 페이지
- [ ] 프로젝트 소개 + 인터랙티브 데모를 하나의 HTML로
- [ ] 참고: elastic-brand-skill.md, Ashish/David의 vibe-coded presso

---

## 전체 우선순위 요약

| 순서 | Phase | 핵심 | 예상 난이도 | 상태 |
|------|-------|------|-------------|------|
| — | Phase 1~7 | v1 Agent Battle | — | ✅ 완료 |
| 1 | **Phase 8: Hybrid Search (RRF)** | ES의 최대 강점 데모 | 중 | ✅ 구현 완료 (8.5 데모 시나리오 제외) |
| 2 | Phase 9: 다양한 문서 포맷 | Morphik 대응 | 하 | ✅ 구현 완료 |
| 3 | Phase 10: 비디오 처리 | Morphik 대응 | 중상 | 미착수 |
| 4 | Phase 11: Agentic Search | Agent Battle 2.0 | 상 | ✅ 구현 완료 |
| 5 | Phase 12: 웹 배포 | 데모 사이트 | 하 | 미착수 |

---

## 데모 시나리오 (최종)

### DevTools 데모 (5분)
1. BM25 검색 → "키워드만으로는 이 정도"
2. Visual 검색 (maxSimDotProduct) → "이미지를 직접 이해하면 이만큼"
3. Hybrid RRF → "ES가 알아서 합쳐주면 이렇게 됩니다"

### PolySight UI 데모 (10분)
1. Agent Battle: Visual vs OCR 2분할 → "차이가 보이시죠?"
2. Hybrid Search: Visual + BM25 RRF → "합치면 더 좋습니다"
3. (향후) Agentic Search → "에이전트가 직접 판단합니다"
