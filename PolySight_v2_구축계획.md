# PolySight v2 구축 계획: Morphik의 ES 버전 + Agent Battle

## 프로젝트 비전

PolySight는 두 가지 차원으로 구성된다:
1. **ColPali vs OCR 비교 (Agent Battle)** — PolySight 고유의 차별점 (v1 완료)
2. **Morphik의 ES 버전** — Morphik이 PostgreSQL+pgvector로 하는 것을 Elasticsearch로 구현

---

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
- [ ] `backend/utils/elastic_client.py`에 `hybrid_search()` 메서드 추가
- [ ] RRF retriever 구조:
  ```json
  {
    "retriever": {
      "rrf": {
        "retrievers": [
          {
            "standard": {
              "query": {
                "script_score": {
                  "query": { "match_all": {} },
                  "script": {
                    "source": "maxSimDotProduct(params.query_vector, 'visual_embedding')",
                    "params": { "query_vector": "..." }
                  }
                }
              }
            }
          },
          {
            "standard": {
              "query": {
                "multi_match": {
                  "query": "...",
                  "fields": ["text_content"]
                }
              }
            }
          }
        ]
      }
    }
  }
  ```
- [ ] 현재 인덱스가 분리(visual_index / text_index)되어 있으므로 **통합 인덱스 생성** 필요
  - ES RRF retriever는 하나의 인덱스 안에서 동작하므로 통합이 필수

### 8.2 통합 인덱스 설계
- [ ] `polysight-unified` 인덱스 매핑:
  ```
  - page_image: binary (base64)
  - visual_embedding: rank_vectors (Jina V4 multi-vector, pooled)
  - text_content: text (Docling OCR 추출 텍스트)
  - metadata: { source_file, page_number, indexed_at }
  ```
- [ ] 인덱싱 파이프라인 수정: 한 페이지 → 한 문서에 visual + text 동시 저장
- [ ] 기존 분리 인덱스는 Agent Battle 비교용으로 유지

### 8.3 2-way RRF 쿼리
- [ ] Sub-retriever 1: `script_score` + maxSimDotProduct (Visual)
- [ ] Sub-retriever 2: `multi_match` BM25 (Text/Keyword)
- [ ] (향후 필요 시 semantic_text 필드 추가 + reindex → 3-way 확장 가능)

### 8.4 UI 확장
- [ ] Agent Battle 탭에 Hybrid 결과 패널 추가 (3분할 또는 탭 전환)
- [ ] 또는 별도 "Hybrid Search" 탭 신설
- [ ] 각 결과의 RRF 순위/스코어 표시
- [ ] DevTools용 쿼리 복사 버튼 (데모 시 활용)

### 8.5 데모 시나리오
- [ ] DevTools에서 RRF 쿼리 직접 실행 → "이게 ES의 Hybrid Search입니다"
- [ ] PolySight UI에서 같은 쿼리를 시각적으로 → "이걸 UI로 감싸면 이렇게 됩니다"
- [ ] MSG FAQ 데모처럼 단계별 진행: BM25만 → Visual만 → Hybrid RRF

---

## Phase 9: 다양한 문서 포맷 지원

### 목표
Morphik 수준의 문서 지원 (PDF 외 DOCX, PPTX, XLSX 등)

### 9.1 Office 문서 처리
- [ ] python-docx → 텍스트 추출 + 페이지 이미지 변환
- [ ] python-pptx → 슬라이드 이미지 변환
- [ ] openpyxl → 시트 이미지 변환
- [ ] 공통 인터페이스: `DocumentProcessor.to_pages(file) → List[PageImage]`

### 9.2 텍스트/데이터 파일
- [ ] TXT, MD, CSV, JSON, HTML → 텍스트 추출
- [ ] CSV/JSON → 테이블 시각화 후 이미지 변환 (Visual Agent용)

### 9.3 업로드 UI 확장
- [ ] Gradio 파일 업로드에 지원 포맷 추가
- [ ] 파일 타입 자동 감지 → 적절한 processor 라우팅

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
- [ ] 직접 LLM API (Anthropic/OpenAI) tool_use로 에이전트 루프 구현
- [ ] Visual Agent: visual_search tool만 사용 가능
- [ ] Text Agent: text_search tool만 사용 가능
- [ ] Hybrid Agent: hybrid_search tool 사용

### 11.2 추론 과정 스트리밍
- [ ] 에이전트의 사고 과정(Thought Log)을 Gradio Chatbot에 실시간 표시
- [ ] intermediate_steps 캡처 → UI 업데이트

### 11.3 Agent Battle 2.0
- [ ] 2분할: Visual Agent vs Text Agent의 추론 과정 비교
- [ ] 검색 결과뿐 아니라 "왜 이 결과를 골랐는지" 사고 로그 표시

---

## Phase 12: 웹 데모 배포

### 목표
URL 부여해서 지속적으로 접근 가능한 데모 사이트

### 12.1 배포
- [ ] HuggingFace Spaces (ZeroGPU) 또는 GCP Cloud Run
- [ ] API Key 보안 (HF Secrets / Secret Manager)
- [ ] ES Cloud Serverless 외부 연결 설정

### 12.2 인터랙티브 프레젠테이션 (별도 세션)
- [ ] Elastic 브랜드 가이드라인 적용한 소개 페이지
- [ ] 프로젝트 소개 + 인터랙티브 데모를 하나의 HTML로
- [ ] 참고: elastic-brand-skill.md, Ashish/David의 vibe-coded presso

---

## 우선순위 요약

| 순서 | Phase | 핵심 | 예상 난이도 |
|------|-------|------|-------------|
| 1 | **Phase 8: Hybrid Search (RRF)** | ES의 최대 강점 데모 | 중 |
| 2 | Phase 9: 다양한 문서 포맷 | Morphik 대응 | 하 |
| 3 | Phase 10: 비디오 처리 | Morphik 대응 | 중상 |
| 4 | Phase 11: Agentic Search | Agent Battle 2.0 | 상 |
| 5 | Phase 12: 웹 배포 | 데모 사이트 | 하 |

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
