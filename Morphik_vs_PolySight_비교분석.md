# Morphik vs PolySight 비교 분석

## 1. 프로젝트 포지셔닝

| | Morphik | PolySight |
|---|---|---|
| **정체** | 범용 Visual RAG 플랫폼 (Y Combinator, $500K) | Visual vs Text Agent 성능 비교 대시보드 |
| **핵심 가치** | "문서를 이미지로 이해하는 검색" | "Visual이 OCR보다 낫다는 것을 눈으로 증명" |
| **라이선스** | BSL 1.1 (상업용 제한) | 자체 프로젝트 |
| **GitHub Stars** | ~3,530 | - |

---

## 2. 기능별 상세 비교

### 2.1 Visual Embedding

| | Morphik | PolySight |
|---|---|---|
| **모델** | ColPali (SigLIP-So400m + PaliGemma-3B) | Jina V4 |
| **벡터 방식** | Multi-vector (1,024 패치/페이지) | Multi-vector (128 dim) + Token Pooling |
| **검색 방식** | Late Interaction (ColBERT 스타일) | MaxSim (Late Interaction) |
| **압축** | 32x32 grid 고정 | Token Pooling (pool_factor=3, 벡터 1/3 감소) |

→ **차이점**: 모델이 다름. Morphik은 ColPali 계열, PolySight는 Jina V4. 검색 원리(Late Interaction)는 동일.

### 2.2 텍스트/OCR 처리

| | Morphik | PolySight |
|---|---|---|
| **기본 접근** | OCR 안 씀 — Vision LM이 직접 이해 | Docling OCR → 텍스트 추출 → BM25 |
| **대안** | Docling 통합도 지원 | Jina Reader API (계획) |
| **철학** | "OCR은 필요 없다" | "OCR vs Visual의 차이를 보여준다" |

→ **핵심 차이**: Morphik은 OCR을 *대체*하려 하고, PolySight는 OCR과 *비교*하려 함. 이게 PolySight의 가장 큰 차별점.

### 2.3 검색 모드

| | Morphik | PolySight |
|---|---|---|
| **Deep Search** | ✅ (시맨틱 이해, 다단계) | ❌ |
| **Shallow Search** | ✅ (빠른 키워드 기반) | ❌ |
| **Hybrid Search** | ✅ (BM25 + Vector 블렌딩) | ❌ (Visual/Text 별도 실행) |
| **Visual Search** | ✅ | ✅ |
| **비교 검색** | ❌ | ✅ (2분할 동시 비교) |

→ **PolySight에 없는 것**: Hybrid Search (두 결과를 블렌딩해서 하나의 최적 결과를 내는 것). 현재는 각각 독립 실행만 함.

### 2.4 Agent 기능

| | Morphik | PolySight |
|---|---|---|
| **Autonomous Agent** | ✅ (LLM 기반 도구 선택, 다단계 추론) | ❌ (버튼 누르면 고정 로직 실행) |
| **도구 선택** | LLM이 쿼리 분석 → 적절한 도구 자동 선택 | 사용자가 수동 |
| **다단계 추론** | ✅ (Analyze → Plan → Execute → Process → Generate) | ❌ |
| **코드 실행** | ✅ | ❌ |
| **Knowledge Graph** | ✅ (Entity 추출, 관계 그래프) | ❌ |

→ **가장 큰 갭**: Morphik에는 "생각하는 Agent"가 있고, PolySight에는 아직 없음. Gemini 대화에서 마지막에 논의했던 "Agent Battle Arena"가 바로 이 갭을 메우려는 시도.

### 2.5 인프라/백엔드

| | Morphik | PolySight |
|---|---|---|
| **DB** | PostgreSQL + pgvector | Elasticsearch Cloud Serverless |
| **작업 큐** | ARQ (Async Redis Queue) | ❌ (동기 처리) |
| **API** | FastAPI REST | MCP Server (stdio) |
| **Multi-tenancy** | ✅ (user/folder 격리) | ❌ |
| **SDK** | Python + TypeScript | ❌ |

### 2.6 문서 및 미디어 포맷 지원

| 카테고리 | 포맷 | Morphik | PolySight |
|---|---|---|---|
| **문서** | PDF | ✅ | ✅ |
| | DOCX, DOC | ✅ | ❌ |
| | PPTX, PPT, PPSX | ✅ | ❌ |
| | XLSX, XLS, XLSM | ✅ | ❌ |
| **이미지** | PNG, JPG, WEBP, TIFF, BMP, SVG, GIF | ✅ | ✅ (PNG, JPG, WEBP, TIFF) |
| **비디오** | MP4, MOV, AVI, WEBM, MKV, 3GP, MPEG | ✅ (프레임 추출 + 오디오 전사) | ❌ |
| **텍스트/데이터** | TXT, MD, RST, LOG | ✅ | ❌ |
| | JSON, CSV, TSV, YAML, XML | ✅ | ❌ |
| | HTML, HTM | ✅ | ❌ |

→ **심각한 갭**: Morphik은 20+ 포맷을 지원. PolySight는 PDF+이미지만. 특히 **비디오 처리**(프레임 추출 + 오디오 전사)는 Morphik의 강력한 차별점.

### 2.7 MCP / 외부 연동

| | Morphik | PolySight |
|---|---|---|
| **MCP Server** | ✅ morphik-mcp (별도 repo, TypeScript) | ✅ (backend/mcp_server/, Python) |
| **MCP 도구 수** | **16개** (인제스트 4 + 검색 4 + 관리 5 + 파일시스템 3) | 4개 (comparison, visual_only, text_only, index_status) |
| **MCP Transport** | stdio + Streamable HTTPS | stdio만 |
| **Claude Desktop** | ✅ (`npx morphik-mcp`로 즉시 연결) | ✅ (수동 설정 필요) |
| **파일시스템 접근** | ✅ (`--allowed-dir` 보안 제어) | ❌ |
| **End-User 격리** | ✅ (`endUserId` 파라미터) | ❌ |
| **Mastra 연동** | 미확인 | 팀원이 사용 중, 추후 검토 |

### 2.8 SDK (morphik-ts 분석)

| | Morphik | PolySight |
|---|---|---|
| **TypeScript SDK** | ✅ morphik-ts (`import Morphik from 'morphik'`) | ❌ |
| **Python SDK** | ✅ morphik-core 내장 | ❌ (직접 코드만) |
| **초기화** | `new Morphik({ apiKey, baseURL })` | N/A |
| **주요 메서드** | `ingest.ingestFile()`, `retrieve.chunks.create()`, `query.generateCompletion()` | N/A |
| **ColPali 토글** | `use_colpali: true/false` 파라미터 | 별도 엔진 분리 |
| **메타데이터 필터** | ✅ (eq, regex, number_range, date_range) | ❌ |

→ **Morphik의 3-repo 구조**: `morphik-core`(Python 서버) + `morphik-mcp`(MCP 브릿지, 16개 도구) + `morphik-ts`(TypeScript 클라이언트 SDK). 이 세 개가 하나의 생태계를 구성. PolySight는 단일 repo에 모든 것이 들어있음.

---

## 3. PolySight에 없는 핵심 기능 (Gap 분석)

### 🔴 Critical (차별화에 필수)

1. **Autonomous Agent (자율 추론)**
   - 쿼리를 분석하고, 어떤 도구를 쓸지 LLM이 판단
   - 다단계 추론 (검색 → 분석 → 재검색 → 답변)
   - *이것이 Gemini 대화 마지막에 논의한 "Agentic Search"의 핵심*

2. **Hybrid Search (결과 블렌딩)**
   - Visual + Text 결과를 합쳐서 최적 결과 도출
   - BM25 스코어 + Vector 유사도를 가중 합산
   - *현재 PolySight는 비교만 하고, 블렌딩은 안 함*

### 🟡 Important (경쟁력 강화)

3. **Knowledge Graph**
   - 문서에서 Entity 추출 → 관계 그래프 구성
   - "이 문서에 나오는 사람/회사 관계도" 시각화

4. **Deep Search vs Shallow Search**
   - 간단한 질문은 빠르게 (Shallow), 복잡한 질문은 깊게 (Deep)
   - Agent가 질문 난이도를 판단해서 자동 선택

5. **비동기 처리 (작업 큐)**
   - 대용량 PDF 인덱싱 시 백그라운드 처리
   - 현재 PolySight는 동기 처리 → 큰 파일에서 UI 멈춤 가능

### 🟡 Important (경쟁력 강화) — 추가

6. **비디오 처리**
   - 비디오 → 프레임 추출 → ColPali 임베딩 + 오디오 전사(Whisper 등)
   - Morphik의 핵심 차별점 중 하나. PolySight에 전무.

7. **다양한 문서 포맷** (DOCX, PPTX, XLSX, HTML, CSV, JSON 등 20+)
   - Morphik은 거의 모든 문서를 지원, PolySight는 PDF+이미지만

### 🟢 Nice to Have (확장성)

8. **Python/TypeScript SDK**
9. **Multi-tenancy** (다중 사용자 격리)
10. **LiteLLM 통합** (100+ LLM 제공자 지원)

---

## 4. PolySight만의 강점 (Morphik에 없는 것)

1. **Visual vs Text 나란히 비교 UI** — Morphik의 어디에도 없음
2. **Elasticsearch 기반** — Kibana 생태계 활용 가능, Elastic 내부 시연에 적합
3. **Jina V4 + Token Pooling** — ColPali 대비 다른 모델 선택지 제공
4. **벤치마크 중심 설계** — ViDoRe 데이터셋으로 정량적 비교 가능
5. **경량 구조** — PostgreSQL+Redis+pgvector 없이 ES 하나로 동작

---

## 5. 추천 로드맵: Morphik 경쟁 버전으로 가는 길

### Phase A: Agent Battle 완성 (현재 → 단기)
- [ ] 직접 LLM API 호출로 두 Agent에 추론 루프 추가
- [ ] 사고 과정(Thought Log) 실시간 스트리밍 UI
- [ ] 검색 품질 메트릭 (MRR, NDCG) 자동 계산

### Phase B: Hybrid Search 추가 (중기)
- [ ] Visual + Text 결과를 RRF(Reciprocal Rank Fusion)로 블렌딩
- [ ] 3번째 패널: "Hybrid Result" 추가 (또는 탭 전환)
- [ ] Deep/Shallow Search 모드 분리

### Phase C: 플랫폼화 (장기)
- [ ] Knowledge Graph (Entity 추출 + Neo4j 또는 ES graph)
- [ ] 비동기 인덱싱 (Celery 또는 ARQ)
- [ ] 다양한 문서 포맷 지원
- [ ] SDK 제공 (Python)

---

## 6. PolySight 웹 데모 플랫폼 설계 (신규)

### 비전
PolySight를 "설치형 Gradio 앱"에서 **"Elastic 기반 Morphik 경쟁 웹 데모 플랫폼"**으로 확장.
ColPali vs OCR 비교는 메뉴 하나로 유지하면서, Morphik이 제공하는 모든 핵심 기능을 Elastic 버전으로 데모 가능하게 구성.

### 메뉴 구조 (탭 기반)

```
PolySight Web Demo
├── 📊 Agent Battle (기존 핵심 기능)
│   └── Visual(ColPali/Jina) vs Text(OCR) 2분할 비교
│
├── 📄 Document Search (Morphik 대응)
│   ├── PDF / 이미지 / Office(DOCX, PPTX, XLSX)
│   ├── 업로드 → 자동 인덱싱 → 자연어 검색
│   └── Hybrid Search (Visual + Text + BM25 블렌딩)
│
├── 🎬 Video Search (Morphik 대응 — 차별화 포인트)
│   ├── 비디오 업로드 → 프레임 추출 → Visual 임베딩
│   ├── 오디오 전사(Whisper) → 텍스트 인덱싱
│   └── "이 장면 찾아줘" → 타임스탬프 + 프레임 반환
│
├── 🤖 Agentic Search (Morphik Agent 대응)
│   ├── LLM이 쿼리 분석 → 도구 자동 선택 → 다단계 추론
│   ├── 사고 과정(Thought Log) 실시간 표시
│   └── 코드 실행, 요약, 비교 등 복합 작업
│
├── 📈 Benchmark (PolySight 고유)
│   ├── ViDoRe 벤치마크 자동 실행
│   ├── MRR, NDCG, Recall@K 시각화
│   └── 모델별 성능 비교 차트
│
└── ⚙️ Settings (기존)
    ├── ES / Jina / LLM API Key 관리
    └── 인덱스 상태 확인
```

### 기술 스택 확장

| 기능 | 현재 | 추가 필요 |
|------|------|-----------|
| **UI** | Gradio | Gradio Tabs 확장 (또는 FastAPI + React 전환 검토) |
| **비디오 처리** | ❌ | ffmpeg (프레임 추출) + Whisper API (전사) |
| **Office 문서** | ❌ | python-docx, python-pptx, openpyxl → 이미지 변환 |
| **Hybrid Search** | ❌ | ES RRF (Reciprocal Rank Fusion) 쿼리 |
| **Agent 추론** | ❌ | 직접 LLM API (tool_use) |
| **벤치마크** | ViDoRe 데이터만 | 자동 평가 스크립트 + 시각화 (Plotly) |
| **배포** | 로컬 | HuggingFace Spaces 또는 Cloud Run |

### 배포 전략

**추천: HuggingFace Spaces (ZeroGPU)**
- Gradio 앱 그대로 배포 가능
- ZeroGPU로 Jina V4 로컬 추론도 가능 (무료 할당)
- URL 자동 부여: `huggingface.co/spaces/ByungjooChoi/PolySight`
- ES Cloud Serverless는 외부 API로 연결 (API Key는 HF Secrets에 저장)

**대안: Elastic Cloud + GCP Cloud Run**
- 사내 시연용으로 더 안정적
- 커스텀 도메인 가능 (`polysight.elastic.co` 등)

### Morphik과의 최종 포지셔닝 차이

| | Morphik | PolySight |
|---|---|---|
| **DB** | PostgreSQL + pgvector | **Elasticsearch** (Elastic 생태계) |
| **고유 기능** | SDK, Multi-tenancy | **Agent Battle (비교 UI)**, 벤치마크 |
| **핵심 메시지** | "Document AI 플랫폼" | **"Elastic 위에서 돌아가는 Visual RAG 데모"** |
| **타겟** | 개발자/기업 | Elastic 고객/내부 시연 |
