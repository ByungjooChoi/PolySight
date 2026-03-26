# Gemini 대화 분석 정리: PDF 시각 검색 프로젝트 기획

## 대화 개요

Gemini와의 대화에서 **PolySight 프로젝트의 아키텍처와 UI 설계**에 대해 여러 차례 방향을 잡아가며 논의함. 핵심은 **ColPali(Visual Search) vs OCR(Text Search)의 성능 비교를 시각적으로 보여주는 시스템**을 어떻게 구현할 것인가에 대한 기획 토론.

---

## 논의 흐름 요약 (7개 턴)

### 1. 초기 문제 제기: Kibana Agent Builder의 한계
- **질문**: app.py로 PDF 업로드 → ES 인덱싱은 잘 되는데, Kibana Agent Builder로 2분할 화면을 만들 수 있는가?
- **결론**: Kibana Agent Builder는 **로직(Logic) 도구**이지 **UI 도구가 아님**. Base64 이미지를 Split View로 보여주는 기능 없음. → app.py가 검색/뷰어 역할까지 해야 함 (Gradio 사용)

### 2. Kibana 재검토 요청
- Agent Builder 문서를 다시 검토해봐도 결론은 동일
- Kibana는 **텍스트 기반 JSON 결과** 표시에 최적화 → ColPali처럼 **이미지를 크게 띄우는 용도에는 부적합**

### 3. Agent 기능 + Morphik MCP 검토
- **Philip의 의도**: 단순 검색 앱이 아니라 Agent도 적용하고 싶음
- **Morphik의 MCP 지원** 언급 → Claude Desktop이 Agent + UI 역할을 동시에 할 수 있다는 제안
- app.py를 MCP Server로 만들면 Claude Desktop에서 자연어 검색 + 이미지 확인 가능

### 4. 핵심 목적 재정의: OCR vs ColPali 비교
- **Philip의 진짜 의도 확인**: 2분할은 "채팅 | 문서뷰어"가 아니라 **"ColPali 결과 | OCR 결과"를 나란히 비교**하는 것
- Morphik과의 차별점: Morphik은 Visual Search 도구지만, **Philip의 프로젝트는 Visual vs OCR 성능 차이를 한눈에 보여주는 것**
- Dual Pipeline 필요: Pipeline A(ColPali) + Pipeline B(OCR API)

### 5. 기술 스택 보정: Mastra + API OCR
- 팀원이 **Mastra** 사용 중 → 연동 가능성 검토
- OCR은 로컬 라이브러리(easyocr) 대신 **Jina Reader / Reducto / Unstructured.io API 호출** 방식으로 확정 (최초 기획 반영)
- 3파일 분리 구조 제안: `ingest.py`(공유 로직) + `app.py`(UI) + `mcp_server.py`(Agent 연동)

### 6. Mastra에 대한 오해 해소
- **Mastra는 앱이 아니라 프레임워크(코드 라이브러리)**
- Claude Desktop 같은 완제품이 아님 → 개발자가 코딩해야 프로그램이 됨
- **결론**: Philip은 Mastra를 쓸 필요 없음. app.py(Gradio)가 비교 대시보드 역할

### 7. 최종 방향: "Agent Battle Arena"
- **Philip의 최종 의도**: 단순 검색 결과 비교가 아닌, **두 Agent의 추론 과정(Reasoning)까지 비교**
- User가 Search 한 번 누르면 → 내부적으로 **2개의 Agentic Search가 동시에 실행**
- 좌측: ColPali Agent의 사고 과정 + 이미지 결과
- 우측: OCR Agent의 사고 과정 + 텍스트 결과
- 기술 스택: **LangChain AgentExecutor** + Gradio + `intermediate_steps` 캡처

---

## 최종 합의된 아키텍처

```
[User] → 검색 버튼 클릭 → app.py (Gradio)
                              ├── Agent A (Visual): ColPali 임베딩 → ES visual_index → 이미지 결과 + 사고 로그
                              └── Agent B (Text): OCR API(Jina) → ES text_index → 텍스트 결과 + 사고 로그
                              → 2분할 화면에 동시 표시
```

## 핵심 결정 사항

| 항목 | 결정 |
|------|------|
| UI 프레임워크 | Gradio |
| Agent 프레임워크 | LangChain (AgentExecutor) |
| OCR 엔진 | Jina Reader API (default), Reducto/Unstructured (대안) |
| Visual 검색 | ColPali (vidore/colpali-v1.2) |
| 데이터 저장 | Elasticsearch (visual_index + text_index) |
| LLM | GPT-4o 또는 Claude 3.5 Sonnet |
| Kibana Agent Builder | 사용 안 함 |
| Mastra | 팀원 선택사항, 본인은 불필요 |
| MCP Server | mcp_server.py로 별도 분리 (나중에 구현) |

## 파일 구조 (제안됨)

```
backend/
  ingest.py       # Dual Pipeline (ColPali + Jina OCR) 공유 로직
  search.py       # ES 검색 함수 (visual + text)
app.py            # Gradio Agent Battle Dashboard
mcp_server.py     # MCP Server (Mastra/Claude Desktop 연동용, 껍데기)
requirements.txt  # gradio, langchain, elasticsearch, requests, mcp 등
.env              # ES_URL, ES_API_KEY, JINA_API_KEY, OPENAI_API_KEY
```

## 미해결 / 추후 과제

1. **Agent의 LLM 선택**: GPT-4o vs Claude — 비용/성능 비교 필요
2. **MCP Server 실제 구현**: 현재는 껍데기만 만들어두기로 함
3. **Mastra 연동**: 팀원이 필요하면 mcp_server.py로 연결
4. **벤치마크 데이터셋**: ViDoRe 등 표/그래프가 포함된 금융 문서로 테스트 권장
5. **LangChain intermediate_steps 스트리밍**: 실시간 사고 과정 표시 구현 필요
