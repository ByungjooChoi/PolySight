# PolySight Cowork 운영 규칙 (Ralph v4)

## 개요
이 문서는 Cowork(Claude)가 PolySight 프로젝트에서 **자율적으로 태스크를 밀어붙이며** 작업을 수행할 때 따라야 할 운영 규칙입니다.

> **핵심 원칙**: 사용자가 한 번 작업을 시작하면, PRD.md의 모든 미완료 태스크가 끝날 때까지 **자율적으로 계속 진행**한다.

---

## 1. 자율 루프 (Autonomous Task Loop)

### 1.1 루프 시작 조건
사용자가 다음 중 하나를 말하면 자율 루프를 시작한다:
- "작업 시작해", "진행해", "밀어붙여"
- "PRD 태스크 처리해줘"
- "Ralph 시작"

### 1.2 루프 동작
```
WHILE 미완료 태스크가 존재:
    1. PRD.md에서 첫 번째 `- [ ]` 태스크 선택
    2. 태스크 실행 (코드 작성/수정)
    3. 검증 실행
    4. IF 검증 성공:
         PRD.md 업데이트 (`- [x]`)
         다음 태스크로 자동 이동
       ELSE:
         재시도 (최대 3회)
         3회 실패 시 `- [!]`로 표시하고 다음으로 이동
    5. 간단히 진행 상황 보고 (1-2문장)
    6. 다음 루프로 계속 진행 (멈추지 않음)

IF 모든 태스크 완료:
    사용자에게 "<promise>DONE</promise>" 보고
```

### 1.3 토큰 효율성 (Cost Optimization)
- 응답은 **간결하게** (불필요한 설명 최소화)
- 코드 작성 시 **핵심 변경만** 표시
- 반복적인 확인 메시지 생략
- 진행 보고: "✅ Phase 1.2 완료. 다음: Phase 1.3" 수준으로 짧게

### 1.4 루프 중단 조건
다음 경우에만 사용자에게 확인 요청:
- 새로운 의존성 추가 (requirements.txt 수정)
- API 키 또는 환경변수 신규 추가
- 아키텍처 변경 (새 모듈/폴더 생성)
- 100줄 이상의 대규모 변경
- 3회 재시도 후에도 해결 불가능한 오류

---

## 2. 태스크 관리

### 2.1 태스크 상태
| 표기 | 의미 |
|------|------|
| `- [ ]` | 미완료 (다음 처리 대상) |
| `- [x]` | 완료 |
| `- [!]` | 블록됨 (사용자 확인 필요) |
| `- [~]` | 진행 중 |

### 2.2 우선순위
1. PRD.md 위에서 아래 순서
2. Phase 번호 순서 (Phase 1 → Phase 2 → ...)
3. 한 번에 **하나의 태스크만** 처리

---

## 3. 검증 규칙 (Validation)

### 3.1 코드 검증
- Python 파일: `python -m py_compile <file>`
- tests/ 폴더 존재 시: `pytest tests/ -v`
- Import 오류 확인: 관련 모듈 임포트 테스트

### 3.2 재시도 정책
```
실패 시:
  1차 재시도: 오류 메시지 분석 후 수정
  2차 재시도: 다른 접근 방식 시도
  3차 재시도: 최소한의 동작 보장

3회 모두 실패:
  - PRD.md에서 `- [!]`로 표시
  - 실패 사유 간단히 기록
  - 다음 태스크로 자동 이동 (멈추지 않음)
```

---

## 4. 파일 참조 규칙

| 파일 | 용도 | 자동 수정 |
|------|------|----------|
| `PRD.md` | 태스크 목록 및 상태 | ✅ 가능 |
| `progress.md` | 작업 로그 (세션 재시작 대비) | ✅ 가능 |
| `architecture.md` | 시스템 구조 참고 | ❌ 읽기만 |
| `README.md` | 사용자 가이드 | ⚠️ 태스크 범위 내 |
| `config.json` | 앱 설정 | ✅ 가능 |
| `.env.example` | 환경변수 템플릿 | ⚠️ 필요 시 |

### 4.1 progress.md 사용법
- 세션 시작 시: `progress.md`를 읽어서 이전 진행상황 파악
- 태스크 완료/실패 시: 간단히 기록 (시간, 결과, 오류 사유)
- 형식:
```
## 2025-01-21
- [완료] Phase 1.1: 프로젝트 구조 생성
- [완료] Phase 1.2: 의존성 설정
- [실패] Phase 2.1: ES 연결 오류 → 재시도 예정
```

---

## 5. 코드 작성 규칙

### 5.1 스타일
- Python: PEP 8 준수
- 함수/클래스에 docstring 작성
- Type hints 사용 권장

### 5.2 안전 규칙
- API 키 하드코딩 금지
- `os.getenv()` 또는 `ConfigManager` 사용
- 민감 정보 로깅 금지

### 5.3 Git 규칙
- 커밋은 **사용자 요청 시에만**
- 커밋 메시지: `[Ralph v4] <작업 내용>`

---

## 6. 커뮤니케이션

### 6.1 진행 보고 (간결하게)
```
좋은 예: "✅ Settings UI 완료. 다음: ViDoRe 로더"
나쁜 예: "Settings UI 구현을 완료했습니다. 이 기능은... (장황한 설명)"
```

### 6.2 언어
- 사용자 대화: **한국어**
- 코드 주석: 영어
- 커밋 메시지: 영어

---

## 7. 현재 프로젝트 컨텍스트

### 목표
PolySight는 **Visual Agent (Jina V4 + MaxSim)** vs **Text Agent (Docling + BM25)** 비교 데모입니다.

### 기술 스택
- Python 3.10+, pip (requirements.txt)
- Elasticsearch Cloud Serverless
- Jina Embeddings v4 (Local/API 모드)
- Gradio (Frontend)
- Docling (OCR)
- ViDoRe Benchmark v3 (데모 데이터)

### 핵심 개념
- **Late Interaction / MaxSim**: `maxSimDotProduct` 스크립트 쿼리
- **Token Pooling**: HierarchicalTokenPooler (pool_factor=3)
- **Multi-vector**: 128차원 rank_vectors

---

## 8. 자율 루프 예시

```
사용자: "PRD 태스크 처리해줘"

Claude:
✅ Phase 1.1: 프로젝트 구조 생성 완료
✅ Phase 1.2: 의존성 설정 완료
✅ Phase 2.1: Elasticsearch 클라이언트 완료
⏳ Phase 2.2: Visual Embedder 진행 중...
✅ Phase 2.2: Visual Embedder 완료
...
(계속 자동 진행)
...
✅ 모든 태스크 완료!
<promise>DONE</promise>
```

---

**버전**: Ralph v4 (2025.01)
**업데이트**: Cowork 자율 루프 + 토큰 효율성 + Gradio/pip 반영
