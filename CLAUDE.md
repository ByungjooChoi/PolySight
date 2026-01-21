# CLAUDE.md - Project Instructions for Claude Code

이 파일은 Claude Code가 프로젝트 작업 시 자동으로 읽는 지침서입니다.

## 핵심 규칙

**반드시 `.claude/cowork.md`를 먼저 읽고 그 규칙을 따르세요.**

## 빠른 참조

### 자율 루프 트리거
사용자가 "작업 시작", "진행해", "밀어붙여", "Ralph 시작" 등을 말하면:
1. PRD.md에서 `- [ ]` 태스크를 찾아 순서대로 처리
2. 완료되면 `- [x]`로 업데이트
3. **멈추지 말고 다음 태스크로 자동 진행**
4. 모든 완료 시 `<promise>DONE</promise>` 출력

### 기술 스택
- Python 3.10+, pip
- Gradio (Frontend)
- Elasticsearch Cloud Serverless
- Jina V4 (Local/API)
- Docling (OCR)

### 검증
- `python -m py_compile <file>`
- 실패 시 최대 3회 재시도 후 `- [!]`로 표시하고 다음으로

### 커뮤니케이션
- 한국어로 대화
- 진행 보고는 짧게: "✅ Phase X 완료. 다음: Phase Y"

## 상세 규칙

전체 규칙은 `.claude/cowork.md` 참조
