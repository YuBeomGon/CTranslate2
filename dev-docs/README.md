# dev-docs

이 디렉토리는 **이 포크 전용 개발 문서**입니다. 본인과 Claude Code 에이전트가 봅니다.

- OpenNMT 공식 문서(`docs/`)와 분리되어 있어, upstream과 동기화(rebase/merge)해도 충돌이 없습니다.
- 진입점은 저장소 루트의 [`CLAUDE.md`](../CLAUDE.md)이며, 여기 문서들을 가리킵니다.
- **이 포크의 단일 진실 공급원은 [`SSOT.md`](SSOT.md)** 입니다. 작업 전 먼저 읽으세요.

## 문서 목록

| 파일 | 내용 |
|------|------|
| [`SSOT.md`](SSOT.md) | **단일 진실 공급원** — 목적·설계 결정·수정 파일맵·API·가드레일·로드맵 |
| [`impl-plan.md`](impl-plan.md) | 구현 플랜 (Phase 1 block MVP, TDD step-by-step) + Deferred Phases |
| [`testing-manual.md`](testing-manual.md) | 구현 완료 후 vanilla vs fork A/B 검증 매뉴얼 |
| [`deep-research-report.md`](deep-research-report.md) | 전체 설계 근거·분석·벤치마크 계획 (배경 자료) |
| [`changes.md`](changes.md) | upstream 대비 변경 내역 로그 |
| [`upstream-sync.md`](upstream-sync.md) | upstream(OpenNMT 본가) 동기화 절차 |

## 새 문서를 추가할 때

1. 이 디렉토리에 `*.md`로 추가
2. 위 표와 루트 `CLAUDE.md`의 "커스텀 문서" 목록에 한 줄 추가
3. 설계/결정에 영향이 있으면 `SSOT.md`를 갱신 (SSOT가 항상 기준)
