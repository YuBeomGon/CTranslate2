# dev-docs

이 디렉토리는 **이 포크 전용 개발 문서**입니다. 본인과 Claude Code 에이전트가 봅니다.

- OpenNMT 공식 문서(`docs/`)와 분리되어 있어, upstream 동기화(rebase/merge)해도 충돌이 없습니다.
- 진입점은 저장소 루트의 [`CLAUDE.md`](../CLAUDE.md)이며, 여기 문서들을 가리킵니다.
- **단일 진실 공급원은 [`SSOT.md`](SSOT.md)** 입니다. 작업 전 먼저 읽으세요.

## 문서 목록

| 파일 | 내용 |
|------|------|
| [`SSOT.md`](SSOT.md) | **단일 진실 공급원** — positive phrase bias 결정·금지사항·code map·로드맵·acceptance test |
| [`impl-plan.md`](impl-plan.md) | 구현 플랜 (Phase 1 CPU+trie, 작은 acceptance-test 단위) + Phase 2~4 |
| [`ct2-reference-map.md`](ct2-reference-map.md) | 구현 시 따라할 **기존 CT2 코드 exemplar 위치** (file:line, GPU 포함) |
| [`testing-manual.md`](testing-manual.md) | 구현 후 vanilla vs fork 실제 음성 A/B 검증 |
| [`changes.md`](changes.md) | 변경 내역 로그 |
| [`upstream-sync.md`](upstream-sync.md) | upstream(OpenNMT 본가) 동기화 절차 |
| `archive/deep-research-report.md` | 초기 배경 리서치 (보존용, **현재 결정과 다를 수 있음**) |

## 새 문서를 추가할 때

1. 이 디렉토리에 `*.md`로 추가
2. 위 표와 루트 `CLAUDE.md` 목록에 한 줄 추가
3. 설계/결정 변경이면 `SSOT.md`를 갱신 (SSOT가 항상 기준)
