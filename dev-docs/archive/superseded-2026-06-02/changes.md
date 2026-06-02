# Changes — upstream 대비 변경 내역

> upstream(OpenNMT/CTranslate2)과 다르게 만든 부분을 누적 기록합니다.
> 새 변경마다 위에 항목을 추가하세요 (최신이 위).

## 형식

```
### YYYY-MM-DD — 한 줄 요약
- 무엇을: 변경 내용
- 왜: 이유
- 어디: 파일/함수 경로
- 영향: 성능/호환성/테스트 등
```

---

### 2026-06-02 — 방향 전환: positive-first로 전면 재설계
- 무엇을: 핵심을 **positive soft phrase bias**로 재정의. block/suppress(negative)는 **보류**. SSOT·impl-plan 전면 재작성, deep-research-report.md → `archive/` 이동
- 결정: 입력=문자열+total_bias / special token 제거(leading-space는 유지) / start_bias 없음 continuation만 / step_bias=total/(len-1) / overlap=합산 후 clamp / 1-token skip / canonical top-1 / reverse trie / init-time compile / empty no-op
- 로드맵: P1 CPU positive+trie → P2 GPU indexed_add(parity) → P3 Python binding+tokenizer compile → P4 faster-whisper
- 금지사항·acceptance test·microbenchmark를 SSOT/impl-plan에 명시
- 영향: 문서만. 기존 block 기반 impl-plan(Task 2~6) 폐기. Task 1 데이터 모델(commit 34e52dd)은 positive에 재사용

### 2026-06-01 — 디자인 검증 (실제 코드 대조)
- 무엇을: SSOT 가정 7개를 현재 체크아웃 코드와 1:1 대조 → 전부 일치 확인. refinement V1~V5 추가(processor 순서, disable 1회 적용, merged 레이아웃, step0 null 가드, soft=RepetitionPenalty 템플릿)
- 왜: 보고서가 ZIP 정적 분석 기반이라 실제 코드와 어긋날 리스크 제거
- 어디: `dev-docs/SSOT.md` §7 신설. 검증 근거 file:line 기록
- 영향: 문서만. 재디자인 불필요 결론 → 구현 진행 가능

### 2026-06-01 — 설계 갱신: init-time compile + 도메인별 모델 로딩 (D8/D9)
- 무엇을: SSOT에 컴파일 수명주기 결정 추가 — 배포는 **도메인/고객사 단위 모델 로딩**, vocab을 **모델 init에 baking**(모델 로드당 1회 compile, chunk마다 X). `CompiledPhraseBias`(immutable, shared_ptr<const>) 도입. 동적 키워드 변경은 MVP 범위 밖(도메인 vocab 변경 시 모델 재로드)
- 왜: 도메인별로 모델이 따로 로딩되므로 baking이 메인 경로. 매 chunk 재컴파일 방지
- 어디: `dev-docs/SSOT.md` (D8, D9, §4.5, 가드레일 path cap, 로드맵)
- 영향: 문서만. (미래에 단일 모델 다중 도메인 공유 시에만 `shared_ptr<const CompiledPhraseBias>` 핸들 전달 경로 필요 — 현재 불필요)

### 2026-06-01 — dev-docs 체계 + SSOT 수립
- 무엇을: 루트 `CLAUDE.md` + `dev-docs/`(SSOT, deep-research-report, changes, upstream-sync) 신설
- 왜: 포크 전용 개발 문서를 공식 `docs/`와 분리해 upstream 동기화 충돌 없이 관리. Whisper phrase bias 설계의 단일 진실 공급원 확립
- 어디: `CLAUDE.md`, `dev-docs/`
- 영향: 문서만. 코드 영향 없음

<!-- 코드 변경 시작 시 위에 새 항목 추가 (최신이 위) -->
