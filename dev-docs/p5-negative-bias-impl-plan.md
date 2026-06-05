# Phase 5 Implementation Plan — Signed Negative Phrase Bias

## Goal

기존 phrase bias 경로를 **signed soft bias**로 확장한다.
같은 `terms[].bias` 필드에서 양수는 boost, 음수는 soft suppress로 해석한다.
hard block/suppress와 start token bias는 이번 범위가 아니다.

## Scope

- CT2는 계속 token ids + signed `step_bias`만 받는다.
- faster-whisper는 문자열 config를 token paths로 컴파일하고, `bias < 0`도 같은 2-variant 경로(`" "+surface`, `surface`)로 처리한다.
- 1-token phrase는 계속 skip한다. start token에 음수 bias를 주는 기능은 별도 P5.1 후보로 둔다.
- 모델 로딩 테스트는 이번 단계에서 제외한다. 비모델 단위 테스트와 synthetic logits 테스트만 게이트로 사용한다.

## CT2 Tasks

1. C++ synthetic logits 테스트 추가:
   - overlap된 negative delta가 기본 `max_token_delta=2.0`에서 `-2.0`으로 clamp되는지 검증.
2. `PhraseBiasProcessor` 최종 clamp를 signed로 변경:
   - 기존: `min(sum, max_token_delta)`
   - 변경: `clamp(sum, -max_token_delta, +max_token_delta)`
3. Python binding 비모델 테스트 추가:
   - `PhraseBiasPath(step_bias=-0.25)`가 그대로 왕복되는지 확인.
4. 문서 동기화:
   - `SSOT.md`, `changes.md`, 관련 readiness 문서에서 positive-only 표현 제거.

## faster-whisper Tasks

1. config parser:
   - `default_total_bias`, `min_total_bias`, `max_total_bias`, term `bias`는 finite signed float 허용.
   - 기본값: `default_total_bias=5.0`, `min_total_bias=-5.0`, `max_total_bias=5.0`, `max_step_bias=2.0`.
2. scheduler:
   - uniform: `clamp(total_bias / continuation_count, -max_step_bias, +max_step_bias)`.
   - ramp: 누적 목표 delta와 increment 모두 signed 값을 보존한다.
3. tests:
   - negative uniform이 leading-space/non-leading-space 두 path를 생성하는지 검증.
   - negative ramp가 음수 increment를 보존하는지 검증.
4. docs/examples:
   - README와 dev-docs architecture에 signed semantics 반영.
   - example config에 negative suppress term 추가.

## Verification

- CT2:
  - `cmake --build build -j2 --target ctranslate2_test`
  - `./build/tests/ctranslate2_test --gtest_filter='PhraseBiasTest.*' tests/data`
  - Python binding 비모델 테스트만 실행
- faster-whisper:
  - `tests/test_phrase_bias.py`의 비모델 phrase bias tests
  - `ruff check faster_whisper/phrase_bias.py faster_whisper/transcribe.py tests/test_phrase_bias.py`
  - `py_compile` for touched Python files

## Out of Scope

- hard block/suppress mode
- start token bias
- whisper model loading tests
- production A/B 평가
- beam/nbest 튜닝 결론
