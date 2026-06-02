# SSOT — Whisper Positive Phrase Bias (Single Source of Truth)

> 이 문서가 이 포크의 **단일 진실 공급원**입니다. 충돌/혼선 시 이 문서 기준.
> 배경 리서치는 `archive/deep-research-report.md`(보존용, 현재 결정과 다를 수 있음).
>
> **상태**: 설계 확정, Phase 1 구현 전 (last updated: 2026-06-02)
> **워크플로우**: 개인 fork(`YuBeomGon/CTranslate2`) `feature/whisper-phrase-bias` 브랜치 → 개인 repo push. 동기화 = [`upstream-sync.md`](upstream-sync.md).

---

## 1. 목적

CTranslate2 Whisper 디코더에 **도메인 용어 positive phrase bias**를 추가한다.
문장 **아무 위치**에서 도메인 phrase가 나오려 할 때, **다음 token logit에 positive bias**를 줘서 도메인 용어 recall을 올린다.

- 기존 번역용 `prefix_bias_beta`/`target_prefix`/`suppress`와 다른 **새 구현**.
- 외부 hack 아님 — CT2 내부 `LogitsProcessor` 파이프라인에 자연스럽게 주입.
- **negative/suppress는 현재 보류** (positive에 집중).

## 2. 핵심 결정 (확정)

| # | 결정 |
|---|------|
| **입력** | 문자열 + `total_bias` (예: `"트랜스포머": 0.5`) |
| **값 의미** | `total_bias`는 **logit(score) 가산** (퍼센트·배수 아님). `logits[token] += step`. +b는 상대 가중치 ×exp(b). **기본/권장 0.5** (≈×1.65). 범위 0.1~1.5 |
| **토큰화** | special token(SOT/lang/task/timestamp 등) 제거. **leading-space는 제거 금지** (문장 중간 형태 = `" 트랜스포머"`로 인코딩해야 모델 실제 출력과 매칭됨) |
| **path** | canonical **top-1 path만** 사용 (alias·다중 path 보류) |
| **start_bias** | **없음 (0)**. 첫 토큰은 안 올림 → insertion 방지. 음향 근거로 단어가 시작된 뒤 완성만 도움 |
| **분배** | continuation step에만. `step_bias = total_bias / (len(ids) - 1)`. 예: `[A,B,C,D]`, total=0.5 → 각 +0.167 |
| **동작** | suffix `[A]`→B에 +step, `[A,B]`→C에 +step, `[A,B,C]`→D에 +step (조건부 continuation) |
| **overlap** | 같은 step에서 여러 phrase가 같은 `(row, token)`에 bias → **합산**. 예: +0.2, +0.3 → +0.5 |
| **1-token phrase** | **skip + 경고 로그** (continuation 불가, start_bias=0이라 bias 0) |
| **bias 상한** | config validation에서 **error 아닌 clamp** (운영 중 안 죽게). `total_bias` 0.1~1.5, `step_bias` ≤ 0.5. **clamp는 overlap 합산 *후* per-token delta에 적용** (hot token 폭주 방지) |
| **매칭 구조** | **reverse trie** (suffix 역방향 탐색). naive scan과 결과 동일, 성능 최적화 |
| **compile 시점** | 모델 로드(도메인)당 **1회** init-time compile → trie 재사용. chunk/generate마다 재컴파일 **금지** |
| **empty option** | `phrase_biases` 비면 **processor 생성 안 함** → 기존 decode와 결과/속도 동일 |
| **배포** | 도메인/고객사별 모델 로딩 → vocab을 모델 init에 baking. private wheel/Docker, rollback = flag off |

## 3. 🚫 금지사항 (에이전트 필독)

구현 시 **절대 하면 안 되는 것**:

1. **token을 전역(global) boost 금지** — BPE 토큰은 여러 단어가 공유함. 반드시 **조건부 continuation**(suffix 일치 시에만 다음 토큰).
2. **start token boost 금지** — 첫 토큰은 안 올림 (start_bias=0).
3. **leading-space 제거 금지** — `" 트랜스포머"`의 앞 공백 마커는 유지. special token만 제거.
4. **매 generate마다 trie 재생성 금지** — init-time 1회 compile, 이후 lookup만.
5. **`phrase_biases` 비면 processor 생성 금지** — empty = no-op, 기존 path 그대로.
6. **새 CUDA 스타일 만들기 금지** — GPU는 기존 CT2 primitive(`DisableTokens`, `RepetitionPenalty`, `indexed_fill`, gather/scatter) 패턴을 **먼저 조사하고 재사용**. 검증된 exemplar 위치는 [`ct2-reference-map.md`](ct2-reference-map.md) (`indexed_add`는 `indexed_fill`을 `=`→`+=`로 복사).

## 4. 단계별 로드맵

> 한 번에 "trie+GPU+compiler+faster-whisper"를 맡기면 실패. **아주 작은 acceptance-test 단위**로 쪼갠다. GPU는 CPU 결과가 맞은 뒤 붙인다.

| Phase | 내용 | 검증 |
|-------|------|------|
| **P1** | **CPU positive phrase bias + reverse trie** — leading-space correctness, start_bias 없음, total_bias 분배, canonical top-1 path | C++ 단위 테스트. whisper-tiny A/B로 recall↑ 실측 |
| **P2** | **GPU sparse `indexed_add`** — 기존 primitive 패턴 따름 | **CPU와 동일 결과(parity)**, float32/float16, beam 1/5 |
| **P3** | **Python binding + tokenizer compile** — 문자열+total_bias 입력, special 제거, leading-space 유지 | encode/decode 검증 테스트 |
| **P4** | **faster-whisper / WhisperLiveKit init-time 연동** | 도메인 모델 init 시 baking |
| 보류 | negative/suppress (block) | — |

## 5. Code Map (수정 대상)

| 파일 | 변경 | Phase |
|------|------|-------|
| `include/ctranslate2/models/whisper.h` | phrase bias 옵션 (입력 표현) + `WhisperOptions` 필드 | P1/P3 |
| `include/ctranslate2/decoding_utils.h` | `PhraseBiasProcessor` + reverse trie 선언 (plain 타입, models 의존 금지) | P1 |
| `src/decoding_utils.cc` | trie build + per-step suffix lookup + soft bias apply (CPU) | P1 |
| `src/models/whisper.cc` | 옵션→entry 변환 + `logits_processors` 주입 (no-speech 후, empty면 skip) | P1/P3 |
| `include/ctranslate2/primitives.h` + `src/cpu/primitives.cc` + `src/cuda/primitives.cu` | `indexed_add` (CPU 먼저, GPU 다음) | P1(CPU)/P2(GPU) |
| `python/cpp/whisper.cc` | `phrase_biases` / `phrase_bias_config` kwarg | P3 |
| `tests/decoding_test.cc` | acceptance tests | P1 |

상세 step-by-step + acceptance test + microbenchmark는 [`impl-plan.md`](impl-plan.md).

## 6. Acceptance Tests (핵심)

- `" 트랜스포머"` encode 결과에 **special token 없음**
- `decode(ids) == " 트랜스포머"` (leading-space 포함)
- `[A,B,C]`: suffix `[A]` → B boost, suffix `[A,B]` → C boost
- suffix 불일치 → **no-op** (logits 불변), suffix 없음 → 첫 토큰 boost 없음 (start_bias=0)
- 1-token phrase → skip
- 같은 token에 여러 phrase bias → **합산** 후 clamp
- `phrase_biases` empty → 기존 결과와 **동일** (회귀 테스트 필수)
- **CPU/GPU parity** (P2): float32 타이트, **float16은 tolerance(allclose)**. beam 1/5, batch>1
- `step_bias == total_bias / (len-1)` 확인

## 7. Microbenchmark (성공 조건)

| 축 | 값 |
|----|-----|
| phrase count | 0 / 10 / 100 / 500 |
| beam size | 1 / 5 |
| 측정 | tokens/sec, p50/p95 decode latency, overhead % |

평가 지표(A/B)는 recall만 보지 말 것 → **domain recall + precision + false insertion rate + CER/WER + latency** 항상 함께. 상세 = [`testing-manual.md`](testing-manual.md).

## 8. 디자인 검증 (실제 코드 대조, 2026-06-01)

| 발견 | 의미 |
|------|------|
| `LogitsProcessor::apply(step, logits, disable_tokens, sequences, batch_offset, prefix)`, `apply_first()` 기본 false (`decoding_utils.h:74`) | PhraseBiasProcessor는 `apply_first()=false` → no-speech(`apply_first=true`) 뒤 실행 |
| decode 루프가 processor 순회 후 logits 사용 (`decoding.cc:518`/`863`, beam+greedy 둘 다) | soft bias는 logits in-place 수정. block/soft 둘 다 자동 적용 |
| **soft 구현 템플릿 = `RepetitionPenalty::apply`** — `ops::Gather`+`DEVICE_AND_TYPE_DISPATCH`로 logits in-place 수정 (`indexed_fill`/`penalize_previous_tokens`가 sparse update 전례) | `indexed_add`는 이 패턴의 sibling. **새 스타일 만들지 말 것** |
| step 0엔 `sequences` null | apply()에 `if (!sequences) return;` 가드 필수 |
| `sequences`는 merged `[batch*beam, length]` int32 | trie는 각 row tail을 역방향 탐색 |

## 9. 관련 문서

- [`impl-plan.md`](impl-plan.md) — Phase 1 구현 플랜 (tiny acceptance-test 단위)
- [`testing-manual.md`](testing-manual.md) — vanilla vs fork A/B 검증
- [`changes.md`](changes.md) — 변경 로그
- [`upstream-sync.md`](upstream-sync.md) — upstream 동기화
- `archive/deep-research-report.md` — 초기 배경 리서치 (보존, 현재 결정과 다를 수 있음)
