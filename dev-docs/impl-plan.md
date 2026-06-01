# Implementation Plan — Positive Phrase Bias

> 실행 도구: superpowers:subagent-driven-development. 각 task = **작은 acceptance-test 단위**. GPU는 CPU 결과가 맞은 뒤.
> 결정·금지사항·code map은 [`SSOT.md`](SSOT.md)가 기준. 이 문서는 **task 분해 + 성공조건** 위주.

**Goal:** Whisper 디코더에 도메인 용어 positive phrase bias (조건부 continuation, start_bias=0, total_bias 분배, reverse trie, init-time compile).

**Scope of this plan = Phase 1 (CPU core) + Phase 2 (GPU) + Phase 3 (Python/tokenizer) + Phase 4 (faster-whisper).** Phase 1을 가장 상세히, 뒤로 갈수록 outline.

---

## 🚫 금지사항 (모든 task 공통 — 위반 시 리뷰 반려)

1. token **전역 boost 금지** → 조건부 continuation (suffix 일치 시에만).
2. **start token boost 금지** (start_bias=0).
3. **leading-space 제거 금지** (special token만 제거).
4. **매 generate마다 trie 재생성 금지** (init-time 1회).
5. `phrase_biases` 비면 **processor 생성 금지** (no-op).
6. GPU는 **새 CUDA 스타일 금지** → 기존 `DisableTokens`/`RepetitionPenalty`/`indexed_fill`/gather·scatter 패턴 조사 후 재사용.

---

## Phase 1 — CPU positive bias + reverse trie

저수준 입력 = **token-id path + 계산된 step_bias** (문자열→path 변환은 Phase 3). 레이어링: `decoding_utils.h`는 `models/whisper.h` 의존 금지 (plain 타입).

### P1-T1: 데이터 모델 확정 (whisper.h)
- **Files:** `include/ctranslate2/models/whisper.h`, `tests/decoding_test.cc`
- 이미 Task 1에서 `PhraseBiasPath{ids, start_bias, step_bias, min_prefix_len, mode}` / `PhraseBias{token_paths}` / `WhisperOptions.phrase_biases` 추가됨(commit 34e52dd). positive 설계에 맞게 **확인/정리**: `step_bias`가 continuation bias, `start_bias`는 0 고정 사용. (block `mode`는 보류 — 필드는 둬도 되나 P1에서 미사용.)
- **성공조건:** 기존 PhraseBiasTest 2개 통과 유지. 빌드 OK.

### P1-T2: reverse trie 자료구조 + build (decoding_utils)
- **Files:** `include/ctranslate2/decoding_utils.h` (plain `PhraseBiasEntry{ids, step_bias}` + trie), `src/decoding_utils.cc`, `tests/decoding_test.cc`
- trie: 각 path의 prefix를 **역방향**으로 저장. 노드에 "다음에 boost할 token_id + step_bias" action. lookup = 현재 sequence tail을 뒤에서부터 따라가며 매칭된 모든 action 수집.
- **Acceptance tests** (synthetic token ids):
  - paths `[A,B,C]`, `[A,B,D]`, `[X,Y]` build
  - suffix `[A]` → {B} boost
  - suffix `[A,B]` → {C, D} boost (**공유 prefix 둘 다 반환**)
  - suffix `[X]` → {Y}
  - suffix `[Z]` / suffix 없음 → ∅ (no boost, start_bias=0)
  - 1-token path → build 시 skip (경고)
- **성공조건:** 위 테스트 전부 PASS. trie는 build 1회, lookup은 path 수가 아닌 tail 길이에 비례.

### P1-T3: CPU soft bias apply (PhraseBiasProcessor)
- **Files:** `include/ctranslate2/decoding_utils.h` (`PhraseBiasProcessor : LogitsProcessor`), `src/decoding_utils.cc`, `tests/decoding_test.cc`
- `apply_first()=false`. `apply()`: `if(!sequences) return;` 가드. 각 row tail을 trie lookup → boost할 `(token_id, step_bias)` 모음 → 같은 token **합산** → **합산 후 clamp** → logits in-place 수정. 구현 템플릿 = `RepetitionPenalty::apply` (Gather + DEVICE_AND_TYPE_DISPATCH). CPU는 직접 더해도 되나 device dispatch 패턴 유지.
- **Acceptance tests:**
  - `[A,B,C]`, total=0.6→step=0.3: suffix `[A]` → B logit **+0.3**, suffix `[A,B]` → C **+0.3**
  - suffix `[X]` → no-op (logits 불변)
  - suffix 없음(step 0, null sequences) → no-op
  - 같은 token에 +0.2,+0.3 겹침 → **+0.5** (합산), clamp 상한 초과 시 clamp
  - `step_bias == total_bias/(len-1)` 계산 확인
- **성공조건:** 전부 PASS. 회귀(기존 테스트) 영향 없음.

### P1-T4: generate 주입 (whisper.cc, empty no-op)
- **Files:** `src/models/whisper.cc` (옵션→`PhraseBiasEntry` 변환 + `logits_processors`에 주입, no-speech 후), `tests/decoding_test.cc`
- **Acceptance tests:**
  - 변환 헬퍼: `models::PhraseBias` → `PhraseBiasEntry` (step_bias 계산 포함)
  - `phrase_biases` empty → processor **생성 안 됨** (주입 skip)
- **성공조건:** 변환 테스트 PASS. 전체 C++ 테스트 회귀 없음 (baseline Gemm 3개 제외).

### P1-T5: Whisper.generate 통합 스모크 (C++ 또는 Python로 위임)
- 작은 whisper-tiny CT2 + dummy features로 `generate`가 phrase_biases 받아 도는지. (실제 음성은 Phase 3 이후 testing-manual.)
- **성공조건:** crash 없이 decode result 반환, empty와 non-empty 결과가 의미있게 다름.

---

## Phase 2 — GPU sparse indexed_add

- **Files:** `include/ctranslate2/primitives.h` (`indexed_add` 선언), `src/cpu/primitives.cc`, `src/cuda/primitives.cu`
- **반드시:** 기존 `indexed_fill`(`primitives.h:21`)·`penalize_previous_tokens` CUDA 구현을 먼저 읽고 **같은 스타일**로. memory layout/dtype/device dispatch/half·float 처리 동일하게. dedupe 후 unique index 1회 add.
- **Acceptance tests (parity):** 동일 logits/sequences/trie로 CPU vs GPU:
  - float32 → **타이트 일치**
  - float16 → **allclose(tolerance)** (bitwise 동일 기대 금지)
  - beam_size 1 / 5, batch_size > 1
  - 같은 row/token 다중 bias 합산 일치
- **성공조건:** 모든 parity 테스트 통과.

---

## Phase 3 — Python binding + tokenizer compile

- **Files:** `python/cpp/whisper.cc` (`phrase_biases`/`phrase_bias_config` kwarg), Python compile helper.
- compile: 문자열 → `" 트랜스포머"`(**leading-space 유지**) → tokenizer encode → **special token 제거** → `decode(ids)==" 트랜스포머"` 검증 → `len>=2` 아니면 skip → canonical **top-1 path** → step_bias 계산.
- compiled bias는 `model_id`/`tokenizer_hash`/`vocab_size`/`config_version`와 묶어 관리 (model 바뀌면 재compile).
- **Acceptance tests (최우선):**
  - `" 트랜스포머"` encode에 **special/SOT/lang/task/timestamp token 없음**
  - `decode(ids) == " 트랜스포머"` (앞 공백 포함)
  - `len(ids) >= 2`, 아니면 skip
  - `phrase_biases=[]` → 기존 generate 결과/속도 동일 (회귀)
- **성공조건:** 위 통과 + Phase 1 trie와 연결되어 end-to-end 동작.

---

## Phase 4 — faster-whisper / WhisperLiveKit init-time 연동

- 도메인별 모델 init 시 custom vocabulary → token paths → trie compile → 이후 decode 재사용.
- streaming: partial은 약하게/off, final/2-pass에 정상 bias (MVP는 final/2-pass만 가능).
- fallback: `PHRASE_BIAS_ENABLED=false` 또는 config 없음 → 기존 decode 즉시 전환.
- 배포: custom wheel + Docker + version pin(`ctranslate2==custom-4.x.y-phrasebias.1`) + rollback tag + upstream rebase 전략.

---

## 테스트 우선순위 (전체)

1. **tokenizer correctness** (P3) — 가장 error-prone, 최우선
2. **reverse trie 단위** (P1-T2)
3. **CPU logits update** (P1-T3)
4. **Whisper generate 통합** (P1-T5)
5. **GPU parity** (P2)
6. **실제 음성 A/B** (P4, [`testing-manual.md`](testing-manual.md))

분리 원칙: 기능 correctness / tokenizer correctness / GPU parity / 성능 / 실제 ASR 효과를 **각각 따로** 검증.

## Microbenchmark (성공 조건)

| 축 | 값 |
|----|-----|
| phrase count | 0 / 10 / 100 / 500 |
| beam size | 1 / 5 |
| device | CPU / GPU |
| dtype | float32 / float16 |
| 측정 | tokens/sec, p50/p95 latency, RTF, GPU memory 변화 |

목표: **phrase 100개에서 overhead 체감 거의 없음**, phrase 500개 p95 증가 측정·문서화.

## 빌드/테스트 환경 (검증됨)

- CPU 빌드(셀프컨테인드): `cmake -DBUILD_TESTS=ON -DBUILD_CLI=OFF -DWITH_MKL=OFF -DWITH_RUY=ON -DWITH_CUDA=OFF -DOPENMP_RUNTIME=NONE -DCMAKE_BUILD_TYPE=Release ..` → `make -j ctranslate2_test`
- 테스트: `./tests/ctranslate2_test --gtest_filter='PhraseBiasTest.*' /tmp` (positional data-dir 인자 필요)
- **baseline known-fail (무시):** `CPU/OpDeviceFPTest.Gemm/GemmBias/GemmResidual /float32` (Ruy 수치 아티팩트).
