# CTranslate2 Whisper — Positive Phrase Bias (SSOT)

> **이 문서가 단일 진실 공급원(Single Source of Truth)입니다.** 결정·범위·금지사항·테스트·구현 참조가 전부 여기 있습니다.
> 충돌/혼선 시 이 문서 기준. 배경 리서치는 `archive/deep-research-report.md`, 이전 분절 문서는 `archive/superseded-2026-06-02/`.
>
> 상태: 설계 확정, Phase 1 구현 전 · last updated 2026-06-02
> 워크플로우: 개인 fork(`YuBeomGon/CTranslate2`) `feature/whisper-phrase-bias` 브랜치 → 개인 repo push.

---

## 0. 개요

CTranslate2 Whisper 디코더에 **도메인 용어 positive phrase bias**를 추가한다.
문장 **아무 위치**에서 도메인 phrase가 나오려 할 때, **다음 token logit에 positive bias**를 줘서 도메인 용어 recall을 올린다.

- 기존 번역용 `prefix_bias_beta`/`target_prefix`/`suppress`와 **다른 새 구현**. 외부 hack 아님 — CT2 내부 `LogitsProcessor` 파이프라인에 자연스럽게 주입.
- **negative/suppress(block)는 보류.** positive에 집중.
- **핵심 위험**: 잘못 설계하면 recall은 오르지만 없는 단어를 **insertion**한다. → 작은 bias + continuation-only + bounded path + init-time compile로 간다.

---

## 1. 핵심 결정 (확정)

| 항목 | 결정 |
|------|------|
| 입력 | 문자열 + `total_bias`. 예: `"트랜스포머": 0.5` |
| **값 의미** | `total_bias`는 **logit(score) 가산** (퍼센트·배수 아님). `logits[token] += step`. +b는 상대 가중치 ×exp(b). **기본 0.5** (≈×1.65) |
| 토큰화 | special token(SOT/lang/task/timestamp/start·end) 제거. **leading-space는 제거 금지** (§3) |
| path | canonical **top-1**, 단 **leading-space 버전 + non-leading-space 버전 둘 다** (문장 중간 + segment 시작 커버). 두 path는 len이 다를 수 있어 각자 step 계산. alias·기타 다중 path 보류 |
| **start_bias** | **없음 (0)**. 첫 토큰 안 올림 → insertion 방지. 음향 근거로 단어가 시작된 뒤 완성만 도움 |
| 분배 | continuation step에만. `step_bias = total_bias / (len(ids) - 1)`. 예: `[A,B,C,D]`, total=0.5 → 각 +0.167 |
| 동작 | suffix `[A]`→B에 +step, `[A,B]`→C에 +step, `[A,B,C]`→D에 +step (조건부 continuation) |
| overlap | 같은 step에서 여러 phrase가 같은 `(row,token)`에 → **합산**. 예 +0.2,+0.3 → +0.5 |
| 상한 | config validation에서 **error 아닌 clamp** (운영 중 안 죽게). `total_bias` 0.1~1.5, `step_bias` ≤ 0.5. **clamp는 overlap 합산 후 per-token delta에 적용** |
| 1-token phrase | **skip + 경고 로그** (continuation 불가, start_bias=0이라 bias 0) |
| 매칭 구조 | **reverse trie** (suffix 역방향 탐색). naive scan과 결과 동일, 성능용 |
| primitive | soft bias는 **항상 `indexed_add` primitive를 dispatch 경유로 호출**. **P1에 CPU 구현, P2에 CUDA만 추가 → 인터페이스 불변**. `indexed_add`는 **unique index만** 받음(§7-7). dedupe/합산은 processor가 호출 전에 수행 |
| compile 시점 | 모델 로드(도메인)당 **1회** init-time compile → trie 재사용. chunk/generate마다 재컴파일 **금지** |
| compiled bias 관리 | `model_id`/`tokenizer_hash`/`vocab_size`/`config_version`와 묶음 (model 바뀌면 재compile) |
| empty option | `phrase_biases` 비면 **processor 생성 안 함** → 기존 decode와 결과/속도 동일 |
| 배포 | 도메인/고객사별 모델 로딩 → vocab을 모델 init에 baking. private wheel/Docker, rollback = flag off |

## 2. 입력 → bias 계산 (worked example)

```
입력:  "트랜스포머", total_bias = 0.5
컴파일: " 트랜스포머"  (leading-space 포함) → tokenizer encode → [A, B, C, D]
        special token 제거 / decode([A,B,C,D]) == " 트랜스포머" 검증 / len >= 2

step_bias = total_bias / (len - 1) = 0.5 / 3 ≈ 0.167

decode 중 동작 (start bias 없음):
  ... A        → B에 +0.167
  ... A B      → C에 +0.167
  ... A B C    → D에 +0.167
  ... (suffix 불일치) → no-op
  (아무 suffix 없음 / 첫 토큰) → A는 boost 안 함
```

**clamp/합산 순서 (구현 고정 — 에이전트 혼란 방지):**
```
1. total_bias clamp            (0.1 ~ 1.5)
2. step_bias = total_bias / (len(ids) - 1)
3. step_bias clamp             (≤ 0.5)
4. 같은 (row, token) delta 합산  (overlap)
5. 최종 per-token delta clamp   (≤ max_token_delta, 기본 1.0)
→ clamp 발생 시 debug counter 증가 (운영 로그)
```

## 3. tokenizer 규칙 (correctness — 제일 중요)

Whisper는 byte-level BPE(tiktoken)라 **문장 중간 단어는 앞 공백이 첫 토큰에 붙는다.**
`"트랜스포머"`를 그냥 encode하면 모델이 실제 문장 중간에서 내뱉는 `" 트랜스포머"`(앞 공백)의 token ids와 **달라져서 매칭이 영영 안 된다.**

- canonical path = **"문장 중간에 나올 때 형태" = leading-space 버전**으로 인코딩.
- 제거 대상 = **special prefix/suffix(SOT/lang/task/timestamp/start·end)**. **앞 공백 마커는 절대 제거 금지.**
- **MVP는 두 path를 모두 컴파일**: ① leading-space `" 트랜스포머"`(문장 중간) ② non-leading-space `"트랜스포머"`(segment 시작). 각 path는 별도 검증·step 계산.
- 검증: `decode(ids) == 해당 surface`(공백 포함/미포함 각각), special token 없음, `len(ids) >= 2`. (둘 중 한 path만 유효하면 그것만 사용.)

## 4. MVP 정의

- positive soft bias만
- 문자열 + `total_bias` 입력 (logit 가산, 기본 0.5)
- tokenizer special token 제거 + **leading-space 유지**
- canonical **top-1 path만**
- **start_bias 없음**, continuation step 균등 분배
- **reverse trie** + **init-time compile**
- empty option no-op
- 1-token phrase skip
- bias clamp (합산 후)

→ 상용화 관점에서 과하지 않고 효과를 빠르게 검증 가능.

## 5. 로드맵 (작은 acceptance-test 단위로 분할)

> 한 번에 "trie+GPU+compiler+faster-whisper"를 맡기면 실패. GPU는 **CPU 결과가 맞은 뒤** 붙인다.

| Phase | 내용 | 성공조건 |
|-------|------|---------|
| **P1** | CPU positive bias + **reverse trie** — leading-space correctness, start_bias 없음, total_bias 분배, canonical top-1 | C++ 단위 테스트 통과 |
| **P2** | GPU sparse `indexed_add` — 기존 primitive 패턴 재사용 | **CPU/GPU parity** (fp32 타이트, fp16 tolerance), beam 1/5, batch>1 |
| **P3** | Python binding + tokenizer compile — 문자열+total_bias, special 제거, leading-space 유지 | tokenizer correctness 테스트 + end-to-end |
| **P4** | **faster-whisper** init-time 연동 — 도메인 모델 init 시 baking | A/B로 recall↑ 실측 |
| 보류 | negative/suppress (block) | — |

권장 실행 순서: ① CPU trie processor → ② CPU 테스트 통과 → ③ GPU indexed_add → ④ CPU/GPU parity → ⑤ Python binding → ⑥ faster-whisper.

## 6. Code Map (수정 대상)

| 파일 | 변경 | Phase |
|------|------|-------|
| `include/ctranslate2/models/whisper.h` | phrase bias 옵션 타입 + `WhisperOptions.phrase_biases` (이미 일부 추가됨, commit 34e52dd) | P1/P3 |
| `include/ctranslate2/decoding_utils.h` | `PhraseBiasProcessor` + reverse trie (plain 타입, models 의존 금지) | P1 |
| `src/decoding_utils.cc` | trie build + per-step suffix lookup + soft bias apply (CPU) | P1 |
| `src/models/whisper.cc` | 옵션→entry 변환 + `logits_processors` 주입 (no-speech 후, empty면 skip) | P1/P3 |
| `include/ctranslate2/primitives.h`, `src/cpu/primitives.cc`, `src/cuda/primitives.cu` | `indexed_add` (CPU 먼저, GPU 다음) | P1(CPU)/P2(GPU) |
| `python/cpp/whisper.cc` | `phrase_biases` / `phrase_bias_config` kwarg | P3 |
| `tests/decoding_test.cc` | acceptance tests | P1 |

**저수준 타입 (plain, `decoding_utils.h`, models 의존 금지):**
```cpp
struct PhraseBiasEntry {
  std::vector<size_t> ids;     // token path (leading-space 또는 non-leading-space)
  float step_bias;             // 미리 계산됨 = total_bias/(len-1), clamp 적용
  uint16_t min_prefix_len = 1; // 이 길이만큼 매칭돼야 boost (insertion 안전 노브)
};
```
`models::PhraseBias`(whisper.h) → `PhraseBiasEntry` 변환은 `whisper.cc`. processor 흐름: trie lookup → `(row,token,delta)` 수집 → **dedupe/합산/clamp** → unique `(flat_index, delta)` → `indexed_add(logits, deltas, indices, n)`.

## 7. 🚫 금지사항 (위반 시 리뷰 반려)

1. token **전역(global) boost 금지** — BPE 토큰은 여러 단어 공유. 반드시 **조건부 continuation**(suffix 일치 시에만).
2. **start token boost 금지** (start_bias=0).
3. **leading-space 제거 금지** — special token만 제거.
4. **매 generate마다 trie 재생성 금지** — init-time 1회 compile, 이후 lookup만.
5. `phrase_biases` 비면 **processor 생성 금지** (empty = no-op).
6. **새 CUDA 스타일 만들기 금지** — GPU는 §11 exemplar(`indexed_fill` 등) 패턴을 먼저 조사 후 재사용.
7. **`indexed_add`에 중복 index 넘기기 금지** — CUDA에서 같은 index에 여러 thread `+=` = race. dedupe/합산은 **processor(CPU)에서 primitive 호출 전**에 끝내고, `indexed_add`는 **unique index만** 받는다 (CPU/GPU 공통 계약).

## 8. 상용화 고려사항

1. **special token 제거 보장** (§3) — 가장 중요. `decode(ids)==surface` + special 없음 + `len>=2`.
2. **total bias 분배 고정** — 사용자 값은 전체 phrase bonus. token마다 +0.5(X), continuation에 분배(O).
3. **짧은 phrase**: 1-token skip. 2-token `[A,B]`은 A가 흔하면 insertion 위험 → `min_prefix_len` 노브(필요시 len≥3만).
4. **bias 상한**: clamp (total 0.1~1.5, step ≤ 0.5), error 아님.
5. **insertion 평가 필수**: recall만 보면 안 됨. precision/false-insertion 같이.
6. **streaming partial** (추후, 스트리밍 레이어 붙일 때): partial 약하게/off, final/2-pass에 정상 bias. MVP는 final/2-pass만.
7. **tokenizer 버전 고정**: compiled bias를 `model_id/tokenizer_hash/vocab_size/config_version`와 묶음.
8. **init-time compile**: 문자열→encode→검증→step_bias→trie build→decode 재사용. chunk마다 tokenizer 금지.
9. **empty no-op** (§7-5): flag off 시 기존 CT2와 결과·속도 동일 → 회귀 테스트 필수.
10. **로그/디버깅**: debug mode에서만 `phrase/token_ids/step_bias/matched_prefix_count/boosted_token_count`.
11. **fallback**: `PHRASE_BIAS_ENABLED=false` 또는 config 없음 → 기존 decode 즉시 전환.
12. **배포**: custom wheel + Docker + version pin(`ctranslate2==custom-4.x.y-phrasebias.1`) + rollback tag + upstream rebase 전략.

## 9. 테스트 플랜

분리 원칙: **기능 correctness / tokenizer correctness / GPU parity / 성능 / 실제 ASR 효과를 각각 따로** 검증.

| # | 테스트 | 핵심 |
|---|--------|------|
| 1 | **tokenizer correctness** (P3) | `" 트랜스포머"` encode에 special/SOT/lang/task/timestamp 없음, `decode(ids)==" 트랜스포머"`, `len>=2`. 앞 공백 유지 |
| 2 | **phrase bias 로직 — forced synthetic logits** (P1, 1차 안전망) | 손으로 만든 logits+sequences로: suffix `[A]`→B만 정확히 +step, `[A,B]`→C +step, `[X]`/`[B]`→no-op, suffix 없음→A boost 안 함, **overlap 합산**, **clamp** 동작. (모델 의존 X — 가장 결정적) |
| 3 | **reverse trie** (P1) | `[A,B,C]`,`[A,B,D]`,`[X,Y]`: suffix `[A]`→{B}, `[A,B]`→**{C,D}(공유 prefix 둘 다)**, `[X]`→{Y} |
| 4 | **empty no-op** (P1) | phrase 없음 → processor 생성 안 됨 → 기존 결과·속도 동일 (회귀) |
| 5 | **CPU/GPU parity** (P2) | 동일 logits/sequences/trie: fp32 타이트, **fp16 tolerance(allclose)**, beam 1/5, batch>1, overlap 합산 일치 |
| 6 | **Whisper generate 통합** (P1/P3) | whisper-tiny + dummy features로 `generate`가 phrase_biases 받아 decode result 변경 |
| 7 | **실제 음성 A/B** (P4) | vanilla vs bias-on: domain recall + **precision + false insertion** + CER/WER + latency |
| 8 | **성능 벤치마크** | §10 |
| 9 | streaming (추후) | partial flicker, final commit 품질, 재decode 안정성 |

**테스트 우선순위**: 1 tokenizer → 2 trie → 3 CPU logits → 6 generate 통합 → 5 GPU parity → 7 실제 A/B → (9 streaming).
핵심: tokenizer leading-space 틀리면 기능 자체가 안 먹고, GPU parity 틀리면 운영 못 넣고, A/B에서 insertion 안 보면 좋아진 것처럼 착각함.

## 10. Microbenchmark (성공 조건)

| 축 | 값 |
|----|-----|
| phrase count | 0 / 10 / 100 / 500 |
| beam size | 1 / 5 |
| device | CPU / GPU |
| dtype | float32 / float16 |
| 측정 | tokens/sec, p50/p95 latency, RTF, GPU memory 변화 |

목표: **phrase 100개에서 overhead 체감 거의 없음**, phrase 500개 p95 증가 측정·문서화.

## 11. CT2 내부 exemplar map (구현 참조 — 검증됨 2026-06-02)

> 외부 CUDA 튜토리얼 보지 말 것. 아래 기존 코드를 먼저 읽고 그대로 따라 한다.

**LogitsProcessor 추가 (PhraseBiasProcessor):**
- `LogitsProcessor` base / `apply()` 시그니처 / `apply_first()` → `include/ctranslate2/decoding_utils.h:74`
- `SuppressSequences::apply` (suffix 매칭 `std::equal(end-N,end,...)`) → `src/decoding_utils.cc` (naive 매칭 원형)
- **`RepetitionPenalty::apply` = soft bias 정답 템플릿** (`ops::Gather` + `DEVICE_AND_TYPE_DISPATCH`로 logits in-place) → `src/decoding_utils.cc`
- `DisableTokens` (device별 sparse 쓰기, flat index = `row*vocab+token`) → `decoding_utils.h:36`
- generate 주입 지점 → `src/models/whisper.cc:325-340` (empty면 skip)
- processor 순서 (`make_logits_processors`, apply_first=false는 맨 뒤) → `src/decoding.cc:1091`

**`indexed_add` primitive (✅ P1 CPU `975e26f8` / P2 CUDA `38a5b4c0` 구현 완료):**
- 시그니처: `static void indexed_add(T* x, const T* deltas, const int32_t* indices, dim_t num_indices);` (선언 `primitives.h`, `indexed_fill` 옆)
- **CPU** `src/cpu/primitives.cc` — `indexed_fill` 복사 후 `x[i] = T(float(x[i]) + float(deltas[i]))`. ⚠️ raw `+=` 금지: `DECLARE_IMPL`이 `bfloat16_t`/`float16_t`까지 인스턴스화하는데 half엔 `operator+=` 없음 → **float 경유 필수**.
- **CUDA** `src/cuda/primitives.cu` — `indexed_add_kernel`(=`penalize_previous_tokens_kernel` 패턴, half는 float 경유, **atomic 미사용**: unique index라 race 없음) + launch(`dim3 block(32)`, `device_cast`/`get_cuda_stream`).
- **명시적 인스턴스화 매크로** (없으면 link 실패) → `DECLARE_IMPL` 안 `indexed_fill` 옆에 추가 (CPU/CUDA 각각).
- dispatch 매크로: `DEVICE_AND_TYPE_DISPATCH` `src/dispatch.h:6`. processor는 호출 전 `std::map`으로 dedupe/합산 → **unique index만 전달**(§7-7).

**한 줄 요약(GPU):** `indexed_fill`(set) 복사 → `indexed_add`(+=), `penalize_previous_tokens` launch 패턴 차용, half는 float 경유, atomic 금지. 그 이상 새 CUDA 금지.

## 12. 디자인 검증 (실제 코드 대조)

| 발견 | 의미 |
|------|------|
| `LogitsProcessor::apply(step, logits, disable_tokens, sequences, batch_offset, prefix)`, `apply_first()` 기본 false | PhraseBiasProcessor `apply_first()=false` → no-speech(`apply_first=true`) 뒤 실행 (오염 방지) |
| decode 루프가 processor 순회 후 logits 사용 (`decoding.cc:518` beam / `:863` greedy) | soft bias = logits in-place 수정, 두 경로 자동 적용 |
| step 0엔 `sequences` null | apply()에 `if (!sequences) return;` 가드 필수 |
| `sequences`는 merged `[batch*beam, length]` int32 | trie는 각 row tail 역방향 탐색 |
| 레이어링: `decoding_utils.h`는 `models/whisper.h` 의존 금지 | processor는 plain 타입(`PhraseBiasEntry`), `models::PhraseBias`→entry 변환은 `whisper.cc` |

## 13. 빌드/테스트 환경 (검증됨)

- CPU 빌드(셀프컨테인드): `cmake -DBUILD_TESTS=ON -DBUILD_CLI=OFF -DWITH_MKL=OFF -DWITH_RUY=ON -DWITH_CUDA=OFF -DOPENMP_RUNTIME=NONE -DCMAKE_BUILD_TYPE=Release ..` → `make -j ctranslate2_test` (디렉터리 `build/`)
- **CUDA 빌드(P2~, 검증됨 2026-06-02):** 위 CPU flag에서 `-DWITH_CUDA=ON -DWITH_CUDNN=OFF`로 변경. **두 종류 분리**:
  - `build-cuda/` — `-DCUDA_ARCH_LIST=Auto`(기본). **로컬 GPU 단일 arch(여기선 sm_89)만 감지 → 로컬 smoke/실행 전용.** 빠름.
  - `build-cuda-portable/` — `-DCUDA_ARCH_LIST=Common`. **멀티-arch fat binary(sm_53/60/61/70/75/80/86 + compute_86 PTX) → release portability 게이트.** 실행은 로컬 GPU 한정이라 **컴파일 성공이 게이트**(`--target ctranslate2`). 느림(8 arch).
  - ⚠️ portability는 소스가 아니라 **빌드 설정** 문제다. arch-중립 소스 + `Common` 빌드가 둘 다 있어야 함. `Auto`만으론 그 GPU 전용 바이너리.
- 테스트: `./tests/ctranslate2_test --gtest_filter='PhraseBiasTest.*' /tmp` (**positional data-dir 인자 필수** — `--gtest_list_tests`도 인자 없으면 throw). **파라미터화 테스트(`TEST_P`)는 필터에 `/*` 붙여야 매칭**: 예 `--gtest_filter='*PrimitiveTest.IndexedAdd/*'` (suffix `/0` 때문).
- **baseline known-fail (무시):** `CPU/OpDeviceFPTest.{Gemm,GemmBias,GemmResidual}/float32` (Ruy 수치 아티팩트, 우리와 무관). CUDA 빌드 시 `CUDA/OpDeviceFPTest.Conv1DGroupNoBiasQuantized/*`도 skip(=`WITH_CUDNN=OFF`, 무관). 게이트 판정 = **이 외 신규 실패 0**.
- **레이어링 사실(P2에서 확인, P3/P4 주의):** `DEVICE_AND_TYPE_DISPATCH`를 쓰는 LogitsProcessor는 **모든 device의 primitive 심볼을 강제**한다. 즉 CPU-only primitive 상태에서 CUDA 빌드하면 processor가 `primitives<Device::CUDA>::...`를 참조해 **link 실패**. → primitive의 CPU/CUDA 구현은 **같은 PR/Phase에서 짝으로**(P1 CPU만 머지된 채 CUDA 빌드하면 깨짐).

## 14. 참고

- `archive/deep-research-report.md` — 초기 배경 리서치 (BPE/path 분석, 보존용)
- `archive/superseded-2026-06-02/` — 이전 분절 문서 (impl-plan, testing-manual의 run_ab.py/score.py 스켈레톤 포함 → P4 A/B 때 참고)
