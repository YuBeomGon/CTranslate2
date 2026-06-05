# CTranslate2 Whisper — Signed Phrase Bias (SSOT)

> **이 문서가 단일 진실 공급원(Single Source of Truth)입니다.** 결정·범위·금지사항·테스트·구현 참조가 전부 여기 있습니다.
> 충돌/혼선 시 이 문서 기준. 배경 리서치는 `archive/deep-research-report.md`, 이전 분절 문서는 `archive/superseded-2026-06-02/`.
>
> 상태: **P1·P2 구현 완료** (CPU+GPU bias, parity). P3=CT2 Python 바인딩(ids+bias), P4=faster-whisper(tokenizer compile+A/B), P5=signed negative bias. · last updated 2026-06-05
> ⚠️ 아키텍처 경계 §0.1 필독 — **CT2엔 토크나이저 없음**, tokenizer compile은 faster-whisper(P4).
> 워크플로우: 개인 fork(`YuBeomGon/CTranslate2`) `feature/whisper-phrase-bias` 브랜치 → 개인 repo push.

---

## 0. 개요

CTranslate2 Whisper 디코더에 **도메인 용어 signed phrase bias**를 추가한다.
문장 **아무 위치**에서 domain phrase가 나오려 할 때, **다음 token logit에 양수 또는 음수 bias**를 더한다.
양수 bias는 도메인 용어 recall을 올리고, 음수 bias는 자주 생기는 오인식 후보를 soft suppress한다.

- 기존 번역용 `prefix_bias_beta`/`target_prefix`/`suppress`와 **다른 새 구현**. 외부 hack 아님 — CT2 내부 `LogitsProcessor` 파이프라인에 자연스럽게 주입.
- **hard suppress(block)는 보류.** 현재 음수 bias는 logit에서 값을 빼는 soft bias다.
- **핵심 위험**: 잘못 설계하면 양수 bias는 없는 단어를 **insertion**하고, 음수 bias는 정답 후보를 과도하게 누를 수 있다. → 작은 bias + continuation-only + bounded path + init-time compile로 간다.

### 0.1 아키텍처 경계 — ⚠️ CT2엔 토크나이저가 없다 (확정 2026-06-02, 코드 검증)

외부 사용자는 **CT2에 직접 닿지 않고 faster-whisper를 통해** 접근한다. 토크나이저(text→BPE)는 **faster-whisper**가 들고 있고, CT2엔 **Vocabulary(token 문자열↔id 룩업, `vocabulary.h`)만** 있다 — 임의 텍스트를 BPE로 쪼개는 토크나이저는 **없다**. (converter의 `AutoTokenizer`는 변환(offline)할 때만 vocab 추출용.)

```
[사용자] → faster-whisper ──(token ids + step_bias)──► CT2 (LogitsProcessor)
            ▲ 여기서 키워드를 자기 토크나이저로 compile      ▲ ids로만 매칭. 토크나이저 불필요
              (encode, leading-space 2 path, special 제거,
               roundtrip 검증, step_bias 분배)
```

- **faster-whisper 책임(P4):** 키워드 문자열 → token-id path(+step_bias) **compile**. 이게 §3 tokenizer correctness(제일 중요)의 실제 실행 위치.
- **CT2 책임(P1~P3):** 컴파일된 `PhraseBias`(ids+step_bias)를 받아 reverse trie + soft bias 적용. **문자열을 받지 않는다.**
- 따라서 P3는 **Python 바인딩만**(ids+bias를 generate에 내려보내는 통로). tokenizer compile은 P3가 아니라 **P4(faster-whisper)**. (이전 로드맵이 P3에 "tokenizer compile"을 둔 건 위치 오류 → 수정됨.)

---

## 1. 핵심 결정 (확정)

| 항목 | 결정 |
|------|------|
| 입력 | **(faster-whisper 레이어에서)** 문자열 + signed `total_bias`. 예: `"트랜스포머": 5.0`, `"트랜스포마": -3.0`. **CT2 자체는 token ids + step_bias만 받음** (§0.1) |
| **값 의미** | `total_bias`는 **logit(score) 가산** (퍼센트·배수 아님). `logits[token] += step`. `bias > 0`은 상대 가중치 ×exp(b)로 boost, `bias < 0`은 ×exp(b)로 soft suppress. **기본 total 5.0**, term clamp -5.0~5.0, step 절댓값 상한 2.0 |
| 토큰화 | **faster-whisper 책임(P4).** special token(SOT/lang/task/timestamp/start·end) 제거. **leading-space는 제거 금지** (§3) |
| path | canonical **top-1**, 단 **leading-space 버전 + non-leading-space 버전 둘 다** (문장 중간 + segment 시작 커버). 두 path는 len이 다를 수 있어 각자 step 계산. alias는 faster-whisper config에서 추가 surface로 지원 |
| **start_bias** | **없음 (0)**. 첫 토큰 안 올림 → insertion 방지. 음향 근거로 단어가 시작된 뒤 완성만 도움 |
| 분배 | continuation step에만. `step_bias = total_bias / (len(ids) - 1)`, 이후 `[-max_step_bias,+max_step_bias]`로 clamp. 예: `[A,B,C,D]`, total=5.0 → 각 +1.667, total=-5.0 → 각 -1.667 |
| 동작 | suffix `[A]`→B에 `+/-step`, `[A,B]`→C에 `+/-step`, `[A,B,C]`→D에 `+/-step` (조건부 continuation) |
| overlap | 같은 step에서 여러 phrase가 같은 `(row,token)`에 → **합산**. 예 +0.2,+0.3 → +0.5, -0.8,-0.8 → -1.6 |
| 상한 | config validation에서 **error 아닌 clamp** (운영 중 안 죽게). `total_bias` -5.0~5.0, `step_bias` -2.0~2.0. **CT2 최종 clamp는 overlap 합산 후 per-token delta에 `[-max_token_delta,+max_token_delta]`로 적용** |
| 1-token phrase | **skip + 경고 로그** (continuation 불가, start_bias=0이라 bias 0) |
| 매칭 구조 | **reverse trie** (suffix 역방향 탐색). naive scan과 결과 동일, 성능용 |
| primitive | soft bias는 **항상 `indexed_add` primitive를 dispatch 경유로 호출**. **P1에 CPU 구현, P2에 CUDA만 추가 → 인터페이스 불변**. `indexed_add`는 **unique index만** 받음(§7-7). dedupe/합산은 processor가 호출 전에 수행 |
| compile 시점 | 키워드→ids compile은 **faster-whisper init 1회**(도메인 모델당) → `Whisper(phrase_biases=...)` 생성자로 전달 → **CT2가 trie를 1회 build해 WhisperWrapper에 persistent 보관**(P3). generate는 cached trie 참조, **재토크나이즈·trie rebuild 금지**. ⚠️ P1/P2 구현은 generate마다 rebuild했음(임시) → **P3에서 load-time persistent로 닫음** |
| compiled bias 관리 | `model_id`/`tokenizer_hash`/`vocab_size`/`config_version`와 묶음 (model 바뀌면 재compile) |
| empty option | `phrase_biases` 비면 **processor 생성 안 함** → 기존 decode와 결과/속도 동일 |
| 배포 | 도메인/고객사별 모델 로딩 → vocab을 모델 init에 baking. private wheel/Docker, rollback = flag off |

## 2. 입력 → bias 계산 (worked example)

> 아래 "컴파일" 단계는 **faster-whisper(P4)에서** 자기 토크나이저로 수행 → CT2엔 결과 `[ids], step_bias`만 전달 (§0.1).

```
입력:  "트랜스포머", total_bias = 5.0   [faster-whisper init]
컴파일: " 트랜스포머"  (leading-space 포함) → tokenizer encode → [A, B, C, D]
        special token 제거 / decode([A,B,C,D]) == " 트랜스포머" 검증 / len >= 2

step_bias = clamp(total_bias / (len - 1), -max_step_bias, +max_step_bias)
          = clamp(5.0 / 3, -2.0, +2.0) ≈ +1.667

decode 중 동작 (start bias 없음):
  ... A        → B에 +1.667
  ... A B      → C에 +1.667
  ... A B C    → D에 +1.667
  ... (suffix 불일치) → no-op
  (아무 suffix 없음 / 첫 토큰) → A는 boost 안 함
```

**clamp/합산 순서 (구현 고정 — 에이전트 혼란 방지):**
```
1. total_bias clamp            (-5.0 ~ 5.0)
2. step_bias = total_bias / (len(ids) - 1)
3. step_bias clamp             (-2.0 ~ 2.0)
4. 같은 (row, token) delta 합산  (overlap)
5. 최종 per-token delta clamp   (-max_token_delta ~ +max_token_delta, 기본 ±2.0)
```

> clamp debug counter는 아직 미구현이다. 운영 관찰성이 필요하면 별도 low-priority 작업으로 추가한다.

## 3. tokenizer 규칙 (correctness — 제일 중요) · **실행 위치 = faster-whisper (P4)**

> ⚠️ 이 규칙들은 **faster-whisper 레이어에서** 자기 토크나이저로 실행된다 (CT2엔 토크나이저 없음, §0.1). 단위 테스트도 faster-whisper에서 `encode`/`decode` 주입한 pure 함수로. CT2는 결과 ids만 받으므로 이 규칙을 검증할 수단이 없다.

Whisper는 byte-level BPE(tiktoken)라 **문장 중간 단어는 앞 공백이 첫 토큰에 붙는다.**
`"트랜스포머"`를 그냥 encode하면 모델이 실제 문장 중간에서 내뱉는 `" 트랜스포머"`(앞 공백)의 token ids와 **달라져서 매칭이 영영 안 된다.**

- canonical path = **"문장 중간에 나올 때 형태" = leading-space 버전**으로 인코딩.
- 제거 대상 = **special prefix/suffix(SOT/lang/task/timestamp/start·end)**. **앞 공백 마커는 절대 제거 금지.**
- **MVP는 두 path를 모두 컴파일**: ① leading-space `" 트랜스포머"`(문장 중간) ② non-leading-space `"트랜스포머"`(segment 시작). 각 path는 별도 검증·step 계산.
- 검증: `decode(ids) == 해당 surface`(공백 포함/미포함 각각), special token 없음, `len(ids) >= 2`. (둘 중 한 path만 유효하면 그것만 사용.)

## 4. MVP 정의

- signed soft bias만 (`bias > 0` boost, `bias < 0` soft suppress)
- hard block/suppress 없음
- 문자열 + `total_bias` 입력 (logit 가산, 기본 5.0, 허용 범위 -5.0~5.0) — **faster-whisper 레이어 API**. CT2엔 ids+step_bias.
- tokenizer special token 제거 + **leading-space 유지** — **faster-whisper(P4)**
- canonical **top-1 path만**
- **start_bias 없음**, continuation step 균등 분배
- **reverse trie** + **init-time compile**
- empty option no-op
- 1-token phrase skip
- signed bias clamp (합산 후)

→ 상용화 관점에서 과하지 않고 효과를 빠르게 검증 가능.

## 5. 로드맵 (작은 acceptance-test 단위로 분할)

> 한 번에 "trie+GPU+compiler+faster-whisper"를 맡기면 실패. GPU는 **CPU 결과가 맞은 뒤** 붙인다.

| Phase | 내용 | 성공조건 |
|-------|------|---------|
| **P1** ✅ | CPU continuation bias + **reverse trie** — start_bias 없음, total_bias 분배, canonical top-1 (ids 입력) | C++ 단위 테스트 통과 |
| **P2** ✅ | GPU sparse `indexed_add` — 기존 primitive 패턴 재사용 | **CPU/GPU parity** (fp32 타이트, fp16/bf16 tolerance), batch>1, portable 멀티-arch |
| **P3** ✅ | **CT2 Python binding + load-time persistent trie** — `PhraseBias*` pybind 노출, `Whisper(..., phrase_biases=[...])` **생성자 주입**(compile된 trie를 WhisperWrapper에 1회 build/보관), generate가 cached trie 참조(rebuild 없음). `generate(phrase_biases=...)` per-call override도 지원. **ids+step_bias in, 토크나이저 없음** | 바인딩 왕복 + empty no-op + whisper-tiny 효과(생성자/per-call 둘 다) |
| **P4** ✅ | **faster-whisper 연동 (tokenizer compile 포함)** — init 시 자기 토크나이저로 키워드→2 path compile(special 제거·leading-space·roundtrip 검증·step_bias) → `Whisper(phrase_biases=...)` 생성자로 전달 | tokenizer correctness(pure 함수) + 실제 음성 A/B recall↑ |
| **P5** ✅ | **signed negative bias 확장** — 같은 `terms[].bias`에 음수 허용, faster-whisper는 signed clamp/스케줄, CT2는 overlap 합산 후 양방향 clamp | 비모델 단위 테스트 + CT2 negative clamp 테스트 |
| 보류 | hard suppress (block) | — |

> ⚠️ **tokenizer compile은 P3가 아니라 P4** (§0.1). CT2엔 토크나이저가 없으므로 문자열→ids는 faster-whisper에서. P3는 그 결과(ids+bias)를 받아 **trie를 1회 build해 persistent 보관**.
> **trie 보관 위치 결정(P3):** replica가 아니라 **WhisperWrapper(per-Whisper-object)** 에 `shared_ptr<const PhraseBiasTrie>` 보관 → generate마다 `WhisperOptions.compiled_phrase_bias_trie`로 **read-only 주입**(replica는 읽기만 → ReplicaPool race 없음, lock 불필요). 생성자 주입이라 immutable. **mutable setter는 보류**(replica race·불필요).
> **`generate(phrase_biases=)` 3-way semantics(확정):** `None`=생성자 model-level trie 사용 · `[]`=이 호출만 disable(ablation) · `[...]`=per-call override(매 call build, 실험/테스트용 — 상용은 생성자만). pybind `optional<vector>`가 None/[] 구분.

권장 실행 순서: ① CPU trie processor ✅ → ② CPU 테스트 ✅ → ③ GPU indexed_add ✅ → ④ CPU/GPU parity ✅ → ⑤ **CT2 Python binding(ids+bias)** → ⑥ **faster-whisper(tokenizer compile + A/B)**.

## 6. Code Map (수정 대상)

| 파일 | 변경 | Phase |
|------|------|-------|
| `include/ctranslate2/models/whisper.h` | phrase bias 옵션 타입 + `WhisperOptions.phrase_biases` (이미 일부 추가됨, commit 34e52dd) | P1/P3 |
| `include/ctranslate2/decoding_utils.h` | `PhraseBiasProcessor` + reverse trie (plain 타입, models 의존 금지) | P1 |
| `src/decoding_utils.cc` | trie build + per-step suffix lookup + soft bias apply (CPU) | P1 |
| `src/models/whisper.cc` | 옵션→entry 변환 + `logits_processors` 주입 (no-speech 후, empty면 skip) | P1/P3 |
| `include/ctranslate2/primitives.h`, `src/cpu/primitives.cc`, `src/cuda/primitives.cu` | `indexed_add` (CPU 먼저, GPU 다음) | P1(CPU)/P2(GPU) |
| `include/ctranslate2/decoding_utils.h`, `src/decoding_utils.cc` | `build_phrase_bias_trie(entries)→shared_ptr<const PhraseBiasTrie>` + `PhraseBiasProcessor(shared_ptr<const trie>)` ctor (기존 `(entries)` 편의 ctor는 delegating 유지). processor가 trie를 **shared_ptr로 보유** | P3 |
| `include/ctranslate2/models/whisper.h` | `WhisperOptions.compiled_phrase_bias_trie` (`shared_ptr<const PhraseBiasTrie>`) 추가 | P3 |
| `src/models/whisper.cc` | `WhisperReplica::generate`: `options.compiled_phrase_bias_trie` 있으면 그걸로 processor(**rebuild 없음**), 없고 `phrase_biases` 있으면 build(편의/C++ fallback) | P3 |
| `python/cpp/whisper.cc` | `PhraseBiasPath/PhraseBias` pybind 노출 + **`WhisperWrapper`에 `_compiled_trie` 보관** + 생성자 `phrase_biases` kwarg(1회 compile) + `generate`가 per-call override 또는 `_compiled_trie`를 options에 주입 | P3 |
| `python/tests/` | 바인딩 왕복 + empty no-op + 생성자/per-call 효과 통합 | P3 |
| `tests/decoding_test.cc` | acceptance tests + shared-trie 공유 테스트 | P1/P3 |
| **(faster-whisper repo, 별도)** | 키워드→2 path compile(pure 함수, encode/decode 주입) + `Whisper(phrase_biases=...)` 생성자 연동 + A/B | P4 |

**저수준 타입 (plain, `decoding_utils.h`, models 의존 금지):**
```cpp
struct PhraseBiasEntry {
  std::vector<size_t> ids;     // token path (leading-space 또는 non-leading-space)
  float step_bias;             // 미리 계산된 signed step bias, clamp 적용
  uint16_t min_prefix_len = 1; // 이 길이만큼 매칭돼야 bias 적용
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

1. **special token 제거 보장** (§3, **faster-whisper P4**) — 가장 중요. `decode(ids)==surface` + special 없음 + `len>=2`.
2. **total bias 분배 고정** — 사용자 값은 전체 phrase bonus. token마다 전체 bias를 그대로 더하기(X), continuation에 분배(O).
3. **짧은 phrase**: 1-token skip. 2-token `[A,B]`은 A가 흔하면 insertion 위험 → `min_prefix_len` 노브(필요시 len≥3만).
4. **bias 상한**: clamp (total -5.0~5.0, step -2.0~2.0), error 아님.
5. **insertion 평가 필수**: recall만 보면 안 됨. precision/false-insertion 같이.
6. **streaming partial** (추후, 스트리밍 레이어 붙일 때): partial 약하게/off, final/2-pass에 정상 bias. MVP는 final/2-pass만.
7. **tokenizer 버전 고정** (faster-whisper P4): compiled bias를 `model_id/tokenizer_hash/vocab_size/config_version`와 묶음.
8. **init-time compile** (faster-whisper P4): 문자열→encode→검증→step_bias. chunk마다 tokenizer 금지. (CT2는 받은 entries로 trie build.)
9. **empty no-op** (§7-5): flag off 시 기존 CT2와 결과·속도 동일 → 회귀 테스트 필수.
10. **로그/디버깅**: debug mode에서만 `phrase/token_ids/step_bias/matched_prefix_count/boosted_token_count`.
11. **fallback**: `PHRASE_BIAS_ENABLED=false` 또는 config 없음 → 기존 decode 즉시 전환.
12. **배포**: custom wheel + Docker + version pin(`ctranslate2==custom-4.x.y-phrasebias.1`) + rollback tag + upstream rebase 전략.

## 9. 테스트 플랜

분리 원칙: **기능 correctness / tokenizer correctness / GPU parity / 성능 / 실제 ASR 효과를 각각 따로** 검증.

| # | 테스트 | 핵심 |
|---|--------|------|
| 1 | **tokenizer correctness** (**P4 faster-whisper**, pure 함수) | `" 트랜스포머"` encode에 special/SOT/lang/task/timestamp 없음, `decode(ids)==" 트랜스포머"`, `len>=2`. 앞 공백 유지. 2 path 각각. (CT2 아님 — 토크나이저가 거기 있음) |
| 2 | **phrase bias 로직 — forced synthetic logits** (P1, 1차 안전망) | 손으로 만든 logits+sequences로: suffix `[A]`→B만 정확히 +step, `[A,B]`→C +step, `[X]`/`[B]`→no-op, suffix 없음→A boost 안 함, **overlap 합산**, **clamp** 동작. (모델 의존 X — 가장 결정적) |
| 3 | **reverse trie** (P1) | `[A,B,C]`,`[A,B,D]`,`[X,Y]`: suffix `[A]`→{B}, `[A,B]`→**{C,D}(공유 prefix 둘 다)**, `[X]`→{Y} |
| 4 | **empty no-op** (P1) | phrase 없음 → processor 생성 안 됨 → 기존 결과·속도 동일 (회귀) |
| 5 | **CPU/GPU parity** (P2) | 동일 logits/sequences/trie: fp32 타이트, **fp16 tolerance(allclose)**, beam 1/5, batch>1, overlap 합산 일치 |
| 6 | **Whisper generate 통합** (P3, Python 바인딩) | whisper-tiny로 (a) `Whisper(phrase_biases=...)` **생성자 주입** 출력 변경, (b) `generate(phrase_biases=...)` per-call override 출력 변경, (c) None/[] **empty no-op**, (d) shared trie가 generate마다 rebuild 안 됨(설계 보장) |
| 7 | **실제 음성 A/B** (P4) | vanilla vs bias-on: domain recall + **precision + false insertion** + CER/WER + latency |
| 8 | **성능 벤치마크** | §10 |
| 9 | streaming (추후) | partial flicker, final commit 품질, 재decode 안정성 |

**테스트 우선순위**: (CT2) 2 trie ✅ → 3 CPU logits ✅ → 5 GPU parity ✅ → 6 generate 통합(P3) · (faster-whisper) 1 tokenizer(P4) → 7 실제 A/B(P4) → (9 streaming).
핵심: tokenizer leading-space 틀리면(P4) 기능 자체가 안 먹고, GPU parity 틀리면(✅) 운영 못 넣고, A/B에서 insertion 안 보면 좋아진 것처럼 착각함.

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
