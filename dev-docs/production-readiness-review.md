# Whisper Phrase Bias — 프로덕션 준비도 감사

> 감사일 2026-06-04 · 대상 브랜치 CT2 `feature/whisper-phrase-bias`, faster-whisper `feature/phrase-bias`
> 관점: 기능 동작이 아니라 **운영에서 터지는 것** — 동시성·성능 스케일·메모리 안전·수치·이식성·실패 모드·테스트·유지보수
> 방법: 5개 서브에이전트 병렬 정적 감사(코드 정독, file:line 근거). 런타임 검증 필요 항목은 `needs-runtime-check` 표기.
> 업데이트 2026-06-05: faster-whisper B1/H1/H4 hardening과 signed negative bias가 반영됐다. 아래 findings는 감사 당시 근거와 현재 해소 상태를 함께 남긴다.

---

## TL;DR — 신호등

| 영역 | 평가 | 한 줄 |
|---|---|---|
| **이식성(지원 매트릭스 회귀)** | 🟢 통과 | CPU백엔드·GPU arch(CC≥3.5)·CUDA11·dtype 6종 **어디서도 기존 CT2 매트릭스를 좁히지 않음** |
| **동시성/스레드 안전** | 🟢 견고 | immutable `shared_ptr<const>` trie + stateless processor + unique-index 보장. blocker 없음 |
| **성능(스케일)** | 🟢 견고 | 역방향 walk+즉시 break, 매칭 step만 O(matched). 구조적으로 무시 가능 |
| **메모리 안전/수치** | 🟢 견고 | host-resident seq read(기존 패턴), clamp 순서·fp16/bf16 누산·ramp 수학 정확 |
| **운영 관찰성(observability)** | 🟢 보강됨 | faster-whisper가 드롭 variant를 warning으로 기록 |
| **입력 검증(robustness)** | 🟢 보강됨 | signed bias 범위·min_prefix_len·min/max 역전 검증 |
| **API 정직성** | 🟠 약함 | `start_bias`/`mode`(Block) 정의만 되고 전혀 소비 안 됨(죽은 public API) |
| **테스트 적정성** | 🟠 부족 | beam>1·GPU e2e·eot경계·대형config 무검증. 기능 스모크 수준 |
| **Upstream PR 적합성** | 🟠 하~중 | 코어 품질은 upstream급, 그러나 표면 정리(죽은 필드·한글주석·테스트) 필요 |

**핵심 결론**: **C++/CUDA 코어 메커니즘은 프로덕션 등급이다.** 사용자가 가장 걱정한 "기존 CT2 지원 영역 보장"은 **회귀 없음으로 확정**. 감사 당시 약점이던 Python 컴파일러 측 운영 관찰성·입력 검증은 hardening으로 보강됐다. 남은 위험은 양 레포에 걸친 테스트 공백과 upstream 표면 정리다.

---

## 1. 사용자 핵심 요구 — 지원 매트릭스 회귀 (🟢 회귀 없음)

| 축 | 기존 CT2 | phrase bias 후 | 회귀 |
|---|---|---|---|
| **CPU 백엔드** | MKL/oneDNN/Ruy/OpenBLAS/Accelerate, x86-64 SSE4.1+, ARM64 | 동일 — `indexed_add` CPU는 백엔드 매크로·`#ifdef` 없는 **순수 루프** | **없음** |
| **GPU arch** | CC ≥ 3.5(소스/CUDA11), wheel sm_50+(CUDA12 Common) | 동일 — atomicAdd·warp intrinsic·coop-groups **미사용**, 기존 `penalize_previous_tokens`와 동일한 float-경유 변환 패턴 | **없음** (sm_35 빌드 스모크만 `needs-runtime-check`) |
| **CUDA 버전** | 최소 11.0 | 동일 — 신규 API/헤더 의존 없음(`cuda_bf16.h`는 기존 포함) | **없음** |
| **dtype** | fp32/fp16/bf16/int8/int16 | 동일 — `DEVICE_AND_TYPE_DISPATCH`로 6종 전부 인스턴스화, fp16/bf16 parity 테스트 보유 | **없음** |

근거: `src/cuda/primitives.cu:70-98`, `src/cpu/primitives.cc:65-71`, `src/decoding_utils.cc:160-163`, `CMakeLists.txt:498,530-538`, `docs/hardware_support.md`.
빈 config면 LogitsProcessor 자체가 미부착(`src/models/whisper.cc:359-364`) → 기존 경로 오버헤드 0.

> 잔여 권고: 저장소가 실제로 CUDA 11 + sm_35를 CI에서 더는 빌드하지 않으면(현 wheel CUDA 12.8) 그 경로 자체가 이미 비검증 영역. 보수적으로 sm_35 1회 빌드 스모크면 confirmed 승격.

---

## 2. 코어가 견고한 이유 (🟢 — 확정)

- **동시성**: `_compiled_trie`는 `shared_ptr<const PhraseBiasTrie>`로 모든 replica·동시 `generate`가 공유하나 `lookup`은 `const`, 빌드 후 불변. `PhraseBiasProcessor`에 **step별 멤버 상태가 없음** — 매 step `sequences`(reorder된 alive_seq)에서 batch row별로 재계산하므로 beam reorder/배치 교차오염 구조적 차단. CUDA scatter는 `std::map` dedup + `b*vocab+token` 전역 유니크로 atomic 없이도 race 없음. (`src/decoding_utils.cc:87-160`, `src/decoding.cc:518-588`)
- **성능**: trie `lookup`이 끝에서 역방향으로 걷다 **첫 불일치에서 break** → 비용이 시퀀스 길이가 아닌 최대 phrase 길이로 bounded. 대부분 step은 hash miss 1회 후 early-return. `indexed_add`는 매칭 step만, O(matched tokens) → 이미 도는 O(V) softmax/projection에 묻힘. (`src/decoding_utils.cc:87-98`)
- **메모리/수치**: `alive_seq` host-resident → host read 안전(기존 NoRepeatNgram/SuppressSequences와 동일 패턴). overlap 합산 후 단일 signed clamp `clamp(sum, -max_token_delta, +max_token_delta)` 순서 정확, off-by-one 없음. fp16/bf16는 float 누산. faster-whisper의 ramp/uniform 수학이 C++ 가산 누적과 교차검증으로 정확. (`src/decoding_utils.cc:95-160`, `tests/decoding_test.cc:181-189`)

---

## 3. 실제 리스크 — 심각도순 Findings

### 🔴 Blocker — 운영 신뢰성

**B1. 무음 실패(silent drop) — phrase가 경고 없이 사라짐 (해소됨)**
`faster_whisper/phrase_bias.py:122,137-154` · 파일 전체 로깅 0건(확인).
- 3중 드롭 경로가 전부 무음: ① 1-토큰 용어(`len(ids)<2`) → 두 variant 스킵 → term 누락, ② roundtrip 불일치(`decode(ids)!=variant`) → variant 드롭, ③ 과도한 `min_prefix_len` → 전 위치 제외돼 bias 통째 무력화.
- 운영 시나리오: 용어 50개 등록 → 일부가 **아무 경고 없이** bias 미적용. "컴파일 결과 surface 목록" INFO 로그조차 없어 운영자가 진단 불가.
- 상태: 드롭 surface/variant를 `logger.warning`으로 남긴다(too-few-tokens/roundtrip-mismatch/duplicate token path/min_prefix_len 초과).

### 🟠 High

**H1. 입력 수치 검증 부재 (해소됨)** `phrase_bias.py:76-79,102-108,195`
- 감사 당시: `min_prefix_len` 하한/상한 없음, `min_total_bias>max_total_bias` 역전, 음수 bias 무방비.
- 상태: `min_prefix_len >= 1`, `min_total_bias <= max_total_bias`, `max_step_bias > 0`, finite number 검증이 들어갔다. 음수 bias는 signed soft suppress로 지원한다.

**H2. 죽은 public API — `start_bias`/`mode`/`PhraseBiasMode::Block`** `include/ctranslate2/models/whisper.h:14-27`, `src/models/whisper.cc:232-245`
- struct에 정의돼 있으나 구현 전체에서 **한 번도 read 안 됨**(grep 확인). `to_phrase_bias_entries`는 `ids/step_bias/min_prefix_len`만 복사. Python 바인딩엔 `start_bias`/`mode` 미노출(헤더↔Python 불일치). → 사용자가 `mode=Block`("강제/차단") 기대해도 **조용히 약한 additive bias로 동작**. 정적 디버그 불가.
- 수정: 미구현 필드/enum **제거**(권장), 또는 설정 시 throw. 한번 노출한 public 타입은 후방호환 부담.

**H3. beam>1 end-to-end 미검증** `tests/decoding_test.cc:152-189`, `python/tests/test_phrase_bias.py:80-84`(beam_size=1 고정)
- 전 테스트가 greedy(beam=1)/합성 logits. **프로덕션 기본이 beam=5**(`whisper.h:40`)인데 실제 beam 확장(split/merge_batch_beam) 경로 무검증 — PR 리뷰어가 가장 먼저 물을 케이스. → beam_size=5 통합 테스트 1개 추가.

**H4. 구버전 CT2 capability 선검증 부재 (해소됨)** `transcribe.py:710-733`
- `phrase_biases` kwarg를 모르는 구버전 CT2에 던지면 "unexpected keyword argument"류 난해한 에러. README의 "PhraseBias 지원 빌드 필요"를 코드가 선검사 안 함. → `hasattr(ctranslate2.models,"PhraseBias")` 선검사 후 친절한 에러. 현재는 선검사가 구현됨.

### 🟠 Medium

| ID | 항목 | 근거 | 영향 |
|---|---|---|---|
| M1 | GPU 매 매칭 step pageable H2D 복사 2회 + fp16 convert 커널 | `decoding_utils.cc:152-159`, `cuda/primitives.cu` | 수백 phrase로 매칭 빈도 오르면 호스트 부분 stall 누적. `--log_profiling` 확인 권장(needs-runtime-check) |
| M2 | 한국어 주석 다수 | `decoding_utils.cc`, `primitives.cu`, `decoding_test.cc` | upstream(영어 코드베이스) 머지 차단 |
| M3 | GPU e2e·eot경계·대형config·동시성 테스트 공백, `--log_throughput` 미측정 | tests 전반 | CLAUDE.md 작업규칙(성능 회귀 측정) 미이행 |
| M4 | 실토크나이저 roundtrip 회귀 1케이스("transformer")만 | `phrase_bias.py:151-154`, test:228 | 한국어/숫자/하이픈/혼합 phrase의 variant 생존 미검증(특히 비띄어쓰기 BPE) |
| M5 | 다국어 판정이 모델속성→문자열(`.en`/`-en`) 추정으로 후퇴 | `transcribe.py:624-628,715-719` | 영어전용 로컬모델이 `.en`로 안 끝나면 오판(폴백 한정, tokenizer.json 있으면 무관) |
| M6 | 통합 테스트가 `step_bias=50` 기계적 flip만 | `test_phrase_bias.py:87-113` | "큰 bias면 토큰 바뀜"만 확인, 실제 recall 향상 미검증. 모델/버전 바뀌면 skip으로 무력화 |

### 🟢 Low / 정보

- **L1** `apply` step당 `std::map`/`vector`/temp StorageView 재할당(`decoding_utils.cc:135-153`) — batch16·beam5에서 generate당 수만 회. 재사용 버퍼로 allocator churn 제거 가능(마이크로 최적화).
- **L2** per-call `generate(phrase_biases=...)` 경로는 전부 필터돼 빈 trie여도 processor 부착(`models/whisper.cc:359-364`) — 매 step 빈 작업. 빌드 시 action 0개면 미부착 권장.
- **L3** prompt(prefix) 토큰 미반영 — `forward_prompt` 토큰이 `alive_seq`에 없어, initial_prompt/previous-text 경계에 걸친 phrase 첫 continuation 놓침(`models/whisper.cc:280-307`). 기능 한계, SSOT 문서화 권장.
- **L4** `indexed_add` unique-index 계약이 주석 의존, 헤더 미명문화(`primitives.h:23`) — 향후 비-unique 호출처 재사용 시 silent race. 헤더에 계약 명시 권장.
- **L5** `max_token_delta` 노출/명명 + clamp debug counter 결정 필요. 기본값은 2.0이고 signed clamp는 `[-2.0,+2.0]`.

---

## 4. Upstream PR 준비도

| 레포 | 준비도 | 비고 |
|---|---|---|
| **CTranslate2** | 하~중 | 코어 로직·레이어링·헤더 규율·#990 LogitsProcessor 적합성은 **upstream 품질**. 표면 정리 필요 |
| **faster-whisper** | 중 | 수학 핵심·바인딩 스키마 일치. 로깅(blocker)+검증(high)은 해소, 실토크나이저 회귀 테스트 필요 |

**PR 전 정리 작업 (CT2)** — 코드 커밋이 doc 커밋과 깔끔히 분리돼 cherry-pick으로 깨끗한 PR 브랜치 구성 가능(구조적 재작업 불필요):
1. [high] 죽은 필드 제거 — `start_bias`/`mode`/`PhraseBiasMode` (H2)
2. [high] 한국어 주석 전량 영어화 (M2)
3. [high] beam>1 테스트 추가 (H3)
4. [med] dev-docs/·CLAUDE.md PR 브랜치에서 제외(코드 커밋 ~12개만 추림)
5. [med] `--log_throughput` empty=no-op 오버헤드 0 측정 후 PR 본문 첨부 (M3)
6. [low] `max_token_delta` 노출/명명 + debug counter 결정 (L5)

**PR 전 정리 작업 (faster-whisper)**: M4 실토크나이저 다언어/숫자/혼합 roundtrip 회귀 테스트. B1 로깅 + H1 검증 + H4 capability 체크는 해소됨.

---

## 5. 권장 우선순위 (두 트랙 분리)

**트랙 A — 자체 프로덕션 배포(현 포크, 통제된 config)**: 코어는 배포 가능 등급. B1/H1/H4는 해소됨. 죽은 필드(H2)는 너희가 안 쓰면 무해하나 혼란 방지로 정리 권장.

**트랙 B — upstream PR**: 위 §4 정리 목록. CT2부터(표면 정리 위주), faster-whisper는 로깅+검증 먼저.

**손대지 말 것**: 동시성·이식성·성능·수치 코어 — 이미 견고. 괜히 건드리면 회귀 위험.

---

### 부록 — 감사 산출물
서브에이전트 5종(동시성/API, 성능/메모리, 이식성/매트릭스, 테스트/유지보수, faster-whisper)의 file:line 근거 findings를 본 문서로 종합. 미확정(`needs-runtime-check`): sm_35 빌드 스모크(§1), GPU `--log_profiling`(M1), GPU e2e flip(M3), 실토크나이저 roundtrip(M4), min_prefix_len 음수 throw(H1).
