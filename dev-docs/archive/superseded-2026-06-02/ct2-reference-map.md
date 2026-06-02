# CT2 Internal Reference Map (구현 exemplar)

> 외부 문서/CUDA 튜토리얼 보지 말 것. **각 구현 조각마다 아래 기존 CT2 코드를 먼저 읽고 그대로 따라 한다.**
> 모든 위치는 실제 코드 대조로 검증됨 (2026-06-02). 새 스타일 만들기 금지(SSOT §3-6).

## A. LogitsProcessor 추가 (PhraseBiasProcessor) — Phase 1

| 봐야 할 것 | 위치 | 왜 |
|---|---|---|
| `LogitsProcessor` base, `apply()` 시그니처, `apply_first()` | `include/ctranslate2/decoding_utils.h:74` | 우리 processor가 상속할 인터페이스 |
| `SuppressSequences::apply` (suffix 매칭 `std::equal(end-N, end, ...)`) | `src/decoding_utils.cc` | trie 대신 naive로 먼저 이해. 우리 continuation 매칭의 원형 |
| **`RepetitionPenalty::apply`** (logits in-place 수정 템플릿) | `src/decoding_utils.cc` | **soft bias 구현의 정답 템플릿** — `ops::Gather` + `DEVICE_AND_TYPE_DISPATCH`로 primitive 호출 |
| `DisableTokens` (CPU 직접대입 / GPU flat-index 수집) | `include/ctranslate2/decoding_utils.h:36` | device별 sparse 쓰기 패턴 참고 |
| generate 주입 지점 (`logits_processors.emplace_back`) | `src/models/whisper.cc:325-340` | no-speech/timestamp 등록 패턴. 우리도 동일하게, empty면 skip |
| processor 순서 (`make_logits_processors`) | `src/decoding.cc:1091` | apply_first=false → 맨 뒤 실행 |

## B. 새 primitive `indexed_add` 추가 — Phase 1(CPU) / Phase 2(GPU)

**핵심: `indexed_fill`이 거의 그대로 템플릿.** `=` → `+=`, 스칼라 `a` → per-index `deltas` 배열로만 바꾸면 됨.

| 봐야 할 것 | 위치 | 메모 |
|---|---|---|
| `indexed_fill` 선언 | `include/ctranslate2/primitives.h:21` | `static void indexed_fill(T* x, T a, const int32_t* indices, dim_t num_indices)` |
| `indexed_fill` **CPU** 구현 | `src/cpu/primitives.cc:60` | 여기 mirror → `indexed_add` CPU |
| `indexed_fill` **CUDA** 구현 | `src/cuda/primitives.cu:63` | 여기 mirror → `indexed_add` CUDA |
| `penalize_previous_tokens` **CUDA 커널** (`__global__`) + launch | `src/cuda/primitives.cu:284`, launch `:302/311` | sparse score 수정 커널의 grid/block/stream 패턴 |
| `penalize_previous_tokens` CPU | `src/cpu/primitives.cc:411` | CPU sparse 수정 패턴 |
| **명시적 인스턴스화 매크로** (T별) | CUDA `src/cuda/primitives.cu:762`, CPU `src/cpu/primitives.cc:1168` | `indexed_add`도 여기에 같은 형식으로 추가해야 link됨 (float/float16 등) |

**제안 시그니처** (indexed_fill과 동형, delta 배열만 추가):
```cpp
static void indexed_add(T* x, const T* deltas, const int32_t* indices, dim_t num_indices);
```

## C. device/type dispatch — Phase 1/2

| 매크로 | 위치 | 용도 |
|---|---|---|
| `DEVICE_AND_TYPE_DISPATCH(device, dtype, STMTS)` | `src/dispatch.h:6` | processor에서 `primitives<D>::indexed_add<T>(...)` 호출 시 (RepetitionPenalty가 이 패턴 사용) |
| `DEVICE_DISPATCH` | `src/device_dispatch.h:22` | device만 분기 |
| `TYPE_DISPATCH` | `src/type_dispatch.h:60` | dtype만 분기 |

## D. GPU parity 시 주의 (Phase 2)

- float16 등은 **bitwise 동일 기대 금지** → tolerance(allclose). float32는 타이트.
- 같은 `(row, token)` 다중 bias → **합산 후** 적용 (overlap). GPU에서 atomic 불확실성 줄이려면 dedupe 후 unique index 1회 add (RepetitionPenalty/indexed_fill가 unique index 다루는 방식 참고).
- memory layout: logits는 row-major `[batch*beam, vocab]`. flat index = `row * vocab + token` (`DisableTokens::add` 참고).

## 한 줄 요약

GPU가 어렵게 느껴지면 → **`indexed_fill`(set)을 복사해서 `indexed_add`(+=)로 만들고, `penalize_previous_tokens` 커널의 launch 패턴을 빌린다.** 그 이상의 새 CUDA 코드는 짜지 않는다.
