# CT2 KenLM BPE Fusion Implementation Plan

이 문서는 [ct2_kenlm_fusion_design.md](ct2_kenlm_fusion_design.md)의 결정을 현재 CTranslate2 코드에 적용하는 실행 계획이다.

정책과 계약의 SSOT는 design 문서다. 이 문서는 파일 위치, 변경 순서, 검증 방법만 다룬다.

## 1. Current Code Map

현재 확인한 주요 코드 위치는 다음이다.

| 영역 | 현재 파일 | 현재 역할 |
|---|---|---|
| Decoding API | `include/ctranslate2/decoding.h` | `DecodingOptions`, `SearchStrategy`, `BeamSearch` 선언 |
| Beam loop | `src/decoding.cc` | `BeamSearch::search()`, validation, sampler/search factory |
| Decoder output mapping | `include/ctranslate2/layers/decoder.h` | `to_output_word_id`, `to_original_word_id`, `is_in_output` |
| Decoder output update | `src/layers/decoder.cc` | `update_output_layer()`에서 padding slot을 token 0으로 채움 |
| Whisper C++ API | `include/ctranslate2/models/whisper.h` | `models::WhisperOptions` |
| Whisper wiring | `src/models/whisper.cc` | `WhisperReplica::generate()`에서 `DecodingOptions` 구성 |
| Python API | `python/cpp/whisper.cc` | `WhisperWrapper::generate()`와 pybind keyword/docstring |
| Build | `CMakeLists.txt` | options, source list, include/link list |
| C++ tests | `tests/model_test.cc`, `tests/decoding_test.cc`, `tests/CMakeLists.txt` | decoder mapping과 decoding behavior test |

## 2. Implementation Order

작업은 feature-off parity를 계속 유지하는 순서로 진행한다.

1. Decoder padding output id accessor
2. Fusion internal types and `DecodingOptions`
3. Validation and `BeamSearch` constructor wiring
4. Fake scorer driven candidate selection helper
5. LM state gather/prune wiring
6. Whisper options and Python API
7. KenLM optional build and scorer implementation
8. KenLM cache or simple loader decision
9. Integration tests and benchmarks

## 3. Step 1: Decoder Accessor

Files:

- `include/ctranslate2/layers/decoder.h`
- `src/layers/decoder.cc`
- `tests/model_test.cc`

Current behavior:

- `Decoder::update_output_layer()` pads selected ids with `0`.
- `to_original_word_id(padding_output_id)` can return `0`.
- `tests/model_test.cc` currently expects this legacy mapping.

Change:

- Add `_effective_output_size`.
- Set it before padding is appended.
- Expose `effective_output_size()` and `is_padding_output_id(size_t output_id)`.
- Do not change existing `to_original_word_id()` behavior. Add safe accessor instead.

Planned API:

```cpp
dim_t effective_output_size() const;

bool is_padding_output_id(size_t output_id) const;
```

Expected test updates:

- Keep existing `to_original_word_id(4) == 0` expectation for legacy compatibility.
- Add expectation that output id `4` is `is_padding_output_id(4) == true`.
- Add reset expectation that full-size output layer has no padding output id.

Risk:

- Reset path must clear `_effective_output_size` or restore it to current output size.

## 4. Step 2: Fusion Types

Files:

- Add `include/ctranslate2/lm_fusion.h`
- Add `src/lm_fusion.cc` only if non-header helpers are needed before KenLM
- Update `include/ctranslate2/decoding.h`
- Update `CMakeLists.txt` source list if `src/lm_fusion.cc` is added

Change:

- Define `LmFusionOptions`.
- Define `LmStateBatch`.
- Define `LmFusionScorer`.
- Add `DecodingOptions::lm_fusion` and `DecodingOptions::lm_fusion_scorer`.
- Add `BeamSearch` constructor fields for options and scorer.

Recommended header location:

```cpp
#include "ctranslate2/lm_fusion.h"
```

`DecodingOptions` additions:

```cpp
std::shared_ptr<const LmFusionScorer> lm_fusion_scorer = nullptr;
LmFusionOptions lm_fusion;
```

`BeamSearch` private fields:

```cpp
const LmFusionOptions _lm_fusion;
std::shared_ptr<const LmFusionScorer> _lm_fusion_scorer;
```

Risk:

- Avoid changing `SearchStrategy::search()` signature. That would ripple into GreedySearch and alternatives flow.

## 5. Step 3: Validation and Factory Wiring

Files:

- `src/decoding.cc`

Current locations:

- `validate_decoding_options()` validates standard options.
- `make_search_strategy()` builds `BeamSearch` without LM args.

Change:

- Add helper `lm_fusion_enabled(options)`.
- Add validation when fusion is enabled.
- Pass scorer/options into `BeamSearch` only.
- Keep GreedySearch unchanged.

Validation:

```text
alpha > 0
asr_topk > 0
scorer != nullptr
beam_size > 1
deterministic sampler path
return_alternatives == false
```

Vocabulary-size validation must happen inside `BeamSearch::search()` because `decoder.output_size()` is available there.

Current CT2 sampler factory selects `BestSampler` when `sampling_topk == 1` or `sampling_temperature == 0.0`. Fusion validation should reject only the random sampling path, not an otherwise deterministic `BestSampler` path.

## 6. Step 4: Candidate Selection Helper

Files:

- `src/decoding.cc`
- Optional later extraction to `src/lm_fusion.cc` if helper grows too large

Current code block in `BeamSearch::search()`:

```cpp
// Multiply by the current beam log probs.
...

// Flatten the probs into a list of candidates.
log_probs.reshape({cur_batch_size, -1});

// TopK candidates.
sampler(log_probs, topk_ids, topk_scores, num_candidates);

// Unflatten the ids.
StorageView gather_indices = unflatten_ids(topk_ids, _beam_size, vocabulary_size, is_expanded);
```

Change:

- If fusion is disabled, keep this path unchanged.
- If fusion is enabled, replace this block with `select_fused_candidates_topk_strict(...)`.
- The helper returns:

```text
topk_ids        CPU [cur_batch_size, num_candidates], output token ids
topk_scores     CPU [cur_batch_size, num_candidates], fused cumulative score
gather_indices  CPU [cur_batch_size * num_candidates]
candidate_states matching [cur_batch_size * num_candidates]
```

Algorithm:

1. Use row-wise TopK on `log_probs` before flattening by batch.
2. Copy row top-k ids/scores to CPU.
3. For each row candidate:
   - skip if `decoder.is_padding_output_id(output_id)`
   - convert `output_id` to `original_id`
   - if `original_id >= text_token_limit`, score delta is `0` and state is copied
   - otherwise call scorer and add `alpha * lm_score_ln`
4. For each batch, partial sort `beam_size * asr_topk` candidates.
5. Emit top `num_candidates` in the same shape expected by existing CT2 code.
6. `topk_ids` must contain output token ids, not flattened ids. `gather_indices` carries the selected beam origins.

Risk:

- GPU/CPU dtype conversion: `topk_scores.scalar_at<float>()` is already used later, but helper should avoid unnecessary type assumptions where possible.
- If valid candidates after padding skip are fewer than `num_candidates`, do not expand ASR top-k internally and do not silently fallback. The helper should fail with a clear runtime error. Dummy `-inf` candidates may be used only as an internal sentinel and must not become active beams.

## 7. Step 5: `expand_after_first_step`

Files:

- `src/decoding.cc`

Current code:

```cpp
const bool expand_after_first_step = (device == Device::CPU
                                      && num_candidates <= vocabulary_size);
```

Change:

```cpp
const bool expand_after_first_step = (!lm_fusion_enabled
                                      && device == Device::CPU
                                      && num_candidates <= vocabulary_size);
```

Reason:

- LM state alignment is simplest and safest when beam state is expanded from the beginning.

Risk:

- CPU-only first step performance will regress only when LM fusion is enabled.

## 8. Step 6: LM State Initialization

Files:

- `src/decoding.cc`
- `src/models/whisper.cc`
- `include/ctranslate2/decoding.h`

State initialization:

- When fusion is enabled, create `lm_states` with `batch_size * beam_size`.
- If prompt replay data is available, seed each batch state from prompt text history before beam replication.

Whisper prompt handling currently:

- `src/models/whisper.cc::WhisperReplica::generate()` splits prompt into `prompt_tokens` and `start_tokens`.
- `prompt_tokens` are passed to `_decoder->forward_prompt(...)`.
- `start_step` becomes `inputs.dim(1)`.

Implementation options:

1. Add `DecodingOptions::lm_initial_histories` as original token ids per batch.
2. In `WhisperReplica::generate()`, set it from prompt tokens that were forwarded before decode.
3. Filter history to `original_id < _eot_id` before passing it to decoding.
4. In `BeamSearch`, scorer initializes state by replaying these text histories before beam replication.

Risk:

- Prompt token boundaries must not include non-text Whisper control tokens in initial history. Previous-text prompt behavior beyond this limited replay remains a parity risk.

## 9. Step 7: Hard Prefix Handling

Files:

- `src/decoding.cc`

Current behavior:

- `update_sample_with_prefix()` runs after `sampler()` and `unflatten_ids()`.
- It can modify `topk_ids`, `topk_scores`, and `gather_indices`.

Change:

- Detect forced hard-prefix steps with:

```text
use_hard_prefix
&& any live batch i has step <= prefix_ids[batch_offset[i]].size()
```

- On those steps, use baseline sampler path and run `update_sample_with_prefix()`.
- After prefix update, build candidate LM states by advancing/copying from the final `topk_ids` and `gather_indices`.
- This is a whole-step bypass in v1. Per-batch fusion/baseline mixing is out of scope.

Risk:

- Precomputing candidate states before prefix update causes silent LM state drift.

## 10. Step 8: LM State Reorder

Files:

- `src/decoding.cc`

Current state update sequence:

```cpp
gather(gather_indices, active_beams);
gather_beam_flat(topk_ids, active_beams, _beam_size);
gather_beam_flat(topk_scores, active_beams, _beam_size);
gather_beam_flat(alive_seq, active_beams, _beam_size);
...
decoder.update_state(state, gather_indices, _beam_size, keep_batches.get());
```

Change:

- Before pruning finished batches, gather `candidate_lm_states` with `active_beams` through `LmFusionScorer::gather(...)`.
- If `next_batch_size != cur_batch_size`, prune gathered LM states with `non_finished_index` through `LmFusionScorer::keep_batches(...)`.
- Assign result back to `lm_states`.

Risk:

- The indexing source for LM state gather is `active_beams`, not `gather_indices` after it has been gathered.

## 11. Step 9: Whisper Public API

Files:

- `include/ctranslate2/models/whisper.h`
- `src/models/whisper.cc`
- `python/cpp/whisper.cc`

Changes:

- Add fields to `WhisperOptions`.
- Add matching Python keyword args to `WhisperWrapper::generate()`.
- Wire values into `DecodingOptions`.
- Set `text_token_limit = _eot_id`.
- Document that `scores` are fused scores when LM fusion is enabled.

Current Python binding:

- `WhisperWrapper::generate()` accepts `sampling_topk` and `sampling_temperature` as final kwargs.
- pybind `.def("generate", ...)` keyword list must be extended in the same order as the wrapper signature.

Risk:

- Python ABI/API order mismatch if function signature and pybind args diverge.

## 12. Step 10: KenLM Build and Scorer

Files:

- `CMakeLists.txt`
- Add `include/ctranslate2/kenlm_fusion.h` or merge into `include/ctranslate2/lm_fusion.h`
- Add `src/kenlm_fusion.cc`

CMake changes:

- Add `option(WITH_KENLM "Compile KenLM shallow fusion support" OFF)`.
- Add `KENLM_ROOT` cache path.
- Add `KENLM_MAX_ORDER` cache value matching the linked KenLM build.
- Find `lm/model.hh` or `lm/virtual_interface.hh`.
- Find KenLM libraries.
- Link KenLM compression dependencies: zlib, bzip2, lzma.
- Add `CT2_WITH_KENLM` compile definition only when enabled.
- Compile scorer source in all builds, with a `WITH_KENLM=OFF` runtime error stub.

Scorer behavior:

- Load KenLM binary.
- Build `original_token_id -> lm::WordIndex` table for `text_token_limit`.
- Use `BaseScore` or `FullScore`.
- Convert log10 to natural log with `std::log(10.f)`.
- Score skip tokens as `0` and copy state.

Risk:

- Actual KenLM library names and include layout must be verified against the local KenLM build.
- KenLM license and packaging policy must be checked before distribution.

## 13. Step 11: Cache Policy

Files:

- Optional `src/kenlm_cache.cc`
- Optional `include/ctranslate2/kenlm_fusion.h`

1차 choice:

- If speed of implementation matters, create scorer per request and document it as experimental.
- If multi-worker memory matters from day one, implement path-keyed shared cache.

Recommended operating design:

```text
model: shared_ptr<const KenlmModelHandle>, keyed by path
state: request/beam local
scorer wrapper: cheap object, can be per request
```

Scorer lifetime is fixed from v1:

```text
WhisperReplica::generate()
  -> create or fetch shared_ptr<const LmFusionScorer>
  -> set DecodingOptions::lm_fusion_scorer
  -> BeamSearch keeps shared_ptr but does not own KenLM cache policy
```

## 14. Test Plan

### 14.1 Decoder Mapping

File:

- `tests/model_test.cc`

Test:

- `effective_output_size()` is the non-padded size.
- `is_padding_output_id()` is true for padding slot.
- Existing original id mapping behavior stays unchanged.

### 14.2 Fake Scorer Beam Tests

Files:

- `tests/decoding_test.cc` or new `tests/lm_fusion_test.cc`
- `tests/CMakeLists.txt` if new file is added

Tests:

- LM-preferred ASR top-k candidate wins.
- ASR top-k outside token is not rescued.
- Special/timestamp skip copies state and score delta is zero.
- Beam reorder preserves LM state.
- Finished batch prune preserves LM state order.
- EOS/final result uses fused score.
- Candidate shortage fails explicitly.
- Unsupported modes throw validation errors.

### 14.3 KenLM Integration Tests

Condition:

- Run only when `WITH_KENLM=ON` and fixture binary exists.

Tests:

- `t<ID>` lookup works.
- log10 to ln conversion matches expected value.
- state advances for text token.
- state copies for skip token.

### 14.4 Python Smoke

Files:

- Existing Python test location should be checked before adding a new test.

Tests:

- `Whisper.generate()` accepts new kwargs.
- `WITH_KENLM=OFF` build gives clear failure when fusion is requested.
- Default kwargs preserve existing behavior.

## 15. Benchmark Plan

Compare on the same model/audio subset:

```text
baseline CT2
CT2 + LM fusion topk=20
CT2 + LM fusion topk=50
CT2 + LM fusion topk=100
```

Metrics:

- p50 latency
- p95 latency
- throughput
- average decode steps
- KenLM query count
- CPU time in fusion helper
- GPU utilization
- domain term recall/precision
- CER/WER

Expected query count:

```text
decode_steps * beam_size * asr_topk
```

Example:

```text
80 steps * beam 5 * topk 50 = 20,000 KenLM calls / utterance
```

## 16. Completion Criteria

Implementation is ready for CT2 fusion experiments when:

- `WITH_KENLM=OFF` default build preserves baseline behavior.
- `WITH_KENLM=ON` build succeeds with configured KenLM path.
- Python `Whisper.generate()` accepts LM fusion options.
- Fake scorer tests cover candidate selection and state reorder.
- Padding output id cannot be scored by LM.
- Small sample evaluation shows the same direction as HF POC term recall improvement.
- p95 latency and KenLM query count are documented.
