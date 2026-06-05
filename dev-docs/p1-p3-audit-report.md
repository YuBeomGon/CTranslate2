# P1–P3 Audit Report — Whisper Positive Phrase Bias

> Date: 2026-06-04
> Scope: P1–P3 design ↔ code audit (read + verify only; no source/doc modified except this report)
> Baseline docs: `dev-docs/SSOT.md`, `dev-docs/p1-impl-plan.md`, `dev-docs/p2-impl-plan.md`, `dev-docs/p3-impl-plan.md`, `dev-docs/p3-completion-report.md`

Verdict legend: ✅ PASS · ⚠️ NOTE (harmless/observation) · ❌ BUG (real defect)

---

## P1 — CPU bias + reverse trie

| # | Item | Verdict | Evidence | Detail |
|---|------|---------|----------|--------|
| 1 | `PhraseBiasEntry` plain type (ids/step_bias/min_prefix_len), no models dep | ✅ | `include/ctranslate2/decoding_utils.h:139-144` | Matches SSOT §6 low-level type exactly. |
| 2 | Trie `add` skips `ids.size()<2` (1-token = no continuation, no start_bias) | ✅ | `src/decoding_utils.cc:71-74` | `if (ids.size() < 2) return;`. Test `TrieSkipsTooShortEntry` confirms. |
| 3 | Trie `add` skips `j < min_prefix_len` | ✅ | `src/decoding_utils.cc:76-78` | Rules start at `j=1` (continuation-only) and the `j<min_prefix_len` guard filters short matches. Test `TrieMinPrefixLenSkipsShortMatch`. |
| 4 | Continuation-only / no start_bias (first token never boosted) | ✅ | `src/decoding_utils.cc:76` (`j` from 1) | No rule ever maps to `ids[0]`. Satisfies SSOT §7-2. |
| 5 | Reverse trie: shared prefix returns BOTH continuations, lookup walks tail backwards and sums | ✅ | `src/decoding_utils.cc:79-97`; test `tests/decoding_test.cc:34-51` | `[A,B]→{C,D}` summed; `[A]→{B:0.5}` (0.25+0.25). lookup loop `for (k=tail_len; k-->0;)` walks backwards, `out[...] += action`. |
| 6 | Unique-index contract: dedupe/sum via `std::map` BEFORE `indexed_add` | ✅ | `src/decoding_utils.cc:139-145` | Per-row `std::map<size_t,float> boost` collapses duplicate tokens; one `(flat_index,delta)` per token emitted. No duplicate flat index can reach `indexed_add` (each `b*vocab+token` is unique across the b/token loop). Satisfies SSOT §7-7. |
| 7 | Final per-token clamp AFTER summation, `std::min(sum, max_token_delta)` | ✅ | `src/decoding_utils.cc:142` | `const float delta = std::min(kv.second, _max_token_delta);` applied to the already-summed map value. CT2's job (sum-then-clamp) is correct; steps 1–3 of SSOT §2 are caller-side per §0.1. Test `ProcessorOverlapSumsAndClamps` (0.6+0.6→1.0). |
| 8 | `max_token_delta` default 2.0 | ✅ | `include/ctranslate2/decoding_utils.h:169,172` | Both ctors default `max_token_delta = 2.0f`. |
| 9 | `apply_first()==false` (runs after no-speech) | ✅ | `decoding_utils.h:175` | Returns false → runs in non-first pass. |
| 10 | step-0 / null sequences guard | ✅ | `src/decoding_utils.cc:128-129` | `if (!sequences) return;`. Test `ProcessorNoMatchAndNullAreNoOp`. |
| 11 | `indexed_add` CPU handles half via float (not raw `+=`) | ✅ | `src/cpu/primitives.cc:67-72` | `x[i] = static_cast<T>(static_cast<float>(x[i]) + static_cast<float>(deltas[i]))`. Avoids missing `operator+=` on bf16/fp16. |
| 12 | CPU `indexed_add` explicit instantiation present (else link fail) | ✅ | `src/cpu/primitives.cc:1178-1179` | Inside `DECLARE_IMPL` next to `indexed_fill`. |
| 13 | `indexed_add` declared next to `indexed_fill` | ✅ | `include/ctranslate2/primitives.h:21,23` | Same signature `(T*, const T*, const int32_t*, dim_t)`. |
| 14 | `to_phrase_bias_entries` flattens PhraseBias→entries | ✅ | `src/models/whisper.cc:232-245`; header `whisper.h:35-36` | Copies ids/step_bias/min_prefix_len. Test `ConvertModelOptionToEntries`. |
| 15 | generate injection only when bias present, after no-speech/timestamp processors | ✅ | `src/models/whisper.cc:359-364` | Injected after `ApplyTimestampRules` block (which follows the no-speech processor at :340-345). |
| 16 | Empty no-op (no processor when empty) | ✅ | `src/models/whisper.cc:359-362` | `bias_trie` stays null when both `compiled_phrase_bias_trie` is null and `phrase_biases` empty → no processor. SSOT §7-5. |
| 17 | leading-space NOT stripped in CT2 (no tokenization in CT2) | ✅ | grep over `decoding_utils.cc`, `whisper.cc`, `python/cpp/whisper.cc` | No string/space/encode logic in the bias path; only a docstring mentions "tokenization" as caller responsibility. CT2 consumes ids only (SSOT §0.1). |
| 18 | Uses `indexed_add` via `DEVICE_AND_TYPE_DISPATCH` (P2-ready, interface unchanged) | ✅ | `src/decoding_utils.cc:156-159` | Dispatch path identical to `RepetitionPenalty`/`DisableTokens` exemplars. |

**P1 tests run:** `PhraseBiasTest.*` (10) + `CPU/PrimitiveTest.IndexedAdd` (1) → **11 passed**.

---

## P2 — GPU parity

| # | Item | Verdict | Evidence | Detail |
|---|------|---------|----------|--------|
| 1 | CUDA `indexed_add_kernel` mirrors `penalize_previous_tokens_kernel`, grid-stride, no atomics | ✅ | `src/cuda/primitives.cu:70-84` | Grid-stride loop, `x[idx] = float(x[idx]) + float(deltas[i])`, comment cites SSOT §7-7 unique-index → no race → no atomic. |
| 2 | Half/bf16 via float cast | ✅ | `src/cuda/primitives.cu:81-82` | `static_cast<float>` on `device_cast`-ed pointer; completion report confirms it compiled across all `Common` arches without fallback. |
| 3 | Launch uses `device_cast`/`get_cuda_stream`, `block(32)` like exemplar | ✅ | `src/cuda/primitives.cu:86-99` | `dim3 block(32)`, `cuda::get_cuda_stream()`, `cuda::device_cast(...)`. `num_indices==0` early return. |
| 4 | CUDA explicit instantiation present (else link fail) | ✅ | `src/cuda/primitives.cu:794-795` | Inside `DECLARE_IMPL` next to `indexed_fill`. |
| 5 | `IndexedAdd` device-parameterized `TEST_P` | ✅ | `tests/primitives_test.cc:54-65`, instantiated CPU :67 / CUDA :69 | Replaces the P1 plain test; runs on both devices. |
| 6 | Processor parity FloatType-parameterized, batch=2, shared prefix, overlap | ✅ | `tests/decoding_test.cc:149-190` | row0 `[...,1]→token2 +=0.6` (sum), row1 `[1,2]→token3,4 +=0.3`. CPU fp32 (1e-5); CUDA fp32/fp16(1e-2)/bf16(4e-2) under `#ifdef CT2_WITH_CUDA`. |
| 7 | Interface unchanged in P2 (P1 processor/trie code untouched) | ✅ | `primitives.h:23` signature identical | Only CUDA impl + tests added, as plan File Structure asserts. |

**P2 GPU parity:** Not re-run in this audit (no CUDA build present in `build/`; this is the CPU-only test dir). Parity correctness is verified statically against the kernel + tests and the p2 plan's recorded run (`38a5b4c0`/`424a7450`, 4 parity PASS + portable `Common` compile). The CPU half of the parity (`CPU/PhraseBiasProcessorFPTest.FLOAT32`) is covered by the CPU build.

---

## P3 — Python binding + persistent trie

| # | Item | Verdict | Evidence | Detail |
|---|------|---------|----------|--------|
| 1 | `build_phrase_bias_trie(entries)→shared_ptr<const PhraseBiasTrie>` | ✅ | `decoding_utils.h:161-163`; `src/decoding_utils.cc:101-107` | Builds once; entries filtered inside `add`. |
| 2 | shared-trie ctor delegates from entries ctor (P1 tests stay green) | ✅ | `src/decoding_utils.cc:109-120` | `(entries)` ctor delegates to `(trie)` ctor via `build_phrase_bias_trie`. `apply` uses `_trie->lookup`. Test `SharedTrieReusedByProcessors`. |
| 3 | `WhisperOptions.compiled_phrase_bias_trie` (`shared_ptr<const>`) added | ✅ | `include/ctranslate2/models/whisper.h:90-92` | Documented "takes precedence over phrase_biases". |
| 4 | `WhisperReplica::generate` uses compiled trie if present, else builds from entries; NO rebuild on ctor path | ✅ | `src/models/whisper.cc:359-364` | `bias_trie = options.compiled_phrase_bias_trie;` first; only builds when compiled trie absent AND raw `phrase_biases` non-empty. Constructor path (None) injects the cached trie → no per-generate rebuild. |
| 5 | `WhisperWrapper` holds `shared_ptr<const PhraseBiasTrie> _compiled_trie`, built once in ctor | ✅ | `python/cpp/whisper.cc:34-38, 151` | Custom ctor (replaces `using ReplicaPoolHelper::ReplicaPoolHelper;`) builds trie once when `phrase_biases` provided & non-empty. |
| 6 | ReplicaPool race safety (trie is `const`, read-only) | ✅ | `whisper.cc:104` injects `_compiled_trie` into options; trie type is `shared_ptr<const>` everywhere | Replicas only read; trie immutable. No mutable setter (parked per plan). |
| 7 | 3-way `generate(phrase_biases=)`: None→model trie, []→disable, [...]→override | ✅ (via simplification) | `python/cpp/whisper.cc:101-105` + core guard `src/models/whisper.cc:360` | See "Plan-vs-code deviations" #1 — implemented branch is shorter than the plan text but semantically equivalent: `[]` sets empty `options.phrase_biases` and leaves `compiled_phrase_bias_trie` null → core guard `!phrase_biases.empty()` is false → no processor = disabled. |
| 8 | Empty no-op (null trie → no processor → baseline) | ✅ | `whisper.cc:359-362`; tests `test_phrase_bias.py:130-131,171-173` | None on a no-bias model and `[]` both yield baseline output. |
| 9 | Python types expose only ids/step_bias/min_prefix_len (start_bias/mode NOT exposed) | ✅ | `python/cpp/whisper.cc:159-189` | `def_readwrite` only for ids/step_bias/min_prefix_len/token_paths. SSOT P3 scope (parked fields hidden). |
| 10 | re-export in models `__init__` | ✅ | `python/ctranslate2/models/__init__.py:7-8` | `PhraseBias`, `PhraseBiasPath` re-exported. |
| 11 | py::init + generate arg ordering, docstrings document 3-way | ✅ | `whisper.cc:254-299` (init), `:332-392` (generate) | `phrase_biases=py::none()` last arg in both; generate docstring states None/[]/[...] semantics. |
| 12 | Tests cover None/[]/[...] incl. `[]`==baseline on a constructor-bias model | ✅ | `python/tests/test_phrase_bias.py:120-173` | `test_percall_...` (per-call None/[]/override) and `test_constructor_..._persistent` (ctor effect, None==biased, `[]`==baseline on a ctor-bias model, out1==out2 stability). |

**P3 tests:** C++ `PhraseBiasTest.*` re-run here → 10 passed (includes `SharedTrieReusedByProcessors`). Python `test_phrase_bias.py` not re-run in this audit (requires the editable install + `LD_PRELOAD`/`LD_LIBRARY_PATH` shim described in the completion report, item 2; out of read-only scope). Completion report records 6 passed.

---

## Bugs / risks found

**None found.** No correctness defects, no contract violations, no link-instantiation gaps. The unique-index contract (the critical CUDA race concern) is upheld by the per-row `std::map` dedup before every `indexed_add` call.

Minor non-blocking observations (not bugs):

- **R1 (NOTE, low):** `to_phrase_bias_entries` copies `step_bias` and `min_prefix_len` but silently ignores `start_bias` and `mode` on `PhraseBiasPath`. This is by design (start_bias parked = 0, Block mode parked), and the Python binding does not expose those fields, so they can only be non-default via C++ direct use. No action needed for P1–P3; revisit if/when negative/Block is unparked.
- **R2 (NOTE, low):** `apply` casts `b * vocab_size + kv.first` (a `size_t` token id summed into a row offset) to `int32_t` for the flat index. For Whisper vocab (~51865) × small batch this never overflows int32. Consistent with `DisableTokens` which uses the same flat-index scheme. No action needed.

---

## Plan-vs-code deviations

### Real (but harmless-equivalent)

1. **P3 Task 5 Step 4 — 3-way branch simplified.** Plan text:
   ```cpp
   if (phrase_biases) {
     if (!phrase_biases->empty())
       options.phrase_biases = *phrase_biases;   // [...] override
     // [] : disable — inject nothing
   } else {
     options.compiled_phrase_bias_trie = _compiled_trie;  // None
   }
   ```
   Implemented (`python/cpp/whisper.cc:101-105`):
   ```cpp
   if (phrase_biases) {
     options.phrase_biases = *phrase_biases;     // covers [...] and []
   } else {
     options.compiled_phrase_bias_trie = _compiled_trie;
   }
   ```
   For `[]`, the implementation assigns an empty vector to `options.phrase_biases` and leaves `compiled_phrase_bias_trie` null. The core guard `if (!bias_trie && !options.phrase_biases.empty())` (`src/models/whisper.cc:360`) is then false → no processor built → bias disabled for that call. **Semantically identical** to the plan's explicit inner `if`. Verified by `test_phrase_bias.py:171-173` (`[]` on a ctor-bias model == baseline). Harmless.

### Harmless / expected

2. **P2 Task 1 baseline build ordering** — recorded in p2-impl-plan "실행 노트" #1: P1's `DEVICE_AND_TYPE_DISPATCH` already forces `primitives<Device::CUDA>::indexed_add`, so the "baseline CUDA build before kernel" step actually link-failed until the kernel was added. Documented, expected (SSOT §13 layering fact). Not a code defect.

3. **P3 Task 6 test target selection** — p3-impl-plan used a fixed `step_bias=50.0` with `target=(base[1]+1)%1000` at index 1. The completion report (item 3) and the implemented test instead auto-scan baseline `return_logits_vocab` for a step whose 2nd-place margin is within the final clamp via `_select_flippable_target` (`test_phrase_bias.py:87-107`). This is a **test robustness improvement**, not a contract change: the implementation's final per-token clamp (`+2.0`) is unchanged; the test simply stops assuming a specific step is flippable. `step_bias=50.0` is still passed but gets clamped to 2.0 by the processor — the test relies on the clamped value, which is the correct contract. Sound.

4. **`PhraseBiasPath` C++ struct carries extra parked fields** (`start_bias`, `mode`) beyond the SSOT §6 low-level `PhraseBiasEntry` sketch (`whisper.h:21-27`). This is the model-level type (allowed to be richer); the low-level `PhraseBiasEntry` stays minimal as specified. Expected.

---

## Overall verdict

**P1–P3 are sound to build P4 on.** The design in SSOT (continuation-only positive bias, reverse trie, sum-then-clamp at `max_token_delta=2.0`, unique-index `indexed_add` contract, load-time persistent immutable trie with read-only replica injection, ids-only Python API with 3-way `generate` semantics) is faithfully implemented across `decoding_utils`, `primitives` (CPU+CUDA), `whisper.{h,cc}`, and the Python binding. No bugs were found; the only deviation that touches behavior (the simplified 3-way branch) is provably equivalent and test-covered. The CUDA race hazard called out in SSOT §7-7 is correctly prevented by CPU-side `std::map` dedup before every dispatch.

P4 (faster-whisper tokenizer compile: string → leading-space/non-leading-space token-id paths + step_bias distribution + roundtrip validation) can proceed against the stable `PhraseBias(token_paths=[PhraseBiasPath(ids, step_bias, min_prefix_len)])` Python API with confidence that the CT2 consumer side is correct.
