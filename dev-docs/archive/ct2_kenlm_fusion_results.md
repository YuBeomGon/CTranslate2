# CT2 KenLM BPE Fusion Results

This document is the release-readiness snapshot for KenLM BPE fusion.
The design and IO contract SSOT is [ct2_kenlm_fusion_design.md](ct2_kenlm_fusion_design.md).

## 1. Evidence Checked

| repo | document/artifact | checked evidence |
|---|---|---|
| `../ct2-lm-fusion-runner` | `docs/poc_results_ct2.md` | CT2 alpha sweep, CER/WER, term recall/precision, RTF |
| `../ct2-lm-fusion-runner` | `runs/eval_validation_honest/*summary.jsonl` | raw summary metrics |
| `../stt-wrapper` | `docs/SSOT.md`, `docs/design.md` | Python wrapper API boundary, CT2 fusion kwargs, speed notes |

Some referenced repos contain internal absolute paths and experiment logs. Public release text should use only sanitized summaries.

## 2. Evaluation Setup

| item | value |
|---|---|
| CT2 commit | `943f41c0` |
| runner commit | `c9a6f69` |
| backend | patched CTranslate2 `WITH_KENLM=ON`, ct2 `4.7.2` |
| model | `openai/whisper-large-v3-turbo`, fp16 CUDA |
| decode | beam `5`, n-best `5`, deterministic |
| fusion | `score = ASR_logprob + alpha * LM_logprob`, `asr_topk=50` |
| LM | honest train-text only, 5-gram BPE-id KenLM |
| dataset | AIG validation, 420 files, 4 long-audio files skipped |
| audio | 5,333.9 s, about 89 minutes |

Evaluation/currently checked environment:

| item | value |
|---|---|
| OS/kernel | Ubuntu 24.04 family, Linux `6.17.0-35-generic` |
| CPU | Intel Core i7-14700K, 20 cores / 28 threads |
| RAM | 62 GiB |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| GPU check | based on the runner result document and `lspci`; `nvidia-smi` failed in the current shell because it could not communicate with the driver |

## 3. Alpha Sweep

| alpha | CER | WER | term recall | precision | insertion | repeated | RTF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.1153 | 0.3337 | 0.822 | 0.966 | 0.048 | 0.000 | 0.0154 |
| 0.05 | 0.1074 | 0.3032 | 0.851 | 0.971 | 0.039 | 0.000 | 0.0194 |
| 0.10 | 0.1020 | 0.2865 | 0.869 | 0.971 | 0.035 | 0.000 | 0.0194 |
| 0.15 | 0.0964 | 0.2684 | 0.887 | 0.969 | 0.031 | 0.000 | 0.0193 |
| 0.20 | 0.0937 | 0.2558 | 0.895 | 0.971 | 0.028 | 0.000 | 0.0194 |
| 0.30 | 0.1006 | 0.2571 | 0.912 | 0.974 | 0.025 | 0.000 | 0.0194 |
| 0.40 | 0.1098 | 0.2615 | 0.908 | 0.970 | 0.024 | 0.005 | 0.0188 |
| 0.60 | 0.1766 | 0.3223 | 0.893 | 0.969 | 0.019 | 0.024 | 0.0183 |

## 4. Interpretation

- The KenLM BPE fusion effect is reproduced in the CT2 path.
- Best CER is at `alpha=0.20`: baseline `0.1153` improves to `0.0937`.
- Best term recall is at `alpha=0.30`: baseline `0.822` improves to `0.912`.
- Repetition starts at `alpha=0.40`; do not use `alpha>=0.40` as an operational default.
- Fusion-on cost is about fixed +26% latency: baseline RTF `0.0154` to about `0.0194`.
- This evaluation only measured `asr_topk=50`. `20/100`, p50/p95 latency, KenLM query count, and CPU-side fusion time remain separate measurements.

Recommended values:

| purpose | alpha |
|---|---:|
| best CER / default report value | 0.20 |
| domain term recall priority | 0.30 |
| conservative operation | 0.05-0.10 |
| forbidden range | >=0.40 |

## 5. Wrapper/API Check

The `../stt-wrapper` public wrapper passes only these kwargs to CT2:

```python
lm_fusion_model_path
lm_fusion_alpha
lm_fusion_asr_topk
lm_fusion_debug
```

`lm_fusion_beta` is not supported.
Fusion is off and baseline decode is used when `lm_fusion_model_path` is empty or `lm_fusion_alpha <= 0`.

The Python smoke is not a quality evaluation. Its purpose is to verify that a `WITH_KENLM=ON` build/install exposes the new Python kwargs and can call `Whisper.generate`.

```python
import ctranslate2
import numpy as np
import transformers

processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = ctranslate2.models.Whisper(
    "/path/to/ct2-whisper-large-v3-turbo",
    device="cuda",
    compute_type="float16",
)

# Smoke only: 1 second of silence. Real decode/resample/windowing is caller-owned.
audio = np.zeros(16000, dtype=np.float32)
inputs = processor(audio, return_tensors="np", sampling_rate=16000)
features = ctranslate2.StorageView.from_array(inputs.input_features)

prompt = processor.tokenizer.convert_tokens_to_ids(
    ["<|startoftranscript|>", "<|ko|>", "<|transcribe|>", "<|notimestamps|>"]
)

results = model.generate(
    features,
    [prompt],
    beam_size=5,
    num_hypotheses=1,
    sampling_temperature=0.0,
    lm_fusion_model_path="/path/to/domain.binary",
    lm_fusion_alpha=0.20,
    lm_fusion_asr_topk=50,
    lm_fusion_debug=False,
    return_scores=True,
)
```

## 6. Cache Decision

The CT2 runtime loader uses a process-local path-keyed shared cache.
The cache key is canonical path plus `text_token_limit`.

This removes repeated KenLM model reloads when the same `.binary` path is passed on repeated transcribe/generate calls.
It does not remove per-beam-step KenLM scoring cost. Treat the +26% RTF cost above as per-step scoring overhead, and treat the cache as a repeated-request load-overhead optimization.

## 7. Release Readiness

An internal/experimental release is reasonable.

Evidence:

- CT2 commit `943f41c0` shows quality improvement in the alpha sweep.
- `WITH_KENLM=ON/OFF` builds and LM-fusion-targeted C++ tests passed.
- `alpha=0` or empty path disables fusion.
- KenLM scorer cache is implemented and covered by a same-path reuse test.

Recommended remaining work before a broader public release:

- Run the Python smoke above against the final wheel/install artifact.
- Triage the existing full C++ suite CPU GEMM failure separately or document the release scope.
- Remove internal paths/dataset names from the public README and keep only install, build, limitations, and sanitized metrics.
- Add `asr_topk=20/100` and p50/p95 latency if broader performance claims are needed.

## 8. KenLM License / Packaging

The local KenLM checkout `LICENSE` identifies KenLM as `LGPL-2.1-or-later`.
It is free to use, but binary distribution must preserve license notices and respect the obligations implied by the chosen linking/distribution method.

Packaging policy for this branch/release:

- The useful release artifact for this work is a KenLM-enabled CTranslate2 build, i.e. `WITH_KENLM=ON`.
- The release artifact should include or link KenLM explicitly and include a KenLM `LGPL-2.1-or-later` license notice.
- Prefer dynamic linking or a Docker/internal image where linked KenLM libraries and license notices are explicit.
- Do not silently vendor/static-link KenLM into a public wheel without a license/relinking review.
- KenLM `.binary` LM files are model/domain artifacts and are not bundled into the CTranslate2 package.
