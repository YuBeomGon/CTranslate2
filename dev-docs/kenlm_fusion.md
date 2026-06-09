# Whisper KenLM BPE Fusion

이 문서가 현재 Whisper KenLM BPE fusion 브랜치의 SSOT이다.
구현 히스토리, 체크리스트, 과거 설계 초안은 [archive/](archive/)에 보관한다.

## 1. Purpose

이 브랜치는 CTranslate2 Whisper beam search 내부에 KenLM BPE shallow fusion을 추가한다.

목적은 Whisper BPE token-id 기반 KenLM으로 ASR 후보를 재채점해 도메인 long-tail 용어 recall을 높이는 것이다.

Fusion score:

```text
fused_score = CT2_ASR_logprob + beam_cumulative_score + alpha * KenLM_BPE_logprob_ln
```

KenLM은 Whisper token id를 pseudo-word로 변환한 corpus로 학습한다.

```text
Whisper token id 1234 -> KenLM word "t1234"
```

## 2. Release Artifact

이 브랜치의 의미 있는 릴리즈 artifact는 KenLM-enabled CTranslate2 build다.

```text
WITH_KENLM=ON
```

기본 upstream-compatible build는 `WITH_KENLM=OFF`를 유지할 수 있지만, 이 경우 fusion 요청은 명확한 runtime error를 낸다.

```text
KenLM fusion requires CTranslate2 built with WITH_KENLM=ON
```

예시 빌드:

```bash
cmake -S . -B build-kenlm \
  -DWITH_KENLM=ON \
  -DKENLM_ROOT=/path/to/kenlm
cmake --build build-kenlm
```

## 3. Python API

`ctranslate2.models.Whisper.generate()`에 다음 kwargs가 추가된다.

```python
lm_fusion_model_path: Optional[str] = None
lm_fusion_alpha: float = 0
lm_fusion_asr_topk: int = 50
lm_fusion_debug: bool = False
```

Fusion off 조건:

```text
lm_fusion_model_path empty or lm_fusion_alpha <= 0
```

호출 예시:

```python
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

`return_scores=True`일 때 fusion enabled 상태의 반환 score는 ASR-only score가 아니라 fused cumulative score다.
`return_logits_vocab`는 기존 의미를 유지하며, 반환 logits/log-probs는 LM fusion 전 CT2 model 출력이다.

`lm_fusion_beta`는 지원하지 않는다.

## 4. Runtime Behavior

1차 구현은 `topk_strict` 방식이다.

- 각 active beam row에서 ASR top `lm_fusion_asr_topk` 후보만 가져온다.
- 그 후보 안에서 KenLM score를 더해 batch top `num_candidates`를 다시 고른다.
- ASR top-k 밖 token은 어떤 LM 점수로도 rescue하지 않는다.
- Whisper text token만 LM state를 advance하고 LM score를 더한다.
- special/timestamp/control token은 LM score 0, state copy로 처리한다.

KenLM scorer는 process-local path-keyed shared cache를 사용한다.

- cache key: canonical path + `text_token_limit`
- cache value: `shared_ptr<const LmFusionScorer>`
- per-request/per-beam LM state는 공유하지 않고 매 요청 새로 만든다.

이 cache는 같은 `.binary` path를 반복 호출할 때 KenLM model reload 비용을 제거한다.
beam step마다 발생하는 KenLM scoring 비용 자체를 없애지는 않는다.

## 5. Benchmark Snapshot

확인한 평가 조건:

| item | value |
|---|---|
| backend | patched CTranslate2 `WITH_KENLM=ON`, ct2 `4.7.2` |
| model | `openai/whisper-large-v3-turbo`, fp16 CUDA |
| decode | beam `5`, n-best `5`, deterministic |
| fusion | `asr_topk=50` |
| LM | honest train-text only, 5-gram BPE-id KenLM |
| audio | 5,333.9 s, about 89 minutes |
| CPU | Intel Core i7-14700K |
| GPU | NVIDIA GeForce RTX 4080 SUPER |

Alpha sweep summary:

| alpha | CER | WER | term recall | precision | repeated | RTF |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.1153 | 0.3337 | 0.822 | 0.966 | 0.000 | 0.0154 |
| 0.05 | 0.1074 | 0.3032 | 0.851 | 0.971 | 0.000 | 0.0194 |
| 0.10 | 0.1020 | 0.2865 | 0.869 | 0.971 | 0.000 | 0.0194 |
| 0.15 | 0.0964 | 0.2684 | 0.887 | 0.969 | 0.000 | 0.0193 |
| 0.20 | 0.0937 | 0.2558 | 0.895 | 0.971 | 0.000 | 0.0194 |
| 0.30 | 0.1006 | 0.2571 | 0.912 | 0.974 | 0.000 | 0.0194 |
| 0.40 | 0.1098 | 0.2615 | 0.908 | 0.970 | 0.005 | 0.0188 |
| 0.60 | 0.1766 | 0.3223 | 0.893 | 0.969 | 0.024 | 0.0183 |

권장값:

| purpose | alpha |
|---|---:|
| CER 최저 / 기본 보고값 | 0.20 |
| domain term recall 우선 | 0.30 |
| 보수 운영 | 0.05-0.10 |
| 금지 구간 | >=0.40 |

해석:

- CT2 경로에서도 KenLM BPE fusion 효과가 재현됐다.
- `alpha=0.20`에서 CER가 baseline `0.1153`에서 `0.0937`로 개선됐다.
- `alpha=0.30`에서 term recall이 baseline `0.822`에서 `0.912`로 개선됐다.
- `alpha=0.40`부터 repetition이 발생하므로 운영 기본값으로 쓰지 않는다.
- fusion on 비용은 RTF `0.0154 -> 0.0194` 수준으로 약 +26%다.

## 6. Python Smoke

Python smoke는 품질 평가가 아니라 `WITH_KENLM=ON` build/install에서 새 kwargs가 실제 호출 가능한지 확인하는 절차다.

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

audio = np.zeros(16000, dtype=np.float32)
inputs = processor(audio, return_tensors="np", sampling_rate=16000)
features = ctranslate2.StorageView.from_array(inputs.input_features)

prompt = processor.tokenizer.convert_tokens_to_ids(
    ["<|startoftranscript|>", "<|ko|>", "<|transcribe|>", "<|notimestamps|>"]
)

model.generate(
    features,
    [prompt],
    beam_size=5,
    num_hypotheses=1,
    sampling_temperature=0.0,
    lm_fusion_model_path="/path/to/domain.binary",
    lm_fusion_alpha=0.20,
    lm_fusion_asr_topk=50,
    return_scores=True,
)
```

현재 상태: Python smoke 호출 예시는 문서화되어 있지만, 최종 wheel/install artifact 대상 실행은 release 직전 1회 더 필요하다.

## 7. License and Packaging

KenLM checkout의 `LICENSE` 기준 KenLM 본체는 `LGPL-2.1-or-later`이다.
무료 사용은 가능하지만, binary distribution에서는 license notice와 linking/distribution 조건을 지켜야 한다.

정책:

- KenLM-enabled release artifact는 KenLM을 명시적으로 포함하거나 링크한다.
- release artifact에는 KenLM `LGPL-2.1-or-later` license notice를 포함한다.
- linked KenLM library와 license notice가 명확한 dynamic link 또는 Docker/internal image 형태를 우선한다.
- public wheel에 KenLM을 조용히 vendoring/static link하지 않는다. 필요하면 별도 license/relinking review를 거친다.
- KenLM `.binary` LM 파일은 model/domain artifact이며 CTranslate2 package에 포함하지 않는다.

## 8. Current Release Readiness

내부/실험 릴리즈는 가능하다.

확인된 것:

- `WITH_KENLM=ON/OFF` build 확인
- LM-fusion-targeted C++ tests 통과
- KenLM scorer cache 구현 및 same-path reuse test 추가
- alpha sweep benchmark와 환경 문서화
- license/packaging 정책 문서화

남은 것:

- 최종 wheel/install artifact 대상 Python smoke 실행
- full C++ suite의 기존 CPU GEMM failure 별도 triage 또는 release scope 명시
- 더 넓은 성능 claim이 필요하면 `asr_topk=20/100`, p50/p95 latency 추가 측정
