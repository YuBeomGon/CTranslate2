# Phase 4 Implementation Plan — faster-whisper Phrase Bias Config + Tokenizer Compile

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a faster-whisper user-facing JSON/dict phrase-bias config that compiles keyword
strings to CT2 `PhraseBias` token paths at model initialization and passes them to the P3
`ctranslate2.models.Whisper(..., phrase_biases=...)` persistent-trie API.

**Architecture:** CT2 remains the numeric ids+bias executor. faster-whisper owns tokenizer
correctness: load JSON/dict config, compile each surface form into leading-space and
non-leading-space token paths, validate roundtrip, distribute total bias into continuation
step bias, convert to CT2 Python DTOs, and pass them during `WhisperModel.__init__`. Runtime
`transcribe()` does not retokenize or rebuild phrase bias.

**Tech Stack:** Python, faster-whisper, tokenizers, CTranslate2 P3 Python API, pytest, JSON.

---

## Repositories

| Repo | Path | Role |
|------|------|------|
| CTranslate2 | `/data/MyProject/stt/CTranslate2` | P3 API is already implemented. P4 only updates docs/status here. |
| faster-whisper | `/data/MyProject/stt/faster-whisper` | P4 implementation lives here. |

> Do not use `dev-docs/summary/` as source material. The P4 source of truth is
> `dev-docs/SSOT.md` plus this plan.

---

## Public API

### Python init argument

Add an optional init argument to `faster_whisper.transcribe.WhisperModel`:

```python
model = WhisperModel(
    "large-v3",
    phrase_bias_config="configs/domain_terms.json",
)
```

Also accept a dict for tests and wrapper applications:

```python
model = WhisperModel(
    "large-v3",
    phrase_bias_config={
        "version": 1,
        "enabled": True,
        "default_total_bias": 0.5,
        "bias_schedule": "uniform",
        "terms": [
            {"text": "트랜스포머", "bias": 0.5},
            {"text": "CTranslate2", "bias": 0.7, "aliases": ["씨트랜슬레이트투"]},
        ],
    },
)
```

### JSON schema

```json
{
  "version": 1,
  "enabled": true,
  "default_total_bias": 0.5,
  "bias_schedule": "uniform",
  "min_total_bias": 0.1,
  "max_total_bias": 1.5,
  "max_step_bias": 0.5,
  "min_prefix_len": 1,
  "terms": [
    {
      "text": "트랜스포머",
      "bias": 0.5,
      "aliases": ["Transformer"],
      "schedule": "ramp",
      "min_prefix_len": 1
    }
  ]
}
```

Rules:
- `version` must be `1`.
- `enabled=false` means no phrase bias is compiled or passed to CT2.
- `terms[].text` is required and must be non-empty after stripping.
- `terms[].bias` overrides `default_total_bias`.
- `terms[].aliases` are additional surfaces for the same domain term.
- `bias_schedule` and per-term `schedule` support:
  - `uniform`: equal continuation bias, matching the current SSOT default.
  - `ramp`: later continuation steps receive larger bias.
- Unknown top-level or term-level keys should raise `ValueError` to catch config typos.

---

## Bias Schedule Semantics

For a token path `ids=[A, B, C, D]`, there are 3 continuation steps:

```text
suffix [A]     -> B
suffix [A, B]  -> C
suffix [A,B,C] -> D
```

### Uniform

`total_bias=0.6`, path length 4:

```text
step deltas = [0.2, 0.2, 0.2]
```

This compiles to one CT2 path:

```python
PhraseBiasPath(ids=[A, B, C, D], step_bias=0.2, min_prefix_len=1)
```

### Ramp

`total_bias=0.6`, path length 4, linear ramp weights `[1, 2, 3]`:

```text
step deltas = [0.1, 0.2, 0.3]
```

CT2 P3 supports one `step_bias` per path, but P3 also supports overlap summation and
`min_prefix_len`. Represent ramp as cumulative increments:

```python
PhraseBiasPath(ids=[A, B, C, D], step_bias=0.1, min_prefix_len=1)
PhraseBiasPath(ids=[A, B, C, D], step_bias=0.1, min_prefix_len=2)
PhraseBiasPath(ids=[A, B, C, D], step_bias=0.1, min_prefix_len=3)
```

At each step CT2 sums matching entries:

```text
prefix len 1 -> 0.1
prefix len 2 -> 0.1 + 0.1 = 0.2
prefix len 3 -> 0.1 + 0.1 + 0.1 = 0.3
```

This lets faster-whisper test ramp behavior without changing CT2.

Clamp order in P4:

```text
1. Clamp total_bias to [min_total_bias, max_total_bias].
2. Compute schedule step deltas.
3. Clamp each scheduled step delta to max_step_bias.
4. Convert to one or more CT2 PhraseBiasPath entries.
5. CT2 P3 still applies its final per-token max delta clamp.
```

---

## File Structure

### faster-whisper repo

| File | Responsibility |
|------|----------------|
| `faster_whisper/phrase_bias.py` | New config loader, tokenizer compiler, schedule expansion, CT2 DTO conversion. |
| `faster_whisper/transcribe.py` | Add `phrase_bias_config` init arg, load tokenizer before CT2 model construction, pass compiled CT2 phrase biases to `ctranslate2.models.Whisper`. |
| `tests/test_phrase_bias.py` | New unit tests for config parsing, tokenizer correctness, uniform/ramp schedules, CT2 DTO conversion, and init handoff. |
| `tests/test_transcribe.py` | Add smoke test that `WhisperModel(..., phrase_bias_config=...)` works with existing audio fixture, if stable. |
| `README.md` | Document JSON config and init argument. |
| `benchmark/phrase_bias_ab.py` | Add A/B evaluation runner for recall, precision, insertion, CER/WER, and latency on a manifest. |

### CTranslate2 repo

| File | Responsibility |
|------|----------------|
| `dev-docs/p4-impl-plan.md` | This plan. |
| `dev-docs/STATUS.md` | Mark P4 plan present now; mark P4 tasks complete after implementation. |
| `dev-docs/p4-completion-report.md` | Create after implementation with commits, verification, and drift notes. |

---

## Task 1: faster-whisper phrase-bias compiler unit tests

**Files:**
- Create: `/data/MyProject/stt/faster-whisper/tests/test_phrase_bias.py`

- [ ] **Step 1: Write failing tests for config parsing and uniform compile**

Create `tests/test_phrase_bias.py`:

```python
"""
tests/test_phrase_bias.py
Tests for faster-whisper phrase bias config compilation.
"""

import json

import pytest

from faster_whisper.phrase_bias import (
    CompiledPhraseBiasPath,
    compile_phrase_bias_config,
    load_phrase_bias_config,
)


class FakeWhisperTokenizer:
    def __init__(self):
        self._eot = 50000
        self._encoded = {
            " 트랜스포머": [10, 11, 12],
            "트랜스포머": [20, 11, 12],
            " Transformer": [30, 31, 32, 33],
            "Transformer": [40, 31, 32, 33],
            " 짧음": [50],
            "짧음": [51],
            " special": [60, 62, 50001],
            "special": [61, 62, 50001],
        }
        self._decoded = {
            (10, 11, 12): " 트랜스포머",
            (20, 11, 12): "트랜스포머",
            (30, 31, 32, 33): " Transformer",
            (40, 31, 32, 33): "Transformer",
            (50,): " 짧음",
            (51,): "짧음",
            (60, 62): " special",
            (61, 62): "special",
        }

    def token_to_id(self, token):
        if token == "<|endoftext|>":
            return self._eot
        raise KeyError(token)

    def encode(self, text, add_special_tokens=False):
        class Encoded:
            def __init__(self, ids):
                self.ids = ids

        return Encoded(list(self._encoded[text]))

    def decode(self, ids):
        return self._decoded[tuple(ids)]


def test_load_phrase_bias_config_from_path(tmp_path):
    path = tmp_path / "phrase_bias.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "enabled": True,
                "default_total_bias": 0.5,
                "terms": [{"text": "트랜스포머"}],
            }
        ),
        encoding="utf-8",
    )

    config = load_phrase_bias_config(str(path))

    assert config["version"] == 1
    assert config["terms"] == [{"text": "트랜스포머"}]


def test_compile_uniform_two_surface_paths():
    compiled = compile_phrase_bias_config(
        {
            "version": 1,
            "enabled": True,
            "default_total_bias": 0.6,
            "bias_schedule": "uniform",
            "terms": [{"text": "트랜스포머"}],
        },
        FakeWhisperTokenizer(),
    )

    assert len(compiled) == 1
    assert compiled[0].surface == "트랜스포머"
    assert compiled[0].token_paths == [
        CompiledPhraseBiasPath(ids=[10, 11, 12], step_bias=0.3, min_prefix_len=1),
        CompiledPhraseBiasPath(ids=[20, 11, 12], step_bias=0.3, min_prefix_len=1),
    ]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cd /data/MyProject/stt/faster-whisper
python -m pytest tests/test_phrase_bias.py -q
```

Expected: import failure for `faster_whisper.phrase_bias`.

- [ ] **Step 3: Commit RED tests**

```bash
cd /data/MyProject/stt/faster-whisper
git add tests/test_phrase_bias.py
git commit -m "test: add phrase bias compiler tests"
```

---

## Task 2: Implement `faster_whisper/phrase_bias.py`

**Files:**
- Create: `/data/MyProject/stt/faster-whisper/faster_whisper/phrase_bias.py`
- Modify: `/data/MyProject/stt/faster-whisper/tests/test_phrase_bias.py`

- [ ] **Step 1: Implement compiler dataclasses and JSON loader**

Create `faster_whisper/phrase_bias.py`:

```python
"""
faster_whisper/phrase_bias.py
Compile user-facing phrase bias config into CTranslate2 Whisper phrase bias DTOs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import ctranslate2


@dataclass(frozen=True)
class CompiledPhraseBiasPath:
    ids: List[int]
    step_bias: float
    min_prefix_len: int = 1


@dataclass(frozen=True)
class CompiledPhraseBias:
    surface: str
    token_paths: List[CompiledPhraseBiasPath]


ConfigInput = Optional[Union[str, Mapping[str, Any]]]


_TOP_LEVEL_KEYS = {
    "version",
    "enabled",
    "default_total_bias",
    "bias_schedule",
    "min_total_bias",
    "max_total_bias",
    "max_step_bias",
    "min_prefix_len",
    "terms",
}
_TERM_KEYS = {"text", "bias", "aliases", "schedule", "min_prefix_len"}


def load_phrase_bias_config(config: ConfigInput) -> Optional[Dict[str, Any]]:
    if config is None:
        return None
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as file:
            loaded = json.load(file)
    elif isinstance(config, Mapping):
        loaded = dict(config)
    else:
        raise TypeError("phrase_bias_config must be a path, a mapping, or None")

    if not isinstance(loaded, dict):
        raise ValueError("phrase_bias_config must be a JSON object")
    unknown = set(loaded) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError("Unknown phrase_bias_config keys: %s" % sorted(unknown))
    if loaded.get("version") != 1:
        raise ValueError("phrase_bias_config.version must be 1")
    if "terms" not in loaded or not isinstance(loaded["terms"], list):
        raise ValueError("phrase_bias_config.terms must be a list")
    return loaded
```

- [ ] **Step 2: Implement compile helpers**

Append to `faster_whisper/phrase_bias.py`:

```python
def compile_phrase_bias_config(
    config: ConfigInput,
    tokenizer,
) -> List[CompiledPhraseBias]:
    loaded = load_phrase_bias_config(config)
    if not loaded or loaded.get("enabled", True) is False:
        return []

    min_total_bias = float(loaded.get("min_total_bias", 0.1))
    max_total_bias = float(loaded.get("max_total_bias", 1.5))
    max_step_bias = float(loaded.get("max_step_bias", 0.5))
    default_bias = float(loaded.get("default_total_bias", 0.5))
    default_schedule = loaded.get("bias_schedule", "uniform")
    default_min_prefix_len = int(loaded.get("min_prefix_len", 1))

    compiled: List[CompiledPhraseBias] = []
    seen_surfaces = set()
    for term in loaded["terms"]:
        if not isinstance(term, dict):
            raise ValueError("Each phrase bias term must be an object")
        unknown = set(term) - _TERM_KEYS
        if unknown:
            raise ValueError("Unknown phrase bias term keys: %s" % sorted(unknown))

        text = str(term.get("text", "")).strip()
        if not text:
            raise ValueError("Phrase bias term text must be non-empty")

        surfaces = [text]
        for alias in term.get("aliases", []) or []:
            alias_text = str(alias).strip()
            if alias_text:
                surfaces.append(alias_text)

        total_bias = _clamp(
            float(term.get("bias", default_bias)),
            min_total_bias,
            max_total_bias,
        )
        schedule = term.get("schedule", default_schedule)
        min_prefix_len = int(term.get("min_prefix_len", default_min_prefix_len))

        for surface in surfaces:
            if surface in seen_surfaces:
                continue
            seen_surfaces.add(surface)
            token_paths = _compile_surface(
                surface,
                tokenizer,
                total_bias,
                schedule,
                max_step_bias,
                min_prefix_len,
            )
            if token_paths:
                compiled.append(CompiledPhraseBias(surface=surface, token_paths=token_paths))
    return compiled


def _compile_surface(
    surface: str,
    tokenizer,
    total_bias: float,
    schedule: str,
    max_step_bias: float,
    min_prefix_len: int,
) -> List[CompiledPhraseBiasPath]:
    paths: List[CompiledPhraseBiasPath] = []
    seen_paths = set()
    for variant in (" " + surface, surface):
        ids = _encode_clean_ids(tokenizer, variant)
        if len(ids) < 2:
            continue
        if tokenizer.decode(ids) != variant:
            continue
        key = tuple(ids)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        paths.extend(_expand_schedule(ids, total_bias, schedule, max_step_bias, min_prefix_len))
    return paths


def _encode_clean_ids(tokenizer, surface: str) -> List[int]:
    ids = list(tokenizer.encode(surface, add_special_tokens=False).ids)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    return [token_id for token_id in ids if token_id < eot_id]


def _expand_schedule(
    ids: Sequence[int],
    total_bias: float,
    schedule: str,
    max_step_bias: float,
    min_prefix_len: int,
) -> List[CompiledPhraseBiasPath]:
    continuation_count = len(ids) - 1
    if continuation_count <= 0:
        return []

    if schedule == "uniform":
        step_bias = min(total_bias / continuation_count, max_step_bias)
        return [
            CompiledPhraseBiasPath(
                ids=list(ids),
                step_bias=step_bias,
                min_prefix_len=min_prefix_len,
            )
        ]

    if schedule == "ramp":
        weight_sum = continuation_count * (continuation_count + 1) / 2
        deltas = [
            min(total_bias * (i + 1) / weight_sum, max_step_bias)
            for i in range(continuation_count)
        ]
        paths = []
        previous = 0.0
        for index, delta in enumerate(deltas, start=1):
            increment = delta - previous
            previous = delta
            if increment <= 0:
                continue
            paths.append(
                CompiledPhraseBiasPath(
                    ids=list(ids),
                    step_bias=increment,
                    min_prefix_len=max(min_prefix_len, index),
                )
            )
        return paths

    raise ValueError("Unsupported phrase bias schedule: %s" % schedule)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))
```

- [ ] **Step 3: Implement CT2 DTO conversion**

Append to `faster_whisper/phrase_bias.py`:

```python
def to_ctranslate2_phrase_biases(
    compiled: Iterable[CompiledPhraseBias],
) -> List[ctranslate2.models.PhraseBias]:
    phrase_biases = []
    for bias in compiled:
        token_paths = [
            ctranslate2.models.PhraseBiasPath(
                ids=path.ids,
                step_bias=path.step_bias,
                min_prefix_len=path.min_prefix_len,
            )
            for path in bias.token_paths
        ]
        if token_paths:
            phrase_biases.append(ctranslate2.models.PhraseBias(token_paths=token_paths))
    return phrase_biases
```

- [ ] **Step 4: Add ramp, skip, validation, and CT2 DTO tests**

Append to `tests/test_phrase_bias.py`:

```python
from faster_whisper.phrase_bias import to_ctranslate2_phrase_biases


def test_compile_ramp_schedule_as_cumulative_min_prefix_paths():
    compiled = compile_phrase_bias_config(
        {
            "version": 1,
            "enabled": True,
            "bias_schedule": "ramp",
            "terms": [{"text": "Transformer", "bias": 0.6}],
        },
        FakeWhisperTokenizer(),
    )

    paths = compiled[0].token_paths
    assert paths[:3] == [
        CompiledPhraseBiasPath(ids=[30, 31, 32, 33], step_bias=0.1, min_prefix_len=1),
        CompiledPhraseBiasPath(ids=[30, 31, 32, 33], step_bias=0.1, min_prefix_len=2),
        CompiledPhraseBiasPath(ids=[30, 31, 32, 33], step_bias=0.1, min_prefix_len=3),
    ]


def test_compile_skips_one_token_surfaces():
    compiled = compile_phrase_bias_config(
        {"version": 1, "enabled": True, "terms": [{"text": "짧음", "bias": 0.5}]},
        FakeWhisperTokenizer(),
    )

    assert compiled == []


def test_compile_strips_special_tokens_and_keeps_roundtrip():
    compiled = compile_phrase_bias_config(
        {"version": 1, "enabled": True, "terms": [{"text": "special", "bias": 0.5}]},
        FakeWhisperTokenizer(),
    )

    assert compiled[0].token_paths == [
        CompiledPhraseBiasPath(ids=[60, 62], step_bias=0.5, min_prefix_len=1),
        CompiledPhraseBiasPath(ids=[61, 62], step_bias=0.5, min_prefix_len=1),
    ]


def test_enabled_false_compiles_empty():
    compiled = compile_phrase_bias_config(
        {"version": 1, "enabled": False, "terms": [{"text": "트랜스포머"}]},
        FakeWhisperTokenizer(),
    )

    assert compiled == []


def test_unknown_config_key_raises():
    with pytest.raises(ValueError, match="Unknown phrase_bias_config keys"):
        compile_phrase_bias_config(
            {"version": 1, "terms": [], "typo": True},
            FakeWhisperTokenizer(),
        )


def test_to_ctranslate2_phrase_biases_roundtrip():
    compiled = compile_phrase_bias_config(
        {"version": 1, "enabled": True, "terms": [{"text": "트랜스포머", "bias": 0.6}]},
        FakeWhisperTokenizer(),
    )

    ct2_biases = to_ctranslate2_phrase_biases(compiled)

    assert len(ct2_biases) == 1
    assert [list(path.ids) for path in ct2_biases[0].token_paths] == [
        [10, 11, 12],
        [20, 11, 12],
    ]
```

- [ ] **Step 5: Run phrase-bias tests**

Run:

```bash
cd /data/MyProject/stt/faster-whisper
python -m pytest tests/test_phrase_bias.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit compiler**

```bash
cd /data/MyProject/stt/faster-whisper
git add faster_whisper/phrase_bias.py tests/test_phrase_bias.py
git commit -m "feat: compile phrase bias config"
```

---

## Task 3: Wire compiled phrase bias into `WhisperModel.__init__`

**Files:**
- Modify: `/data/MyProject/stt/faster-whisper/faster_whisper/transcribe.py`
- Modify: `/data/MyProject/stt/faster-whisper/tests/test_phrase_bias.py`

- [ ] **Step 1: Add RED init handoff test**

Append to `tests/test_phrase_bias.py`:

```python
def test_whisper_model_init_passes_compiled_phrase_biases(monkeypatch, tmp_path):
    from faster_whisper import transcribe
    from faster_whisper.transcribe import WhisperModel

    captured = {}

    class FakeWhisper:
        is_multilingual = False

        def __init__(self, model_path, **kwargs):
            captured["model_path"] = model_path
            captured["kwargs"] = kwargs

    monkeypatch.setattr(transcribe.ctranslate2.models, "Whisper", FakeWhisper)
    monkeypatch.setattr(transcribe, "_load_hf_tokenizer", lambda *args, **kwargs: FakeWhisperTokenizer())

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    WhisperModel(
        str(model_dir),
        device="cpu",
        phrase_bias_config={
            "version": 1,
            "enabled": True,
            "terms": [{"text": "트랜스포머", "bias": 0.6}],
        },
    )

    assert "phrase_biases" in captured["kwargs"]
    assert len(captured["kwargs"]["phrase_biases"]) == 1
    assert [list(path.ids) for path in captured["kwargs"]["phrase_biases"][0].token_paths] == [
        [10, 11, 12],
        [20, 11, 12],
    ]


def test_phrase_bias_config_conflicts_with_raw_ct2_phrase_biases(tmp_path):
    from faster_whisper.transcribe import WhisperModel

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="phrase_bias_config cannot be used with phrase_biases"):
        WhisperModel(
            str(model_dir),
            phrase_bias_config={"version": 1, "enabled": False, "terms": []},
            phrase_biases=[],
        )
```

- [ ] **Step 2: Run RED test**

Run:

```bash
cd /data/MyProject/stt/faster-whisper
python -m pytest tests/test_phrase_bias.py::test_whisper_model_init_passes_compiled_phrase_biases -q
```

Expected: `TypeError` because `phrase_bias_config` is not accepted yet, or assertion failure
because `phrase_biases` is not passed to CT2.

- [ ] **Step 3: Add imports and tokenizer helper in `transcribe.py`**

Modify imports near the top of `faster_whisper/transcribe.py`:

```python
from faster_whisper.phrase_bias import (
    compile_phrase_bias_config,
    to_ctranslate2_phrase_biases,
)
```

Add helper near `WhisperModel` or above it:

```python
def _fallback_tokenizer_name(model_size_or_path: str) -> str:
    name = str(model_size_or_path).lower()
    if name.endswith(".en") or name.endswith("-en"):
        return "openai/whisper-tiny.en"
    return "openai/whisper-tiny"


def _load_hf_tokenizer(model_path, tokenizer_bytes, model_size_or_path):
    tokenizer_file = os.path.join(model_path, "tokenizer.json")
    if tokenizer_bytes:
        return tokenizers.Tokenizer.from_buffer(tokenizer_bytes)
    if os.path.isfile(tokenizer_file):
        return tokenizers.Tokenizer.from_file(tokenizer_file)
    return tokenizers.Tokenizer.from_pretrained(_fallback_tokenizer_name(model_size_or_path))
```

- [ ] **Step 4: Add `phrase_bias_config` init parameter and reorder tokenizer loading**

Change `WhisperModel.__init__` signature:

```python
        use_auth_token: Optional[Union[str, bool]] = None,
        phrase_bias_config: Optional[Union[str, dict]] = None,
        **model_kwargs,
```

In the docstring, add:

```python
          phrase_bias_config: Optional JSON path or dictionary defining domain terms and
            logit bias. Terms are tokenized once during model initialization and passed
            to CTranslate2 as persistent phrase bias paths.
```

Replace the existing tokenizer/model construction block with:

```python
        self.hf_tokenizer = _load_hf_tokenizer(
            model_path,
            tokenizer_bytes,
            model_size_or_path,
        )

        if phrase_bias_config is not None:
            if "phrase_biases" in model_kwargs:
                raise ValueError("phrase_bias_config cannot be used with phrase_biases")
            compiled_phrase_biases = compile_phrase_bias_config(
                phrase_bias_config,
                self.hf_tokenizer,
            )
            ct2_phrase_biases = to_ctranslate2_phrase_biases(compiled_phrase_biases)
            if ct2_phrase_biases:
                model_kwargs["phrase_biases"] = ct2_phrase_biases
            self.phrase_bias_config = phrase_bias_config
            self.compiled_phrase_biases = compiled_phrase_biases
        else:
            self.phrase_bias_config = None
            self.compiled_phrase_biases = []

        self.model = ctranslate2.models.Whisper(
            model_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            intra_threads=cpu_threads,
            inter_threads=num_workers,
            files=files,
            **model_kwargs,
        )
```

Remove the old tokenizer loading block after `self.model = ...` because `self.hf_tokenizer`
is already set.

- [ ] **Step 5: Run init handoff tests**

Run:

```bash
cd /data/MyProject/stt/faster-whisper
python -m pytest tests/test_phrase_bias.py::test_whisper_model_init_passes_compiled_phrase_biases tests/test_phrase_bias.py::test_phrase_bias_config_conflicts_with_raw_ct2_phrase_biases -q
```

Expected: both pass.

- [ ] **Step 6: Run tokenizer and transcribe smoke**

Run:

```bash
cd /data/MyProject/stt/faster-whisper
python -m pytest tests/test_tokenizer.py -q
python -m pytest tests/test_transcribe.py::test_transcribe_signature -q
```

Expected: pass. `test_transcribe_signature` verifies batched pipeline signature parity if
the init signature change affects public API assumptions.

- [ ] **Step 7: Commit init integration**

```bash
cd /data/MyProject/stt/faster-whisper
git add faster_whisper/transcribe.py tests/test_phrase_bias.py
git commit -m "feat: pass phrase bias config to CTranslate2"
```

---

## Task 4: Actual tokenizer correctness smoke

**Files:**
- Modify: `/data/MyProject/stt/faster-whisper/tests/test_phrase_bias.py`

- [ ] **Step 1: Add actual Whisper tokenizer test**

Append:

```python
def test_actual_whisper_tokenizer_compiles_leading_and_non_leading_paths():
    from faster_whisper import WhisperModel

    model = WhisperModel("tiny")
    compiled = compile_phrase_bias_config(
        {
            "version": 1,
            "enabled": True,
            "terms": [{"text": "transformer", "bias": 0.5}],
        },
        model.hf_tokenizer,
    )

    assert len(compiled) == 1
    decoded_paths = [model.hf_tokenizer.decode(path.ids) for path in compiled[0].token_paths]
    assert " transformer" in decoded_paths
    assert "transformer" in decoded_paths
    assert all(len(path.ids) >= 2 for path in compiled[0].token_paths)
```

- [ ] **Step 2: Run actual tokenizer test**

Run:

```bash
cd /data/MyProject/stt/faster-whisper
python -m pytest tests/test_phrase_bias.py::test_actual_whisper_tokenizer_compiles_leading_and_non_leading_paths -q
```

Expected: pass. This may download/cache `tiny` model assets.

- [ ] **Step 3: Commit tokenizer smoke**

```bash
cd /data/MyProject/stt/faster-whisper
git add tests/test_phrase_bias.py
git commit -m "test: verify phrase bias with whisper tokenizer"
```

---

## Task 5: README documentation and example config

**Files:**
- Modify: `/data/MyProject/stt/faster-whisper/README.md`
- Create: `/data/MyProject/stt/faster-whisper/examples/phrase_bias_config.json`

- [ ] **Step 1: Add example config**

Create `examples/phrase_bias_config.json`:

```json
{
  "version": 1,
  "enabled": true,
  "default_total_bias": 0.5,
  "bias_schedule": "uniform",
  "terms": [
    {
      "text": "ComfyUI",
      "bias": 0.5,
      "aliases": ["Comfy UI"]
    },
    {
      "text": "트랜스포머",
      "bias": 0.6,
      "schedule": "ramp"
    }
  ]
}
```

- [ ] **Step 2: Document usage in README**

Add a section:

```markdown
### Phrase bias

`phrase_bias_config` applies a positive continuation bias to domain terms during
Whisper decoding. The config is compiled once when `WhisperModel` is initialized.
It requires a CTranslate2 build that supports `ctranslate2.models.PhraseBias`.

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "large-v3",
    phrase_bias_config="examples/phrase_bias_config.json",
)
```

The JSON file contains user-facing strings and total logit bias values. faster-whisper
tokenizes both `" term"` and `"term"` forms, validates exact tokenizer roundtrip, removes
Whisper special tokens, and passes token ids to CTranslate2. CT2 does not tokenize strings.

`bias_schedule` can be `uniform` or `ramp`. `uniform` splits the total bias evenly across
continuation tokens. `ramp` makes later continuation tokens stronger and should be validated
with A/B tests because insertion risk is domain dependent.
```

- [ ] **Step 3: Commit docs**

```bash
cd /data/MyProject/stt/faster-whisper
git add README.md examples/phrase_bias_config.json
git commit -m "docs: add phrase bias config usage"
```

---

## Task 6: A/B evaluation runner

**Files:**
- Create: `/data/MyProject/stt/faster-whisper/benchmark/phrase_bias_ab.py`

- [ ] **Step 1: Add A/B runner**

Create `benchmark/phrase_bias_ab.py`:

```python
"""
benchmark/phrase_bias_ab.py
Run phrase-bias A/B evaluation over an audio manifest.
"""

from __future__ import annotations

import argparse
import json
import time

from faster_whisper import WhisperModel


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def contains_any(text: str, terms):
    normalized = normalize(text)
    return any(normalize(term) in normalized for term in terms)


def transcribe_one(model, audio_path):
    start = time.perf_counter()
    segments, _ = model.transcribe(audio_path, beam_size=5, temperature=0.0)
    text = "".join(segment.text for segment in segments)
    latency = time.perf_counter() - start
    return text, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phrase-bias-config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="default")
    args = parser.parse_args()

    baseline = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    biased = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        phrase_bias_config=args.phrase_bias_config,
    )

    rows = []
    with open(args.manifest, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    total = 0
    recall_base = 0
    recall_bias = 0
    insertion_base = 0
    insertion_bias = 0
    latency_base = 0.0
    latency_bias = 0.0

    for row in rows:
        total += 1
        terms = row["terms"]
        expected_present = bool(row.get("expected_present", True))
        base_text, base_latency = transcribe_one(baseline, row["audio"])
        bias_text, bias_latency = transcribe_one(biased, row["audio"])
        base_has = contains_any(base_text, terms)
        bias_has = contains_any(bias_text, terms)

        if expected_present:
            recall_base += int(base_has)
            recall_bias += int(bias_has)
        else:
            insertion_base += int(base_has)
            insertion_bias += int(bias_has)

        latency_base += base_latency
        latency_bias += bias_latency
        print(
            json.dumps(
                {
                    "audio": row["audio"],
                    "terms": terms,
                    "baseline_has_term": base_has,
                    "biased_has_term": bias_has,
                    "baseline_text": base_text,
                    "biased_text": bias_text,
                    "baseline_latency": base_latency,
                    "biased_latency": bias_latency,
                },
                ensure_ascii=False,
            )
        )

    print(
        json.dumps(
            {
                "total": total,
                "recall_base": recall_base,
                "recall_bias": recall_bias,
                "insertion_base": insertion_base,
                "insertion_bias": insertion_bias,
                "avg_latency_base": latency_base / max(total, 1),
                "avg_latency_bias": latency_bias / max(total, 1),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add manifest example to README**

Document manifest lines:

```json
{"audio": "tests/data/hotwords.mp3", "terms": ["ComfyUI"], "expected_present": true}
{"audio": "tests/data/jfk.flac", "terms": ["ComfyUI"], "expected_present": false}
```

Run command:

```bash
python benchmark/phrase_bias_ab.py \
  --model tiny \
  --manifest phrase_bias_manifest.jsonl \
  --phrase-bias-config examples/phrase_bias_config.json \
  --device cpu
```

- [ ] **Step 3: Smoke run on existing hotwords fixture**

Create a temporary manifest outside git:

```bash
cd /data/MyProject/stt/faster-whisper
printf '{"audio":"tests/data/hotwords.mp3","terms":["ComfyUI"],"expected_present":true}\n' > /tmp/phrase_bias_manifest.jsonl
python benchmark/phrase_bias_ab.py --model tiny --manifest /tmp/phrase_bias_manifest.jsonl --phrase-bias-config examples/phrase_bias_config.json --device cpu
```

Expected: command exits 0 and prints JSONL rows plus summary. This is a smoke test, not
the final domain acceptance test.

- [ ] **Step 4: Commit A/B runner**

```bash
cd /data/MyProject/stt/faster-whisper
git add benchmark/phrase_bias_ab.py README.md
git commit -m "test: add phrase bias A/B runner"
```

---

## Task 7: Final verification and docs sync

**Files:**
- Modify: `/data/MyProject/stt/CTranslate2/dev-docs/STATUS.md`
- Create after implementation: `/data/MyProject/stt/CTranslate2/dev-docs/p4-completion-report.md`

- [ ] **Step 1: Run faster-whisper focused tests**

Run:

```bash
cd /data/MyProject/stt/faster-whisper
python -m pytest tests/test_phrase_bias.py -q
python -m pytest tests/test_tokenizer.py -q
python -m pytest tests/test_transcribe.py::test_transcribe_signature tests/test_transcribe.py::test_hotwords -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run CT2 P3 compatibility smoke**

Run from CTranslate2 repo:

```bash
cd /data/MyProject/stt/CTranslate2
LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6:/data/MyProject/stt/CTranslate2/install-cpu/lib/libctranslate2.so.4 \
LD_LIBRARY_PATH=/data/MyProject/stt/CTranslate2/install-cpu/lib:/data/MyProject/stt/CTranslate2/install-cpu/lib64:$LD_LIBRARY_PATH \
python -m pytest python/tests/test_phrase_bias.py -q
```

Expected: `6 passed`.

- [ ] **Step 3: Update CTranslate2 status**

Modify `dev-docs/STATUS.md` Phase 4:

```markdown
## Phase 4 — faster-whisper 연동 (tokenizer compile 포함)  ✅ 완료 · plan: [`p4-impl-plan.md`](p4-impl-plan.md)
...
- [x] 키워드→2 path compile **pure 함수** ...
- [x] tokenizer correctness 테스트 ...
- [x] init 시 compile → `phrase_biases`로 CT2 generate 전달
- [x] 실제 음성 A/B smoke ...
```

- [ ] **Step 4: Write P4 completion report**

Create `dev-docs/p4-completion-report.md` with:

```markdown
# P4 Completion Report — faster-whisper Phrase Bias

## 완료 요약

## faster-whisper 커밋

## CTranslate2 문서 커밋

## 검증 결과

## 계획 대비 조정 사항

## 남은 범위
```

Fill each section with actual commit SHAs and command outputs. Include any drift such as
tokenizer fallback behavior, A/B fixture limits, or config schema changes.

- [ ] **Step 5: Commit CTranslate2 docs**

```bash
cd /data/MyProject/stt/CTranslate2
git add dev-docs/STATUS.md dev-docs/p4-completion-report.md
git commit -m "docs(p4): mark faster-whisper integration complete"
```

---

## Acceptance Criteria

- faster-whisper accepts `phrase_bias_config` as JSON path or dict in `WhisperModel.__init__`.
- `phrase_bias_config` is compiled once at init and passed to CT2 `Whisper(..., phrase_biases=...)`.
- No keyword tokenization happens per segment or per `generate()`.
- CT2 still receives only token ids + step bias, never strings.
- Both leading-space and non-leading-space variants are compiled when valid.
- Special Whisper tokens are removed, while leading-space text is preserved.
- `decode(ids) == surface` roundtrip is enforced for each path.
- 1-token paths are skipped.
- `uniform` and `ramp` schedules are tested.
- `enabled=false` is a no-op.
- Existing hotwords prompt behavior remains unchanged.
- Focused tests and A/B smoke pass.

---

## Existing Test Coverage Before P4

Before this plan, P4 testing was documented only at a high level:

- `dev-docs/SSOT.md` defines tokenizer correctness and real A/B requirements.
- `dev-docs/STATUS.md` lists P4 checklist items.
- `dev-docs/archive/superseded-2026-06-02/testing-manual.md` has an older A/B manual.

There was no current P4 implementation plan with concrete faster-whisper files, test names,
JSON schema, or commands. This document fills that gap.
