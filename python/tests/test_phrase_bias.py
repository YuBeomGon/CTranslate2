"""
python/tests/test_phrase_bias.py
Tests for Whisper phrase bias Python bindings.
"""

import os

import numpy as np
import pytest
import test_utils

import ctranslate2

_DEFAULT_PHRASE_BIAS_MAX_TOKEN_DELTA = 2.0


def test_phrase_bias_path_defaults():
    p = ctranslate2.models.PhraseBiasPath(ids=[10, 20, 30], step_bias=0.25)
    assert list(p.ids) == [10, 20, 30]
    assert abs(p.step_bias - 0.25) < 1e-6
    assert p.min_prefix_len == 1


def test_phrase_bias_path_accepts_negative_step_bias():
    p = ctranslate2.models.PhraseBiasPath(ids=[10, 20, 30], step_bias=-0.25)
    assert list(p.ids) == [10, 20, 30]
    assert abs(p.step_bias + 0.25) < 1e-6
    assert p.min_prefix_len == 1


def test_phrase_bias_path_min_prefix_len():
    p = ctranslate2.models.PhraseBiasPath(ids=[1, 2], step_bias=0.5, min_prefix_len=2)
    assert p.min_prefix_len == 2


def test_phrase_bias_wraps_paths():
    p1 = ctranslate2.models.PhraseBiasPath(ids=[100, 200, 300], step_bias=0.25)
    p2 = ctranslate2.models.PhraseBiasPath(ids=[101, 200, 300], step_bias=0.25)
    b = ctranslate2.models.PhraseBias(token_paths=[p1, p2])
    assert len(b.token_paths) == 2
    assert list(b.token_paths[0].ids) == [100, 200, 300]


def test_whisper_ctor_and_generate_expose_phrase_biases():
    assert "phrase_biases" in (ctranslate2.models.Whisper.__init__.__doc__ or "")
    assert "phrase_biases" in (ctranslate2.models.Whisper.generate.__doc__ or "")


def _setup_whisper_tiny(tmp_dir):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")

    model_name = "openai/whisper-tiny"
    converter = ctranslate2.converters.TransformersConverter(model_name)
    output_dir = converter.convert(str(tmp_dir.join("ct2_whisper_tiny")))

    processor = transformers.WhisperProcessor.from_pretrained(model_name)
    audio_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "tests",
        "data",
        "audio",
        "jfk.npy",
    )
    if not os.path.isfile(audio_path):
        pytest.skip("Whisper audio fixture is not available")
    audio = np.load(audio_path)
    features = processor(audio, padding=False, sampling_rate=16000).input_features[0]
    features = np.pad(features, [(0, 0), (0, 3000 - features.shape[-1])])
    features = np.expand_dims(features.astype(np.float32), 0)

    prompts = [
        ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    ]
    special_token_begin = processor.tokenizer.eos_token_id
    return (
        output_dir,
        ctranslate2.StorageView.from_array(features),
        prompts,
        special_token_begin,
    )


def _generate(model, features, prompts, **kwargs):
    return model.generate(
        features,
        prompts,
        beam_size=1,
        suppress_tokens=[],
        **kwargs,
    )[0]


def _select_flippable_target(result, special_token_begin):
    base = result.sequences_ids[0]
    assert len(base) >= 3
    assert result.logits and len(result.logits[0]) >= 3

    for token_index in range(2, min(len(base), len(result.logits[0]))):
        step_logits = np.array(result.logits[0][token_index])
        base_token = base[token_index]
        base_logit = step_logits[base_token]
        ranked = np.argsort(step_logits)[::-1]

        for token in ranked:
            token = int(token)
            if token == base_token or token >= special_token_begin:
                continue
            if not np.isfinite(step_logits[token]):
                continue
            if base_logit - step_logits[token] < _DEFAULT_PHRASE_BIAS_MAX_TOKEN_DELTA - 0.05:
                return base[token_index - 2:token_index], token_index, base_token, token

    pytest.skip("No generated step close enough for a clamped phrase bias to flip")


def _bias_on(prefix_tokens, target):
    path = ctranslate2.models.PhraseBiasPath(
        ids=list(prefix_tokens) + [target],
        step_bias=50.0,
        min_prefix_len=len(prefix_tokens),
    )
    return [ctranslate2.models.PhraseBias(token_paths=[path])]


@test_utils.only_on_linux
def test_percall_phrase_biases_noop_and_effect(tmp_dir):
    output_dir, features, prompts, special_token_begin = _setup_whisper_tiny(tmp_dir)
    model = ctranslate2.models.Whisper(output_dir, device="cpu")

    base_result = _generate(model, features, prompts, return_logits_vocab=True)
    base = base_result.sequences_ids[0]
    prefix_tokens, target_index, base_token, target = _select_flippable_target(
        base_result, special_token_begin
    )

    assert _generate(model, features, prompts, phrase_biases=None).sequences_ids[0] == base
    assert _generate(model, features, prompts, phrase_biases=[]).sequences_ids[0] == base

    biased = _generate(
        model,
        features,
        prompts,
        phrase_biases=_bias_on(prefix_tokens, target),
    ).sequences_ids[0]
    assert biased[:target_index] == base[:target_index]
    assert biased[target_index] == target
    assert biased[target_index] != base_token
    assert biased != base


@test_utils.only_on_linux
def test_constructor_phrase_biases_persistent(tmp_dir):
    output_dir, features, prompts, special_token_begin = _setup_whisper_tiny(tmp_dir)

    baseline_model = ctranslate2.models.Whisper(output_dir, device="cpu")
    base_result = _generate(baseline_model, features, prompts, return_logits_vocab=True)
    base = base_result.sequences_ids[0]
    prefix_tokens, target_index, base_token, target = _select_flippable_target(
        base_result, special_token_begin
    )

    model = ctranslate2.models.Whisper(
        output_dir,
        device="cpu",
        phrase_biases=_bias_on(prefix_tokens, target),
    )
    out1 = _generate(model, features, prompts).sequences_ids[0]
    out2 = _generate(model, features, prompts).sequences_ids[0]
    assert out1[:target_index] == base[:target_index]
    assert out2[:target_index] == base[:target_index]
    assert out1[target_index] == target and out2[target_index] == target
    assert out1 == out2

    none_out = _generate(model, features, prompts, phrase_biases=None).sequences_ids[0]
    assert none_out[target_index] == target

    disabled = _generate(model, features, prompts, phrase_biases=[]).sequences_ids[0]
    assert disabled == base
    assert disabled[target_index] == base_token
