"""
python/tests/test_phrase_bias.py
Tests for Whisper phrase bias Python bindings.
"""

import ctranslate2


def test_phrase_bias_path_defaults():
    p = ctranslate2.models.PhraseBiasPath(ids=[10, 20, 30], step_bias=0.25)
    assert list(p.ids) == [10, 20, 30]
    assert abs(p.step_bias - 0.25) < 1e-6
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
