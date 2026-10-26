"""Prompt-length bucketing for the FLUX runtime.

transformers>=5 returns a ``BatchEncoding`` that subclasses ``UserDict`` rather
than ``dict``. The length probe must still read ``input_ids`` from it, otherwise
every prompt is treated as short and gets truncated before the scene description
is reached.
"""

from __future__ import annotations

from collections import UserDict

from core.backends.flux_worker import _prompt_sequence_length


class _BatchEncodingLike(UserDict):
    """Mirrors transformers>=5 ``BatchEncoding`` (a ``UserDict``, not ``dict``)."""


class _FakeTokenizer:
    def __init__(self, token_count: int) -> None:
        self._token_count = token_count

    def apply_chat_template(self, messages, **_kwargs) -> str:
        return messages[0]["content"]

    def __call__(self, _text, **_kwargs) -> _BatchEncodingLike:
        return _BatchEncodingLike(input_ids=list(range(self._token_count)))


class _Pipe:
    def __init__(self, token_count: int) -> None:
        self.tokenizer = _FakeTokenizer(token_count)


def test_long_persona_prompt_is_not_truncated(monkeypatch):
    monkeypatch.delenv("WEBBDUCK_FLUX_MAX_SEQUENCE_LENGTH", raising=False)
    bucket, counted = _prompt_sequence_length(_Pipe(385), "long persona prompt")
    assert counted == 385
    assert bucket == 512


def test_short_prompt_uses_compact_bucket(monkeypatch):
    monkeypatch.delenv("WEBBDUCK_FLUX_MAX_SEQUENCE_LENGTH", raising=False)
    bucket, counted = _prompt_sequence_length(_Pipe(14), "a woman")
    assert counted == 14
    assert bucket == 64


def test_missing_tokenizer_falls_back_to_full_context(monkeypatch):
    monkeypatch.delenv("WEBBDUCK_FLUX_MAX_SEQUENCE_LENGTH", raising=False)
    bucket, counted = _prompt_sequence_length(object(), "a woman")
    assert bucket == 512
    assert counted == 512


def test_unmeasurable_prompt_falls_back_to_full_context(monkeypatch):
    monkeypatch.delenv("WEBBDUCK_FLUX_MAX_SEQUENCE_LENGTH", raising=False)

    class _Broken:
        def apply_chat_template(self, *_args, **_kwargs):
            raise RuntimeError("no chat template")

    class _PipeWithBrokenTokenizer:
        tokenizer = _Broken()

    bucket, _counted = _prompt_sequence_length(_PipeWithBrokenTokenizer(), "a woman")
    assert bucket == 512
