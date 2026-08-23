"""Retry/error-contract tests for SubtitleTranslator.

Locks in two fixes:
  * transport errors are classified via httpx.TransportError — a ReadError
    (Ollama dying mid-response) used to escape as a raw httpx exception with
    no retry, because the old check matched only "Timeout"/"Connect" in the
    class NAME;
  * a failure record is dropped once the individual retry fixes the segment,
    so translation_failures.json no longer reports failures that were healed.

Fake client only — no live Ollama needed.
"""

import json

import httpx
import pytest
from ollama import ResponseError

from subtitle_forge.core.translator import SubtitleTranslator, TranslationConfig
from subtitle_forge.exceptions import TranslationError
from subtitle_forge.models.subtitle import SubtitleSegment


class _FakeClient:
    """Yields the scripted outcomes in order: Exception -> raise, str -> reply."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return {"message": {"content": outcome}}


def _translator(outcomes, **cfg_kwargs):
    translator = SubtitleTranslator(TranslationConfig(**cfg_kwargs))
    fake = _FakeClient(outcomes)
    translator._client = fake
    return translator, fake


def _segs(*texts):
    return [
        SubtitleSegment(index=i + 1, start=float(i), end=float(i) + 0.9, text=t)
        for i, t in enumerate(texts)
    ]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("subtitle_forge.core.translator.time.sleep", lambda s: None)


def _good(indices_to_text):
    return json.dumps({"translations": indices_to_text})


def test_read_error_is_retried():
    translator, fake = _translator(
        [httpx.ReadError("connection reset"), _good({"1": "你好"})]
    )
    out = translator.translate_batch(_segs("Hello"), "en", "zh")
    assert [s.text for s in out] == ["你好"]
    assert fake.calls == 2


def test_read_error_exhausting_retries_raises_translation_error():
    translator, fake = _translator(
        [httpx.ReadError("x")] * 3, max_retries=3
    )
    with pytest.raises(TranslationError):
        translator.translate_batch(_segs("Hello"), "en", "zh")
    assert fake.calls == 3


def test_client_error_is_not_retried():
    translator, fake = _translator([ResponseError("model not found", 404)])
    with pytest.raises(TranslationError):
        translator.translate_batch(_segs("Hello"), "en", "zh")
    assert fake.calls == 1


def test_server_error_is_retried():
    translator, fake = _translator(
        [ResponseError("boom", 500), _good({"1": "你好"})]
    )
    out = translator.translate_batch(_segs("Hello"), "en", "zh")
    assert [s.text for s in out] == ["你好"]
    assert fake.calls == 2


def test_unrelated_exception_propagates_unwrapped():
    translator, fake = _translator([ValueError("programming error")])
    with pytest.raises(ValueError):
        translator.translate_batch(_segs("Hello"), "en", "zh")
    assert fake.calls == 1


def test_healed_segment_leaves_no_stale_failure_record():
    # Batch reply drops index 2; the individual retry then succeeds. The
    # failure record made during parsing must go with it.
    translator, fake = _translator(
        [_good({"1": "你好"}), "再见"]
    )
    out = translator.translate_batch(_segs("Hello", "Bye"), "en", "zh")
    assert [s.text for s in out] == ["你好", "再见"]
    assert fake.calls == 2
    assert translator.get_failed_translations() == []


def test_unhealed_segment_keeps_its_failure_record():
    # Batch reply drops index 2 and the individual retry echoes the original
    # (a failed retry) — that record must survive.
    translator, fake = _translator(
        [_good({"1": "你好"}), "Bye"]
    )
    out = translator.translate_batch(_segs("Hello", "Bye"), "en", "zh")
    assert [s.text for s in out] == ["你好", "Bye"]
    assert [f["index"] for f in translator.get_failed_translations()] == [2]
