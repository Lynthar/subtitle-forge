"""Retry/error-contract tests for SubtitleTranslator.

Locks in two fixes:
  * transport errors are classified via httpx.TransportError — a ReadError
    (a server dying mid-response) used to escape as a raw httpx exception with
    no retry, because the old check matched only "Timeout"/"Connect" in the
    class NAME;
  * a failure record is dropped once the individual retry fixes the segment,
    so translation_failures.json no longer reports failures that were healed.

Fake chat backend only — no live server needed.
"""

import json
import logging

import httpx
import pytest

from subtitle_forge.core.service import ServiceError
from subtitle_forge.core.translator import SubtitleTranslator
from subtitle_forge.models.config import TranslationConfig
from subtitle_forge.exceptions import TranslationError
from subtitle_forge.models.subtitle import SubtitleSegment


class _FakeChat:
    """Yields the scripted outcomes in order: Exception -> raise, str -> reply."""

    model = "qwen2.5:14b"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def chat(self, prompt, *, json_keys=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _translator(outcomes, **cfg_kwargs):
    fake = _FakeChat(outcomes)
    return SubtitleTranslator(TranslationConfig(**cfg_kwargs), fake), fake


def _segs(*texts):
    return [
        SubtitleSegment(index=i + 1, start=float(i), end=float(i) + 0.9, text=t)
        for i, t in enumerate(texts)
    ]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("subtitle_forge.core.service.time.sleep", lambda s: None)


def _good(indices_to_text):
    return json.dumps({"translations": indices_to_text})


def test_read_error_is_retried():
    translator, fake = _translator([httpx.ReadError("connection reset"), _good({"1": "你好"})])
    out = translator.translate_batch(_segs("Hello"), "en", "zh")
    assert [s.text for s in out] == ["你好"]
    assert fake.calls == 2


def test_read_error_exhausting_retries_raises_translation_error():
    translator, fake = _translator([httpx.ReadError("x")] * 3, max_retries=3)
    with pytest.raises(TranslationError):
        translator.translate_batch(_segs("Hello"), "en", "zh")
    assert fake.calls == 3


def test_client_error_is_not_retried():
    translator, fake = _translator([ServiceError("model not found", 404)])
    with pytest.raises(TranslationError):
        translator.translate_batch(_segs("Hello"), "en", "zh")
    assert fake.calls == 1


def test_server_error_is_retried():
    translator, fake = _translator([ServiceError("boom", 500), _good({"1": "你好"})])
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
    translator, fake = _translator([_good({"1": "你好"}), "再见"])
    out = translator.translate_batch(_segs("Hello", "Bye"), "en", "zh")
    assert [s.text for s in out] == ["你好", "再见"]
    assert fake.calls == 2
    assert translator.get_failed_translations() == []


def test_unhealed_segment_keeps_its_failure_record():
    # Batch reply drops index 2 and the individual retry echoes the original
    # (a failed retry) — that record must survive.
    translator, fake = _translator([_good({"1": "你好"}), "Bye"])
    out = translator.translate_batch(_segs("Hello", "Bye"), "en", "zh")
    assert [s.text for s in out] == ["你好", "Bye"]
    assert [f["index"] for f in translator.get_failed_translations()] == [2]


def test_a_segment_the_model_never_answered_is_reported_as_untranslated(caplog):
    translator, _ = _translator([_good({"1": "你好"}), "Bye"])
    with caplog.at_level(logging.WARNING):
        translator.translate(_segs("Hello", "Bye"), "en", "zh")

    degradation = translator.untranslated_degradation("zh", 2)

    assert degradation is not None
    assert degradation.code == "translation_incomplete"
    assert degradation.message.startswith("zh: 1 of 2")
    assert "1 left untranslated" in caplog.text


def test_an_answer_identical_to_the_source_is_not_reported(caplog):
    # A name translates to itself; the individual retry agrees, so nothing is missing.
    translator, _ = _translator([_good({"1": "你好", "2": "Tokyo"}), "Tokyo"])
    with caplog.at_level(logging.WARNING):
        translator.translate(_segs("Hello", "Tokyo"), "en", "zh")

    assert translator.untranslated_degradation("zh", 2) is None
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
