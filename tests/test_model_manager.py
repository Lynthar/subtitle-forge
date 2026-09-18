"""Tests for Ollama model availability matching.

Locks in the fix for the substring false-positive: requesting "qwen2.5:32b"
with only "qwen2.5:7b" installed used to report "available" (because "qwen2.5"
is a substring of "qwen2.5:7b"), so the download was skipped and translation
died with model-not-found only after transcription had already run.
"""

import pytest

from subtitle_forge.core.model_manager import OllamaModelManager


def _manager(installed):
    m = OllamaModelManager()
    m.list_models = lambda: list(installed)  # type: ignore[assignment]
    return m


def test_different_tag_is_not_a_false_positive():
    assert _manager(["qwen2.5:7b"]).is_model_available("qwen2.5:32b") is False


def test_exact_tag_matches():
    assert _manager(["qwen2.5:7b", "llama3:latest"]).is_model_available("qwen2.5:7b") is True


def test_bare_name_matches_latest_tag():
    assert _manager(["qwen2.5:latest"]).is_model_available("qwen2.5") is True


def test_bare_name_does_not_match_specific_tag():
    assert _manager(["qwen2.5:7b"]).is_model_available("qwen2.5") is False


def test_no_cross_model_substring_match():
    # Old behaviour: "llama3" in "tinyllama:latest" -> True. Must be False now.
    assert _manager(["tinyllama:latest"]).is_model_available("llama3") is False


def test_absent_model_is_unavailable():
    assert _manager(["qwen2.5:7b"]).is_model_available("gemma2:9b") is False


class _DownClient:
    """Stands in for ollama.Client when the daemon cannot be reached."""

    def list(self):
        raise ConnectionError("connection refused")


def _manager_with_down_daemon():
    m = OllamaModelManager()
    m._client = _DownClient()  # type: ignore[assignment]
    return m


# "Cannot ask Ollama" used to come back as [] and then as False, so every caller read an
# unreachable daemon as a missing model and offered a download that could not work.
def test_listing_failure_is_raised_not_an_empty_list():
    with pytest.raises(ConnectionError):
        _manager_with_down_daemon().list_models()


def test_listing_failure_does_not_read_as_model_missing():
    with pytest.raises(ConnectionError):
        _manager_with_down_daemon().is_model_available("qwen2.5:7b")


def test_ensure_model_does_not_start_a_download_it_cannot_finish():
    with pytest.raises(ConnectionError):
        _manager_with_down_daemon().ensure_model("qwen2.5:7b")
