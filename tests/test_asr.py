"""`create_backend`: asr.backend picks the backend, and that backend's config section reaches it."""

import pytest

from subtitle_forge.core.asr import create_backend
from subtitle_forge.models.config import AppConfig


def test_an_unknown_backend_fails_validation_and_the_factory():
    config = AppConfig()
    config.asr.backend = "nope"

    with pytest.raises(ValueError, match="asr.backend"):
        config.validate()
    with pytest.raises(ValueError, match="nope"):
        create_backend(config)


def test_the_whisper_backend_takes_its_settings_from_the_whisper_section():
    pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")
    from subtitle_forge.core.transcriber import Transcriber

    config = AppConfig()
    config.whisper.beam_size = 3
    config.whisper.vad_filter = False
    config.whisper.batch_size = 4
    config.whisper.speech_pad_ms = 123

    backend = create_backend(config)

    assert isinstance(backend, Transcriber)
    assert (backend.beam_size, backend.vad_filter, backend.batch_size) == (3, False, 4)
    assert backend.vad_parameters["speech_pad_ms"] == 123


def test_overrides_replace_the_configured_values():
    pytest.importorskip("faster_whisper", reason="transcriber imports faster_whisper")

    backend = create_backend(AppConfig(), model_name="tiny", beam_size=1)

    assert (backend.model_name, backend.beam_size) == ("tiny", 1)
