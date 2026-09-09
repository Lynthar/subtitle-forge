"""End-to-end contract of `core.pipeline.run_pipeline` with fake components.

The shared flow — extract audio, transcribe, save the original, translate each
target, clean up the scratch file — had no coverage at all, so nothing stopped
an entry point from quietly growing its own copy of it.

Fake transcriber / translator and a stubbed AudioExtractor; no ffmpeg binary,
Whisper weights or Ollama needed. Subtitle files are written for real. Importing
core.pipeline does pull in the ffmpeg-python and faster-whisper packages, so skip
rather than break the "tests run without torch/whisper installed" contract.
"""

from pathlib import Path

import pytest

pytest.importorskip("ffmpeg", reason="core.audio imports ffmpeg-python")
pytest.importorskip("faster_whisper", reason="core.transcriber imports faster_whisper")

from subtitle_forge.core import pipeline as pipeline_module  # noqa: E402
from subtitle_forge.core.pipeline import PipelineHooks, run_pipeline  # noqa: E402
from subtitle_forge.models.config import AppConfig  # noqa: E402
from subtitle_forge.models.subtitle import SubtitleSegment  # noqa: E402


class _FakeInfo:
    def __init__(self, language="en", language_probability=0.98):
        self.language = language
        self.language_probability = language_probability


class _FakeTranscriber:
    """Records what it was asked to do and returns two fixed segments."""

    def __init__(self, language="en"):
        self.info = _FakeInfo(language=language)
        self.calls = []
        self.raise_on_transcribe = None

    def transcribe(self, audio_path, **kwargs):
        self.calls.append({"audio_path": Path(audio_path), **kwargs})
        if self.raise_on_transcribe is not None:
            raise self.raise_on_transcribe
        segments = [
            SubtitleSegment(index=1, start=0.0, end=1.5, text="Hello there"),
            SubtitleSegment(index=2, start=2.0, end=3.5, text="General Kenobi"),
        ]
        return segments, self.info


class _FakeTranslator:
    """Prefixes each line with the target language; records every call."""

    def __init__(self):
        self.calls = []

    def translate(self, segments, source_lang, target_lang, progress_callback=None):
        self.calls.append((source_lang, target_lang))
        if progress_callback is not None:
            progress_callback(len(segments), len(segments))
        return [
            SubtitleSegment(
                index=s.index, start=s.start, end=s.end, text=f"[{target_lang}] {s.text}"
            )
            for s in segments
        ]


@pytest.fixture()
def fake_audio(tmp_path, monkeypatch):
    """Replaces AudioExtractor so extract() just drops a scratch file."""
    created = []

    class _FakeExtractor:
        def extract(self, video_path, output_path=None):
            scratch = tmp_path / f"{Path(video_path).stem}.scratch.wav"
            scratch.write_bytes(b"RIFF")
            created.append(scratch)
            return scratch

    monkeypatch.setattr(pipeline_module, "AudioExtractor", _FakeExtractor)
    return created


@pytest.fixture()
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video")
    return path


def _read(path):
    return path.read_text(encoding="utf-8")


def test_saves_original_and_one_translation(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    translator = _FakeTranslator()

    result = run_pipeline(
        video,
        AppConfig(),
        transcriber=_FakeTranscriber(),
        translator=translator,
        target_languages=["zh"],
        output_dir=out_dir,
    )

    assert result.detected_language == "en"
    assert result.segment_count == 2
    assert [(o.language, o.path.name) for o in result.outputs] == [
        ("en", "clip.en.srt"),
        ("zh", "clip.zh.srt"),
    ]
    assert "Hello there" in _read(out_dir / "clip.en.srt")
    assert "[zh] Hello there" in _read(out_dir / "clip.zh.srt")
    assert translator.calls == [("en", "zh")]
    # The scratch file is the pipeline's own to clean up.
    assert fake_audio and not fake_audio[0].exists()


def test_bilingual_merges_into_one_file(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = run_pipeline(
        video,
        AppConfig(),
        transcriber=_FakeTranscriber(),
        translator=_FakeTranslator(),
        target_languages=["zh"],
        output_dir=out_dir,
        keep_original=False,
        bilingual=True,
    )

    assert [(o.language, o.path.name) for o in result.outputs] == [
        ("en-zh", "clip.en-zh.srt")
    ]
    merged = _read(out_dir / "clip.en-zh.srt")
    assert "Hello there" in merged and "[zh] Hello there" in merged
    assert not (out_dir / "clip.en.srt").exists()


def test_target_equal_to_source_is_skipped(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    translator = _FakeTranslator()
    skipped = []

    result = run_pipeline(
        video,
        AppConfig(),
        transcriber=_FakeTranscriber(language="en"),
        translator=translator,
        target_languages=["en", "zh"],
        output_dir=out_dir,
        hooks=PipelineHooks(on_translation_skipped=skipped.append),
    )

    assert skipped == ["en"]
    assert translator.calls == [("en", "zh")]
    assert [o.language for o in result.outputs] == ["en", "zh"]


def test_transcribe_only_needs_no_translator(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = run_pipeline(
        video,
        AppConfig(),
        transcriber=_FakeTranscriber(),
        target_languages=[],
        output_dir=out_dir,
    )

    assert [o.path.name for o in result.outputs] == ["clip.en.srt"]
    assert not fake_audio[0].exists()


def test_translation_targets_without_a_translator_fail_before_any_work(
    tmp_path, video, fake_audio
):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    transcriber = _FakeTranscriber()

    with pytest.raises(ValueError, match="translator"):
        run_pipeline(
            video,
            AppConfig(),
            transcriber=transcriber,
            target_languages=["zh"],
            output_dir=out_dir,
        )

    assert transcriber.calls == []
    assert fake_audio == []


def test_original_output_path_pins_the_single_output(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pinned = out_dir / "chosen.name.srt"

    result = run_pipeline(
        video,
        AppConfig(),
        transcriber=_FakeTranscriber(),
        target_languages=[],
        output_dir=out_dir,
        original_output_path=pinned,
    )

    assert [o.path for o in result.outputs] == [pinned]
    assert pinned.exists()
    assert not (out_dir / "clip.en.srt").exists()


def test_scratch_audio_is_removed_when_transcription_fails(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    transcriber = _FakeTranscriber()
    transcriber.raise_on_transcribe = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(
            video,
            AppConfig(),
            transcriber=transcriber,
            translator=_FakeTranslator(),
            target_languages=["zh"],
            output_dir=out_dir,
        )

    assert fake_audio and not fake_audio[0].exists()


def test_invalid_language_code_is_rejected_before_extraction(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(ValueError):
        run_pipeline(
            video,
            AppConfig(),
            transcriber=_FakeTranscriber(),
            translator=_FakeTranslator(),
            target_languages=["../evil"],
            output_dir=out_dir,
        )

    assert fake_audio == []


def test_timestamp_and_vad_settings_reach_the_transcriber(tmp_path, video, fake_audio):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    transcriber = _FakeTranscriber()

    run_pipeline(
        video,
        AppConfig(),
        transcriber=transcriber,
        target_languages=[],
        output_dir=out_dir,
        timestamp_mode="full",
        split_sentences=False,
        vad_parameters={"speech_pad_ms": 111},
    )

    call = transcriber.calls[0]
    assert call["vad_parameters"] == {"speech_pad_ms": 111}
    assert call["timestamp_config"]["mode"] == "full"
    assert call["timestamp_config"]["split_sentences"] is False
    assert call["post_process"] is True
