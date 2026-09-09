"""What `transcribe` and `batch` actually write, end to end.

Both used to carry their own copy of the audio→transcribe→translate→save flow
and drifted from it: `batch` had no --bilingual / --timestamp-mode /
--split-sentences at all. These tests pin the files each command produces so a
future divergence shows up as a missing or misnamed output.

Whisper, ffmpeg and Ollama are replaced by fakes; the SRT files are real. The
modules being patched still import the ffmpeg-python and faster-whisper packages,
so skip rather than break the "tests run without torch/whisper installed"
contract.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("ffmpeg", reason="core.audio imports ffmpeg-python")
pytest.importorskip("faster_whisper", reason="core.transcriber imports faster_whisper")

from subtitle_forge.cli.app import app  # noqa: E402
from subtitle_forge.core import pipeline as pipeline_module  # noqa: E402
from subtitle_forge.core import transcriber as transcriber_module  # noqa: E402
from subtitle_forge.core import translator as translator_module  # noqa: E402
from subtitle_forge.models.subtitle import SubtitleSegment  # noqa: E402

runner = CliRunner()

# What the fake transcriber was asked to do, reset per test.
transcribe_calls = []


class _FakeInfo:
    language = "en"
    language_probability = 0.99


class _FakeTranscriber:
    use_whisperx = False

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_config(cls, cfg, **overrides):
        return cls()

    def is_model_cached(self):
        return True

    def unload_model(self):
        pass

    def transcribe(self, audio_path, **kwargs):
        transcribe_calls.append(kwargs)
        segments = [SubtitleSegment(index=1, start=0.0, end=1.5, text="Hello there")]
        return segments, _FakeInfo()


class _FakeTranslator:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_config(cls, cfg, **overrides):
        return cls()

    def translate(self, segments, source_lang, target_lang, progress_callback=None):
        return [
            SubtitleSegment(
                index=s.index, start=s.start, end=s.end, text=f"[{target_lang}] {s.text}"
            )
            for s in segments
        ]


class _FakeExtractor:
    def extract(self, video_path, output_path=None):
        scratch = Path(video_path).with_suffix(".scratch.wav")
        scratch.write_bytes(b"RIFF")
        return scratch


def _fake_components(monkeypatch):
    transcribe_calls.clear()
    monkeypatch.setattr(transcriber_module, "Transcriber", _FakeTranscriber)
    monkeypatch.setattr(translator_module, "SubtitleTranslator", _FakeTranslator)
    monkeypatch.setattr(pipeline_module, "AudioExtractor", _FakeExtractor)


def _video(tmp_path, name="clip.mp4"):
    path = tmp_path / name
    path.write_bytes(b"not really a video")
    return path


def _run(tmp_path, *args):
    return runner.invoke(app, ["--config", str(tmp_path / "config.yaml"), *args])


def test_transcribe_writes_the_output_path_it_was_given(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    video = _video(tmp_path)
    named = tmp_path / "elsewhere" / "chosen.srt"

    result = _run(tmp_path, "transcribe", str(video), "-o", str(named))

    assert result.exit_code == 0, result.output
    assert "Hello there" in named.read_text(encoding="utf-8")
    assert not (tmp_path / "clip.en.srt").exists()


def test_transcribe_defaults_to_the_detected_language_beside_the_video(
    tmp_path, monkeypatch
):
    _fake_components(monkeypatch)
    video = _video(tmp_path)

    result = _run(tmp_path, "transcribe", str(video))

    assert result.exit_code == 0, result.output
    assert "Hello there" in (tmp_path / "clip.en.srt").read_text(encoding="utf-8")
    assert not video.with_suffix(".scratch.wav").exists()


def test_transcribe_passes_its_timing_flags_down(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    video = _video(tmp_path)

    result = _run(
        tmp_path, "transcribe", str(video), "--timestamp-mode", "full", "--no-split-sentences"
    )

    assert result.exit_code == 0, result.output
    call = transcribe_calls[0]
    assert call["timestamp_config"]["mode"] == "full"
    assert call["timestamp_config"]["split_sentences"] is False


def test_batch_writes_original_and_translation_for_every_video(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos, "a.mp4")
    _video(videos, "b.mp4")

    result = _run(tmp_path, "batch", str(videos), "-t", "zh")

    assert result.exit_code == 0, result.output
    for stem in ("a", "b"):
        assert "Hello there" in (videos / f"{stem}.en.srt").read_text(encoding="utf-8")
        assert "[zh] Hello there" in (videos / f"{stem}.zh.srt").read_text(encoding="utf-8")


def test_batch_honours_bilingual(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos, "a.mp4")

    result = _run(tmp_path, "batch", str(videos), "-t", "zh", "--bilingual")

    assert result.exit_code == 0, result.output
    merged = (videos / "a.en-zh.srt").read_text(encoding="utf-8")
    assert "Hello there" in merged and "[zh] Hello there" in merged
    assert not (videos / "a.zh.srt").exists()


def test_batch_rejects_an_invalid_timestamp_mode_before_transcribing(
    tmp_path, monkeypatch
):
    _fake_components(monkeypatch)
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos, "a.mp4")

    result = _run(
        tmp_path, "batch", str(videos), "-t", "zh", "--timestamp-mode", "minimal-ish"
    )

    assert result.exit_code == 1
    assert transcribe_calls == []
