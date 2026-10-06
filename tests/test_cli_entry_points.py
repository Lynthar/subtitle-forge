"""The files each CLI entry point writes; Whisper, ffmpeg and Ollama are fakes, SRTs are real.

Skips when ffmpeg-python or faster-whisper is absent, since the patched modules import them."""

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from typer.testing import CliRunner

pytest.importorskip("ffmpeg", reason="core.audio imports ffmpeg-python")
pytest.importorskip("faster_whisper", reason="core.transcriber imports faster_whisper")

from subtitle_forge.cli.app import app  # noqa: E402
from subtitle_forge.core import asr as asr_module  # noqa: E402
from subtitle_forge.core import pipeline as pipeline_module  # noqa: E402
from subtitle_forge.core import translator as translator_module  # noqa: E402
from subtitle_forge.core.asr import TranscriptionInfo  # noqa: E402
from subtitle_forge.core.transcriber import Transcriber  # noqa: E402
from subtitle_forge.models.degradation import Degradation  # noqa: E402
from subtitle_forge.models.subtitle import SubtitleSegment  # noqa: E402

runner = CliRunner()

# What the fake transcriber was asked to do, reset per test.
transcribe_calls = []
# What the fakes report as degraded, reset per test.
backend_degradations = []
untranslated = {}  # target language -> its Degradation

CPU = Degradation("cpu_fallback", "CUDA is unavailable; ran on the CPU")


class _FakeTranscriber:
    def is_model_cached(self):
        return True

    def unload_model(self):
        pass

    def transcribe(self, audio_path, **kwargs):
        transcribe_calls.append(kwargs)
        segments = [SubtitleSegment(index=1, start=0.0, end=1.5, text="Hello there")]
        return segments, TranscriptionInfo("en", 0.99, 60.0, list(backend_degradations))


class _FakeTranslator:
    LANGUAGE_NAMES: ClassVar[dict] = {}
    chat = SimpleNamespace(label="Fake", endpoint="fake://", model="fake")

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_config(cls, cfg, **overrides):
        return cls()

    def check_connection(self):
        return True

    def check_model_available(self):
        return True

    def effective_batch_size(self):
        return 10

    def untranslated_degradation(self, target_lang, total):
        return untranslated.get(target_lang)

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
    backend_degradations.clear()
    untranslated.clear()
    monkeypatch.setattr(asr_module, "create_backend", lambda config, **o: _FakeTranscriber())
    monkeypatch.setattr(translator_module, "SubtitleTranslator", _FakeTranslator)
    monkeypatch.setattr(pipeline_module, "AudioExtractor", _FakeExtractor)


def _video(tmp_path, name="clip.mp4"):
    path = tmp_path / name
    path.write_bytes(b"not really a video")
    return path


def _run(tmp_path, *args):
    return runner.invoke(app, ["--config", str(tmp_path / "config.yaml"), *args])


def _configure_bilingual_only(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "output:\n  keep_original: false\n  bilingual: true\n", encoding="utf-8"
    )


def test_transcribe_writes_the_output_path_it_was_given(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    video = _video(tmp_path)
    named = tmp_path / "elsewhere" / "chosen.srt"

    result = _run(tmp_path, "transcribe", str(video), "-o", str(named))

    assert result.exit_code == 0, result.output
    assert "Hello there" in named.read_text(encoding="utf-8")
    assert not (tmp_path / "clip.en.srt").exists()


def test_transcribe_defaults_to_the_detected_language_beside_the_video(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    video = _video(tmp_path)

    result = _run(tmp_path, "transcribe", str(video))

    assert result.exit_code == 0, result.output
    assert "Hello there" in (tmp_path / "clip.en.srt").read_text(encoding="utf-8")
    assert not video.with_suffix(".scratch.wav").exists()


def test_transcribe_passes_its_timing_flags_down(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    video = _video(tmp_path)
    used = []

    class _RecordingProcessor:
        def __init__(self, config, language=None):
            used.append(config)

        def process(self, segments, audio_duration=None):
            return segments

    monkeypatch.setattr(pipeline_module, "TimestampProcessor", _RecordingProcessor)

    result = _run(
        tmp_path, "transcribe", str(video), "--timestamp-mode", "full", "--no-split-sentences"
    )

    assert result.exit_code == 0, result.output
    assert [(c.mode, c.split_sentences) for c in used] == [("full", False)]


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


def test_process_falls_back_to_config_output_flags(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    _configure_bilingual_only(tmp_path)
    video = _video(tmp_path)

    result = _run(tmp_path, "process", str(video), "-t", "zh")

    assert result.exit_code == 0, result.output
    assert "[zh] Hello there" in (tmp_path / "clip.en-zh.srt").read_text(encoding="utf-8")
    assert not (tmp_path / "clip.en.srt").exists()


def test_batch_falls_back_to_config_output_flags(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    _configure_bilingual_only(tmp_path)
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos, "a.mp4")

    result = _run(tmp_path, "batch", str(videos), "-t", "zh")

    assert result.exit_code == 0, result.output
    assert "[zh] Hello there" in (videos / "a.en-zh.srt").read_text(encoding="utf-8")
    assert not (videos / "a.en.srt").exists()


def test_translate_falls_back_to_config_bilingual(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    _configure_bilingual_only(tmp_path)
    srt = tmp_path / "clip.en.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,500\nHello there\n", encoding="utf-8")

    result = _run(tmp_path, "translate", str(srt), "-t", "zh")

    assert result.exit_code == 0, result.output
    merged = (tmp_path / "clip.en-zh.srt").read_text(encoding="utf-8")
    assert "Hello there" in merged and "[zh] Hello there" in merged

    result = _run(tmp_path, "translate", str(srt), "-t", "zh", "--no-bilingual")

    assert result.exit_code == 0, result.output
    assert "[zh] Hello there" in (tmp_path / "clip.zh.srt").read_text(encoding="utf-8")


def test_batch_rejects_an_invalid_timestamp_mode_before_transcribing(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos, "a.mp4")

    result = _run(tmp_path, "batch", str(videos), "-t", "zh", "--timestamp-mode", "minimal-ish")

    assert result.exit_code == 1
    assert transcribe_calls == []


def test_batch_debug_log_creates_a_new_output_dir(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    video = _video(tmp_path)
    out = tmp_path / "new" / "subs"

    result = _run(tmp_path, "batch", str(video), "-t", "zh", "-o", str(out), "--save-debug-log")

    assert result.exit_code == 0, result.output
    assert "[zh] Hello there" in (out / "clip.zh.srt").read_text(encoding="utf-8")
    assert (out / "clip_debug" / "run.log").exists()


def test_process_debug_log_creates_a_new_output_dir(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    video = _video(tmp_path)
    out = tmp_path / "new" / "subs"

    result = _run(tmp_path, "process", str(video), "-t", "zh", "-o", str(out), "--save-debug-log")

    assert result.exit_code == 0, result.output
    assert "[zh] Hello there" in (out / "clip.zh.srt").read_text(encoding="utf-8")
    assert (out / "clip_debug" / "run.log").exists()


def test_whisper_only_flags_are_reported_and_dropped_under_another_backend(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    built = []
    monkeypatch.setattr(
        asr_module, "create_backend", lambda config, **o: built.append(o) or _FakeTranscriber()
    )
    (tmp_path / "config.yaml").write_text("asr:\n  backend: qwen3_asr\n", encoding="utf-8")

    result = _run(
        tmp_path,
        "transcribe",
        str(_video(tmp_path)),
        "--whisperx",
        "--model",
        "Qwen/Qwen3-ASR-0.6B",
    )

    assert result.exit_code == 0, result.output
    assert "Whisper-only" in result.output
    assert built == [{"model_name": "Qwen/Qwen3-ASR-0.6B"}]


def test_a_degraded_transcription_says_so_and_exits_3(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    backend_degradations.append(CPU)
    video = _video(tmp_path)

    result = _run(tmp_path, "transcribe", str(video))

    assert result.exit_code == 3, result.output
    assert "ran on the CPU" in result.output
    assert "Hello there" in (tmp_path / "clip.en.srt").read_text(encoding="utf-8")


def test_auto_model_without_readable_vram_falls_back_to_small_and_says_so(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    built = []
    monkeypatch.setattr(
        asr_module, "create_backend", lambda config, **o: built.append(o) or _FakeTranscriber()
    )
    monkeypatch.setattr(Transcriber, "select_optimal_model", classmethod(lambda cls: None))

    result = _run(tmp_path, "transcribe", str(_video(tmp_path)), "--auto-model")

    assert result.exit_code == 3, result.output
    assert built[0]["model_name"] == "small"
    assert "fell back to small" in result.output


def test_process_reports_an_incomplete_translation_and_exits_3(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    untranslated["zh"] = Degradation("translation_incomplete", "zh: 1 of 1 segments untranslated")

    result = _run(tmp_path, "process", str(_video(tmp_path)), "-t", "zh")

    assert result.exit_code == 3, result.output
    assert "zh: 1 of 1 segments untranslated" in result.output


def test_translate_reports_an_incomplete_translation_and_exits_3(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    untranslated["zh"] = Degradation("translation_incomplete", "zh: 1 of 1 segments untranslated")
    srt = tmp_path / "clip.en.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,500\nHello there\n", encoding="utf-8")

    result = _run(tmp_path, "translate", str(srt), "-t", "zh")

    assert result.exit_code == 3, result.output
    assert "zh: 1 of 1 segments untranslated" in result.output


def test_batch_lists_a_degradation_every_task_shares_once_and_exits_3(tmp_path, monkeypatch):
    _fake_components(monkeypatch)
    backend_degradations.append(CPU)
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos, "a.mp4")
    _video(videos, "b.mp4")

    result = _run(tmp_path, "batch", str(videos), "-t", "zh")

    assert result.exit_code == 3, result.output
    assert "Degraded in every task" in result.output
    assert result.output.count("ran on the CPU") == 1
    assert "2 of 2 task(s) completed with degradations" in result.output
