"""Core video → subtitles pipeline shared by the CLI and the HTTP server.

A single audio→transcribe→translate→save flow that both entry points call.
Optional hooks (`PipelineHooks`) let the CLI plug in its Rich progress bar
without the pipeline knowing or caring about UI; the server passes no hooks
and gets a quiet, side-effect-free run.

Caller responsibilities:
- Build the Transcriber and ensure its Whisper model is cached locally.
- Build the SubtitleTranslator and ensure its Ollama model is available.
- Provide an output_dir that exists.
- Clean up Transcriber state if needed (e.g. unload_model()).

The pipeline cleans up its own audio scratch file in a finally block.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ContextManager, List, Optional

from ..models.config import AppConfig
from .audio import AudioExtractor
from .subtitle import SubtitleProcessor
from .transcriber import Transcriber
from .translator import SubtitleTranslator

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    """One produced subtitle file."""

    language: str  # e.g. "en", "ja", "en-zh" (bilingual label)
    path: Path


@dataclass
class PipelineResult:
    """End state of a pipeline run."""

    detected_language: str
    language_probability: float
    segment_count: int
    outputs: List[PipelineOutput] = field(default_factory=list)


@dataclass
class PipelineHooks:
    """Optional UI hooks. Each defaults to None (no-op).

    Server callers pass nothing; the CLI wires these to its progress bar.
    """

    # Called once after the audio scratch file has been written.
    on_audio_extracted: Optional[Callable[[Path], None]] = None

    # Called once after transcription is complete.
    # Args: (segment_count, detected_language)
    on_transcribe_complete: Optional[Callable[[int, str], None]] = None

    # Called once after the original-language SRT has been saved.
    on_original_saved: Optional[Callable[[Path], None]] = None

    # Called once per target language that was skipped (same as source).
    on_translation_skipped: Optional[Callable[[str], None]] = None

    # Context manager factory for per-language translation progress.
    # Args: (target_lang, total_segments)
    # Yields a callback (completed: int, total: int) -> None.
    # If None, translation runs without a progress callback.
    translation_progress_ctx: Optional[
        Callable[[str, int], ContextManager[Callable[[int, int], None]]]
    ] = None

    # Called once per saved translation file.
    # Args: (output_path, label) — label is e.g. "zh" or "en-zh" (bilingual)
    on_translation_saved: Optional[Callable[[Path, str], None]] = None


def build_timestamp_config(
    config: AppConfig,
    *,
    mode_override: Optional[str] = None,
    split_sentences_override: Optional[bool] = None,
) -> Optional[dict]:
    """Build the timestamp_config dict for Transcriber.transcribe().

    Returns None when post-processing is disabled at the config level —
    callers should treat None as "skip the timestamp processor entirely".
    """
    if not config.timestamp.enabled:
        return None
    return {
        "mode": mode_override or config.timestamp.mode,
        "min_duration": config.timestamp.min_duration,
        "max_duration": config.timestamp.max_duration,
        "min_gap": config.timestamp.min_gap,
        "max_gap_warning": config.timestamp.max_gap_warning,
        "chars_per_second": config.timestamp.chars_per_second,
        "cjk_chars_per_second": config.timestamp.cjk_chars_per_second,
        "split_threshold": config.timestamp.split_threshold,
        "split_sentences": (
            split_sentences_override
            if split_sentences_override is not None
            else config.timestamp.split_sentences
        ),
        "lead_in_ms": config.timestamp.lead_in_ms,
        "linger_ms": config.timestamp.linger_ms,
    }


def build_vad_parameters(
    config: AppConfig,
    *,
    mode: Optional[str] = None,
    speech_pad_ms: Optional[int] = None,
    min_silence_duration_ms: Optional[int] = None,
) -> dict:
    """Build the VAD parameters dict for Transcriber.transcribe().

    Three layers of precedence (highest first):
    1. Explicit `speech_pad_ms` / `min_silence_duration_ms` arguments
    2. Named preset via `mode` (one of VAD_PRESETS keys)
    3. config.whisper.{speech_pad_ms, min_silence_duration_ms}
    """
    if mode is not None:
        return Transcriber.get_vad_parameters(
            mode=mode,
            speech_pad_ms=speech_pad_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )
    return Transcriber.get_vad_parameters(
        speech_pad_ms=speech_pad_ms if speech_pad_ms is not None else config.whisper.speech_pad_ms,
        min_silence_duration_ms=(
            min_silence_duration_ms
            if min_silence_duration_ms is not None
            else config.whisper.min_silence_duration_ms
        ),
    )


def run_pipeline(
    video_path: Path,
    config: AppConfig,
    *,
    transcriber: Transcriber,
    translator: SubtitleTranslator,
    target_languages: List[str],
    output_dir: Path,
    source_language: Optional[str] = None,
    keep_original: bool = True,
    bilingual: bool = False,
    timestamp_mode: Optional[str] = None,
    split_sentences: Optional[bool] = None,
    post_process: bool = True,
    vad_parameters: Optional[dict] = None,
    hooks: Optional[PipelineHooks] = None,
) -> PipelineResult:
    """Core video → subtitles pipeline.

    See module docstring for caller responsibilities. Hooks are optional
    callbacks for progress UI; with hooks=None this runs silently (server
    mode) and produces no console output of its own.
    """
    hooks = hooks or PipelineHooks()
    extractor = AudioExtractor()
    subtitle_processor = SubtitleProcessor()
    stem = video_path.stem

    audio_path = extractor.extract(video_path)
    if hooks.on_audio_extracted is not None:
        hooks.on_audio_extracted(audio_path)

    try:
        timestamp_config = build_timestamp_config(
            config,
            mode_override=timestamp_mode,
            split_sentences_override=split_sentences,
        )

        segments, info = transcriber.transcribe(
            audio_path,
            language=source_language,
            beam_size=config.whisper.beam_size,
            vad_filter=config.whisper.vad_filter,
            vad_parameters=vad_parameters,
            post_process=post_process and config.timestamp.enabled,
            timestamp_config=timestamp_config,
        )
        detected_language = info.language

        if hooks.on_transcribe_complete is not None:
            hooks.on_transcribe_complete(len(segments), detected_language)

        outputs: List[PipelineOutput] = []

        if keep_original:
            original_srt = output_dir / f"{stem}.{detected_language}.srt"
            subtitle_processor.save(segments, original_srt)
            outputs.append(
                PipelineOutput(language=detected_language, path=original_srt)
            )
            if hooks.on_original_saved is not None:
                hooks.on_original_saved(original_srt)

        for lang in target_languages:
            if lang == detected_language:
                if hooks.on_translation_skipped is not None:
                    hooks.on_translation_skipped(lang)
                else:
                    logger.info(
                        "Skipping translation to %s (same as source)", lang
                    )
                continue

            if hooks.translation_progress_ctx is not None:
                with hooks.translation_progress_ctx(lang, len(segments)) as progress_cb:
                    translated = translator.translate(
                        segments,
                        detected_language,
                        lang,
                        progress_callback=progress_cb,
                    )
            else:
                translated = translator.translate(
                    segments, detected_language, lang
                )

            if bilingual:
                merged = subtitle_processor.merge_bilingual(segments, translated)
                out_path = output_dir / f"{stem}.{detected_language}-{lang}.srt"
                subtitle_processor.save(merged, out_path)
                label = f"{detected_language}-{lang}"
            else:
                out_path = output_dir / f"{stem}.{lang}.srt"
                subtitle_processor.save(translated, out_path)
                label = lang

            outputs.append(PipelineOutput(language=label, path=out_path))

            if hooks.on_translation_saved is not None:
                hooks.on_translation_saved(out_path, label)

        return PipelineResult(
            detected_language=detected_language,
            language_probability=info.language_probability,
            segment_count=len(segments),
            outputs=outputs,
        )

    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(
                "Failed to clean up audio scratch file %s: %s", audio_path, e
            )
