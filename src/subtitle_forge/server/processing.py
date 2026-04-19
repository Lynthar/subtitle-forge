"""Video processing pipeline used by the HTTP server.

This is the same audio→transcribe→translate→save flow as `cli/commands/process`,
factored as a sync function with no progress-bar / interactive-prompt baggage.
The runner calls it inside `loop.run_in_executor`, so it runs off the event loop.

The Transcriber is held by the holder for the lifetime of the server — Whisper
model load takes 10–30s and is wasteful to repeat per job. The Translator is
constructed per call (it's just an Ollama HTTP client wrapper, no state).
"""

import logging
import threading
from pathlib import Path
from typing import Optional

from ..core.audio import AudioExtractor
from ..core.subtitle import SubtitleProcessor
from ..core.transcriber import Transcriber
from ..core.translator import SubtitleTranslator, TranslationConfig
from ..models.config import AppConfig
from .jobs import Job

logger = logging.getLogger(__name__)


class TranscriberHolder:
    """Lazy, thread-safe holder for a single Transcriber instance.

    Build is deferred to first use so the server boots even if Whisper isn't
    available yet (helpful for diagnostics). After the first build the same
    instance is reused for every job — the model stays resident in VRAM.
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._transcriber: Optional[Transcriber] = None
        self._lock = threading.Lock()

    def get(self) -> Transcriber:
        # Double-checked locking — fast path skips the lock once initialized.
        if self._transcriber is not None:
            return self._transcriber
        with self._lock:
            if self._transcriber is not None:
                return self._transcriber
            cfg = self._config.whisper
            logger.info("Loading Whisper model %s on %s", cfg.model, cfg.device)
            t = Transcriber(
                model_name=cfg.model,
                device=cfg.device,
                compute_type=cfg.compute_type,
                use_whisperx=cfg.use_whisperx,
                whisperx_align=cfg.whisperx_align,
                hf_token=cfg.hf_token,
                hf_endpoint=cfg.hf_endpoint,
            )
            if not t.is_model_cached():
                # Server should not run interactive download. Fail with a
                # clear instruction instead.
                raise RuntimeError(
                    f"Whisper model '{cfg.model}' is not cached. "
                    f"Run `subtitle-forge transcribe <some-video>` once on this "
                    f"machine to download it, then restart the server."
                )
            self._transcriber = t
            return t

    @property
    def is_loaded(self) -> bool:
        return self._transcriber is not None


def make_processor(config: AppConfig, holder: TranscriberHolder):
    """Returns a sync `(Job) -> List[dict]` callable bound to the given config."""

    def process(job: Job) -> list:
        return _run_job(job, config, holder.get())

    return process


def _run_job(job: Job, config: AppConfig, transcriber: Transcriber) -> list:
    video_path = Path(job.video_path)
    out_dir = video_path.parent
    stem = video_path.stem

    extractor = AudioExtractor()
    audio_path = extractor.extract(video_path)

    try:
        timestamp_config = (
            {
                "mode": config.timestamp.mode,
                "min_duration": config.timestamp.min_duration,
                "max_duration": config.timestamp.max_duration,
                "min_gap": config.timestamp.min_gap,
                "max_gap_warning": config.timestamp.max_gap_warning,
                "chars_per_second": config.timestamp.chars_per_second,
                "cjk_chars_per_second": config.timestamp.cjk_chars_per_second,
                "split_threshold": config.timestamp.split_threshold,
                "split_sentences": config.timestamp.split_sentences,
            }
            if config.timestamp.enabled
            else None
        )

        segments, info = transcriber.transcribe(
            audio_path,
            language=job.source_language,
            beam_size=config.whisper.beam_size,
            vad_filter=config.whisper.vad_filter,
            post_process=config.timestamp.enabled,
            timestamp_config=timestamp_config,
        )

        detected_lang = info.language
        # Surface the detected language back through the job record so callers
        # polling /jobs/{id} see it once transcription completes.
        job.source_language = detected_lang

        subtitle_processor = SubtitleProcessor()
        outputs: list = []

        if job.keep_original:
            original_srt = out_dir / f"{stem}.{detected_lang}.srt"
            subtitle_processor.save(segments, original_srt)
            outputs.append({"language": detected_lang, "path": str(original_srt)})

        translator = SubtitleTranslator(
            TranslationConfig(
                model=config.ollama.model,
                host=config.ollama.host,
                temperature=config.ollama.temperature,
                max_batch_size=config.ollama.max_batch_size,
                prompt_template=config.ollama.prompt_template,
                prompt_template_id=config.ollama.prompt_template_id,
            )
        )

        for lang in job.target_languages:
            if lang == detected_lang:
                logger.info("Skipping translation to %s (same as source)", lang)
                continue

            translated = translator.translate(segments, detected_lang, lang)

            if job.bilingual:
                merged = subtitle_processor.merge_bilingual(segments, translated)
                out_path = out_dir / f"{stem}.{detected_lang}-{lang}.srt"
                subtitle_processor.save(merged, out_path)
                outputs.append({"language": f"{detected_lang}-{lang}", "path": str(out_path)})
            else:
                out_path = out_dir / f"{stem}.{lang}.srt"
                subtitle_processor.save(translated, out_path)
                outputs.append({"language": lang, "path": str(out_path)})

        return outputs

    finally:
        # Always remove the extracted audio scratch file.
        try:
            audio_path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to clean up audio scratch file %s: %s", audio_path, e)
