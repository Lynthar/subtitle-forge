"""Video processing entry point for the HTTP server.

Wraps the shared `core/pipeline.run_pipeline` with server-specific lifecycle:
a TranscriberHolder so the Whisper model stays resident across jobs, and
job-state mutation so callers polling /jobs/{id} see the detected language
as soon as transcription completes.

The runner invokes `make_processor(...)`'s closure inside
`loop.run_in_executor`, so this all runs off the event loop.
"""

import logging
import threading
from pathlib import Path
from typing import Optional

from ..core.pipeline import build_vad_parameters, run_pipeline
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

    vad_params = build_vad_parameters(config)

    result = run_pipeline(
        video_path,
        config,
        transcriber=transcriber,
        translator=translator,
        target_languages=job.target_languages,
        output_dir=video_path.parent,
        source_language=job.source_language,
        keep_original=job.keep_original,
        bilingual=job.bilingual,
        vad_parameters=vad_params,
        # No hooks — server runs silently; logging inside translator/
        # transcriber covers operational visibility.
    )

    # Surface the detected language back through the job record so callers
    # polling /jobs/{id} see it once transcription completes.
    job.source_language = result.detected_language

    return [
        {"language": o.language, "path": str(o.path)}
        for o in result.outputs
    ]
