"""CLI main application."""

from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console

from .commands import transcribe, translate, batch, config, serve
from ..models.config import AppConfig
from ..utils.logger import setup_logging

app = typer.Typer(
    name="subtitle-forge",
    help="Local video subtitle generation and translation tool",
    add_completion=True,
    no_args_is_help=True,
)

console = Console()

# transcribe/translate/batch/serve expose a single action each, so they register as plain root
# commands. **Never attach them with add_typer()** — that makes a Click *group*, which never
# auto-invokes its lone command, so `subtitle-forge transcribe <video>` fails with No such command.
app.command("transcribe", help="Transcribe video to subtitles")(transcribe.transcribe_video)
app.command("translate", help="Translate existing subtitles")(translate.translate_subtitle)
app.command("batch", help="Batch process multiple videos")(batch.batch_process)
app.command("serve", help="Run as an HTTP server (job-based REST API)")(serve.serve)
app.add_typer(config.app, name="config", help="Configuration management")

# Global config
_config: Optional[AppConfig] = None
# The config subcommands need the PATH, not just the loaded object, so that
# `--config custom.yaml config set ...` reads AND writes custom.yaml instead of reading it
# and saving to the default location. None = default location.
_config_path: Optional[Path] = None


def get_config() -> AppConfig:
    """Get current configuration."""
    global _config
    if _config is None:
        _config = AppConfig.load(_config_path)
    return _config


def get_config_path() -> Optional[Path]:
    """The --config override from the root callback (None = default path)."""
    return _config_path


@app.callback()
def main(
    ctx: typer.Context,
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file path",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output mode",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Quiet mode, only show errors",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Log file path",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="Disable progress bar",
    ),
):
    """subtitle-forge - Local video subtitle generation and translation tool"""
    global _config, _config_path

    # Load configuration. load() validates field ranges and raises ValueError
    # with the full list of problems — surface that as a clean error instead
    # of a traceback.
    try:
        if config_file:
            _config_path = config_file
            _config = AppConfig.load(config_file)
        else:
            _config = get_config()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # Setup logging
    log_level = "DEBUG" if verbose else ("ERROR" if quiet else _config.log_level)
    setup_logging(log_level, str(log_file) if log_file else _config.log_file)

    # Wire --quiet / --no-progress into the Rich helpers — before this they
    # were parsed but had no effect on panels or progress bars.
    from ..utils.progress import set_ui_options
    set_ui_options(quiet=quiet, no_progress=no_progress)

    # Store in context
    ctx.ensure_object(dict)
    ctx.obj["config"] = _config
    ctx.obj["no_progress"] = no_progress


@app.command()
def process(
    video: Path = typer.Argument(..., help="Video file path", exists=True),
    target_lang: List[str] = typer.Option(
        ...,
        "--target-lang",
        "-t",
        help="Target language(s) (can be specified multiple times)",
    ),
    source_lang: Optional[str] = typer.Option(
        None,
        "--source-lang",
        "-s",
        help="Source language (auto-detect if not specified)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory",
    ),
    whisper_model: Optional[str] = typer.Option(
        None,
        "--whisper-model",
        help="Whisper model name",
    ),
    ollama_model: Optional[str] = typer.Option(
        None,
        "--ollama-model",
        help="Ollama model name",
    ),
    keep_original: Optional[bool] = typer.Option(
        None,
        "--keep-original/--no-keep-original",
        help="Keep original language subtitles (default: config output.keep_original)",
    ),
    bilingual: Optional[bool] = typer.Option(
        None,
        "--bilingual/--no-bilingual",
        help="Generate bilingual subtitles (default: config output.bilingual)",
    ),
    # VAD options for subtitle timing
    vad_mode: Optional[str] = typer.Option(
        None,
        "--vad-mode",
        help="VAD preset mode: default, aggressive, relaxed, precise",
    ),
    speech_pad: Optional[int] = typer.Option(
        None,
        "--speech-pad",
        help="Speech padding in milliseconds (overrides vad-mode)",
    ),
    min_silence: Optional[int] = typer.Option(
        None,
        "--min-silence",
        help="Minimum silence duration in ms for segment breaks (overrides vad-mode)",
    ),
    # Prompt template option
    prompt_template: Optional[str] = typer.Option(
        None,
        "--prompt-template",
        "-p",
        help="Prompt template ID from library (use 'config list-prompts' to see available)",
    ),
    # WhisperX options
    use_whisperx: Optional[bool] = typer.Option(
        None,
        "--whisperx/--no-whisperx",
        help="Use WhisperX for better timestamp accuracy",
    ),
    post_process: bool = typer.Option(
        True,
        "--post-process/--no-post-process",
        help="Enable timestamp post-processing to fix timing issues",
    ),
    timestamp_mode: Optional[str] = typer.Option(
        None,
        "--timestamp-mode",
        help="Timestamp processing mode: off, minimal (default), full",
    ),
    split_sentences: Optional[bool] = typer.Option(
        None,
        "--split-sentences/--no-split-sentences",
        help="Split multi-sentence segments using word timestamps for better timing",
    ),
    hf_mirror: Optional[str] = typer.Option(
        None,
        "--hf-mirror",
        help="HuggingFace mirror URL (e.g., https://hf-mirror.com)",
    ),
    save_failed_log: bool = typer.Option(
        False,
        "--save-failed-log",
        help="Save failed translations to a JSON log file for debugging",
    ),
    save_debug_log: bool = typer.Option(
        False,
        "--save-debug-log",
        help="Save detailed debug logs and failure reports (creates {video}_debug/ folder)",
    ),
):
    """
    Process a video: extract audio -> transcribe -> translate -> save subtitles

    Example:
        subtitle-forge process video.mp4 --target-lang zh
        subtitle-forge process video.mp4 -t zh -t ja --bilingual
    """
    from ..core.transcriber import Transcriber
    from ..core.translator import SubtitleTranslator, TranslationConfig
    from ..utils.progress import (
        SubtitleProgress,
        TranslationProgressTracker,
        print_success,
        print_error,
        print_info,
        print_warning,
        print_translation_explainer,
        progress_disabled,
    )

    cfg = get_config()

    # Override config if specified
    if whisper_model:
        cfg.whisper.model = whisper_model
    if ollama_model:
        cfg.ollama.model = ollama_model
    if prompt_template:
        cfg.ollama.prompt_template_id = prompt_template

    output_dir = output_dir or video.parent
    # Fall back to config when the flag wasn't passed (typer default None), so
    # output.keep_original / output.bilingual in config.yaml actually take effect.
    keep_original = keep_original if keep_original is not None else cfg.output.keep_original
    bilingual = bilingual if bilingual is not None else cfg.output.bilingual
    progress = SubtitleProgress()

    # Handle --save-debug-log option
    debug_dir = None
    debug_log_path = None
    debug_failed_log_path = None
    if save_debug_log:
        debug_dir = output_dir / f"{video.stem}_debug"
        debug_dir.mkdir(exist_ok=True)
        debug_log_path = str(debug_dir / "run.log")
        debug_failed_log_path = str(debug_dir / "translation_failures.json")
        # File handler captures DEBUG; console stays at INFO so the terminal is not flooded with
        # third-party stack traces (torio's FFmpeg-extension probing, which Rich renders in full).
        setup_logging(level="DEBUG", log_file=debug_log_path, console_level="INFO")

    # Build VAD parameters with the full precedence: CLI flag > --vad-mode preset > config.
    # Going through build_vad_parameters (not the bare Transcriber.get_vad_parameters, which
    # ignores config) is what makes configured VAD tuning take effect on the CLI too.
    from ..core.pipeline import build_vad_parameters
    vad_params = build_vad_parameters(
        cfg,
        mode=vad_mode,
        speech_pad_ms=speech_pad,
        min_silence_duration_ms=min_silence,
    )

    try:
        # ========== Phase 1: Prepare models (outside main progress bar) ==========

        # Determine WhisperX usage
        whisperx_enabled = use_whisperx if use_whisperx is not None else cfg.whisper.use_whisperx

        # Initialize transcriber
        # Determine HuggingFace endpoint (CLI option takes precedence)
        hf_endpoint = hf_mirror or cfg.whisper.hf_endpoint

        transcriber = Transcriber(
            model_name=cfg.whisper.model,
            device=cfg.whisper.device,
            compute_type=cfg.whisper.compute_type,
            download_root=cfg.whisper.download_root,
            use_whisperx=whisperx_enabled,
            whisperx_align=cfg.whisper.whisperx_align,
            hf_token=cfg.whisper.hf_token,
            hf_endpoint=hf_endpoint,
        )

        # Log which backend will be used
        if transcriber.use_whisperx:
            print_info("Using WhisperX for improved timestamp accuracy")

        # Check and download Whisper model if needed (separate progress bar)
        if not transcriber.is_model_cached():
            from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn
            import logging

            model_size_mb = transcriber.get_model_size() / (1024 * 1024)
            console.print(f"\n[cyan]Downloading Whisper model: {cfg.whisper.model}[/cyan]")
            console.print(f"[dim]Model size: ~{model_size_mb:.0f}MB (one-time download)[/dim]\n")

            # Suppress logs during download to avoid interfering with progress bar
            hf_logger = logging.getLogger("huggingface_hub")
            sf_logger = logging.getLogger("subtitle_forge")
            original_hf_level = hf_logger.level
            original_sf_level = sf_logger.level
            hf_logger.setLevel(logging.ERROR)
            sf_logger.setLevel(logging.ERROR)

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=40),
                    TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                    DownloadColumn(),
                    console=console,
                    disable=progress_disabled(),
                ) as dl_progress:
                    dl_task = dl_progress.add_task("Downloading...", total=transcriber.get_model_size())
                    last_completed = 0

                    def update_whisper_download(downloaded: int, total: int):
                        nonlocal last_completed
                        # Only update completed, not total (avoid accumulation bug)
                        if downloaded > last_completed:
                            dl_progress.update(dl_task, completed=downloaded)
                            last_completed = downloaded

                    transcriber.ensure_model_downloaded(progress_callback=update_whisper_download)
            finally:
                # Restore log levels
                hf_logger.setLevel(original_hf_level)
                sf_logger.setLevel(original_sf_level)

            print_info("Whisper model downloaded successfully!\n")

        # Initialize translator
        # --save-debug-log implies saving failed log to debug directory
        effective_save_failed_log = save_failed_log or save_debug_log
        if debug_failed_log_path:
            failed_log_path = debug_failed_log_path
        elif save_failed_log:
            failed_log_path = str(output_dir / f"{video.stem}_translation_failures.json")
        else:
            failed_log_path = None

        translator = SubtitleTranslator(
            TranslationConfig(
                model=cfg.ollama.model,
                host=cfg.ollama.host,
                temperature=cfg.ollama.temperature,
                max_batch_size=cfg.ollama.max_batch_size,
                max_retries=cfg.ollama.max_retries,
                request_timeout=cfg.ollama.request_timeout,
                prompt_template=cfg.ollama.prompt_template,
                prompt_template_id=cfg.ollama.prompt_template_id,
                save_failed_log=effective_save_failed_log,
                failed_log_path=failed_log_path,
            )
        )

        # Check and download translation model if needed (separate progress bar)
        if not translator.check_model_available():
            print_warning(f"Translation model '{cfg.ollama.model}' not found")
            if typer.confirm("Download model now?", default=True):
                from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn

                console.print(f"\n[cyan]Downloading model: {cfg.ollama.model}[/cyan]")
                console.print("[dim]This may take a while for large models...[/dim]\n")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=40),
                    TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                    DownloadColumn(),
                    console=console,
                    disable=progress_disabled(),
                ) as dl_progress:
                    dl_task = dl_progress.add_task("Downloading...", total=None)

                    def update_download(dp):
                        if dp.total_bytes and dp.total_bytes > 0:
                            dl_progress.update(
                                dl_task,
                                total=dp.total_bytes,
                                completed=dp.completed_bytes or 0,
                                description=dp.status.replace("_", " ").capitalize(),
                            )

                    ready = translator.ensure_model_ready(progress_callback=update_download)

                # ensure_model_ready returns False on download failure —
                # printing success and only failing after transcription
                # finished wastes the whole GPU run.
                if not ready:
                    print_error(
                        f"Failed to download model '{cfg.ollama.model}'. "
                        "Run: subtitle-forge config pull-model"
                    )
                    raise typer.Exit(1)
                print_info("Model downloaded successfully!\n")
            else:
                print_error("Translation requires the configured model. Run: subtitle-forge config pull-model")
                raise typer.Exit(1)

        # ========== Phase 2: Main processing (single progress bar) ==========

        from contextlib import contextmanager
        from ..core.pipeline import PipelineHooks, run_pipeline

        with progress.track_video(video.name) as tracker:
            tracker.set_description(f"[1/4] Extracting audio: {video.name}")
            phase_3_started = [False]  # mutable so nested closures can flip it

            def _start_phase_3():
                """Run once on the first translate-or-skip event to keep the
                visual sequence (audio→transcribe→original→translation) the
                same as the pre-refactor flow."""
                if phase_3_started[0]:
                    return
                phase_3_started[0] = True
                tracker.set_description(f"[3/4] Translating: {video.name}")
                # Pause main progress bar for translation (avoid two bars)
                tracker.pause()
                print_translation_explainer()

            def hook_audio_extracted(_path):
                tracker.update("[1/4] Audio extraction complete")
                tracker.set_description(f"[2/4] Transcribing: {video.name}")

            def hook_transcribe_complete(_seg_count, _detected_lang):
                tracker.update("[2/4] Transcription complete")

            def hook_original_saved(path):
                print_info(f"Original subtitles saved: {path}")

            def hook_translation_skipped(lang):
                _start_phase_3()
                print_info(f"Skipping translation to {lang} (same as source)")

            @contextmanager
            def trans_progress_ctx(lang, total):
                _start_phase_3()
                lang_name = translator.LANGUAGE_NAMES.get(lang, lang)
                print_info(f"Translating to {lang_name}...")
                with TranslationProgressTracker(
                    total_segments=total,
                    # Use the model-clamped size the translator will actually use
                    # (7B/14B clamp below max_batch_size), so "batch x/y" is right.
                    batch_size=translator.effective_batch_size(),
                    target_lang=lang_name,
                ) as trans_progress:
                    yield trans_progress.update

            def hook_translation_saved(path, _label):
                print_info(f"Translated subtitles saved: {path}")

            result = run_pipeline(
                video, cfg,
                transcriber=transcriber,
                translator=translator,
                target_languages=target_lang,
                output_dir=output_dir,
                source_language=source_lang,
                keep_original=keep_original,
                bilingual=bilingual,
                timestamp_mode=timestamp_mode,
                split_sentences=split_sentences,
                post_process=post_process,
                vad_parameters=vad_params,
                hooks=PipelineHooks(
                    on_audio_extracted=hook_audio_extracted,
                    on_transcribe_complete=hook_transcribe_complete,
                    on_original_saved=hook_original_saved,
                    on_translation_skipped=hook_translation_skipped,
                    translation_progress_ctx=trans_progress_ctx,
                    on_translation_saved=hook_translation_saved,
                ),
            )

            # Only resume if we actually paused (in case nothing translatable)
            if phase_3_started[0]:
                tracker.resume()
            tracker.update("[3/4] Translation complete")

            # Cleanup — pipeline handles the audio scratch file itself.
            tracker.set_description("[4/4] Cleaning up")
            transcriber.unload_model()
            tracker.update("[4/4] Complete")

        print_success(
            f"Processing complete!\n"
            f"  Video: {video.name}\n"
            f"  Detected language: {result.detected_language} ({result.language_probability:.1%})\n"
            f"  Segments: {result.segment_count}\n"
            f"  Output directory: {output_dir}"
        )

    except typer.Exit:
        # typer.Exit subclasses RuntimeError — without this re-raise the
        # handler below would print an empty error panel for clean exits.
        raise
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(1)


@app.command()
def quickstart():
    """
    Interactive first-time setup wizard.

    Guides you through:
    - Checking system requirements (ffmpeg, GPU)
    - Verifying Ollama is running
    - Downloading the translation model

    Example:
        subtitle-forge quickstart
    """
    from ..utils.setup_wizard import run_setup_wizard

    run_setup_wizard()


@app.command()
def version():
    """Show version information."""
    from .. import __version__

    console.print(f"subtitle-forge version {__version__}")


def cli():
    """Entry point."""
    app()


if __name__ == "__main__":
    cli()
