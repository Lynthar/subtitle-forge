"""CLI main application."""

from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console

from .commands import transcribe, translate, batch, config
from ..models.config import AppConfig
from ..utils.logger import setup_logging

app = typer.Typer(
    name="subtitle-forge",
    help="Local video subtitle generation and translation tool",
    add_completion=True,
    no_args_is_help=True,
)

console = Console()

# Register subcommands
app.add_typer(transcribe.app, name="transcribe", help="Transcribe video to subtitles")
app.add_typer(translate.app, name="translate", help="Translate existing subtitles")
app.add_typer(batch.app, name="batch", help="Batch process multiple videos")
app.add_typer(config.app, name="config", help="Configuration management")

# Global config
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get current configuration."""
    global _config
    if _config is None:
        _config = AppConfig.load()
    return _config


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
    global _config

    # Load configuration
    if config_file:
        _config = AppConfig.load(config_file)
    else:
        _config = get_config()

    # Setup logging
    log_level = "DEBUG" if verbose else ("ERROR" if quiet else _config.log_level)
    setup_logging(log_level, str(log_file) if log_file else _config.log_file)

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
    keep_original: bool = typer.Option(
        True,
        "--keep-original/--no-keep-original",
        help="Keep original language subtitles",
    ),
    bilingual: bool = typer.Option(
        False,
        "--bilingual",
        help="Generate bilingual subtitles",
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
    from ..core.audio import AudioExtractor
    from ..core.transcriber import Transcriber
    from ..core.translator import SubtitleTranslator, TranslationConfig
    from ..core.subtitle import SubtitleProcessor
    from ..utils.progress import (
        SubtitleProgress,
        TranslationProgressTracker,
        print_success,
        print_error,
        print_info,
        print_warning,
        print_translation_explainer,
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
        # Re-setup logging with DEBUG level to the debug log file
        setup_logging("DEBUG", debug_log_path)

    # Build VAD parameters
    from ..core.transcriber import Transcriber as TranscriberClass
    vad_params = TranscriberClass.get_vad_parameters(
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

                    translator.ensure_model_ready(progress_callback=update_download)

                print_info("Model downloaded successfully!\n")
            else:
                print_error("Translation requires the configured model. Run: subtitle-forge config pull-model")
                raise typer.Exit(1)

        # ========== Phase 2: Main processing (single progress bar) ==========

        with progress.track_video(video.name) as tracker:
            # 1. Extract audio
            tracker.set_description(f"[1/4] Extracting audio: {video.name}")
            extractor = AudioExtractor()
            audio_path = extractor.extract(video)
            tracker.update("[1/4] Audio extraction complete")

            # 2. Transcribe
            tracker.set_description(f"[2/4] Transcribing: {video.name}")

            # Build timestamp config from settings
            timestamp_config = {
                "mode": timestamp_mode or cfg.timestamp.mode,
                "min_duration": cfg.timestamp.min_duration,
                "max_duration": cfg.timestamp.max_duration,
                "min_gap": cfg.timestamp.min_gap,
                "max_gap_warning": cfg.timestamp.max_gap_warning,
                "chars_per_second": cfg.timestamp.chars_per_second,
                "cjk_chars_per_second": cfg.timestamp.cjk_chars_per_second,
                "split_threshold": cfg.timestamp.split_threshold,
                "split_sentences": split_sentences if split_sentences is not None else cfg.timestamp.split_sentences,
            } if post_process and cfg.timestamp.enabled else None

            segments, info = transcriber.transcribe(
                audio_path,
                language=source_lang,
                beam_size=cfg.whisper.beam_size,
                vad_filter=cfg.whisper.vad_filter,
                vad_parameters=vad_params,
                post_process=post_process and cfg.timestamp.enabled,
                timestamp_config=timestamp_config,
            )
            detected_lang = info.language
            tracker.update("[2/4] Transcription complete")

            # 3. Save original subtitles
            subtitle_processor = SubtitleProcessor()
            if keep_original:
                original_srt = output_dir / f"{video.stem}.{detected_lang}.srt"
                subtitle_processor.save(segments, original_srt)
                print_info(f"Original subtitles saved: {original_srt}")

            # 3. Translate (models already prepared in Phase 1)
            tracker.set_description(f"[3/4] Translating: {video.name}")

            # Pause main progress bar for translation (to avoid two progress bars)
            tracker.pause()

            # Show explanation for first-time users
            print_translation_explainer()

            for lang in target_lang:
                if lang == detected_lang:
                    print_info(f"Skipping translation to {lang} (same as source)")
                    continue

                lang_name = translator.LANGUAGE_NAMES.get(lang, lang)
                print_info(f"Translating to {lang_name}...")

                # Use enhanced translation progress tracker
                with TranslationProgressTracker(
                    total_segments=len(segments),
                    batch_size=cfg.ollama.max_batch_size,
                    target_lang=lang_name,
                ) as trans_progress:
                    translated = translator.translate(
                        segments,
                        detected_lang,
                        lang,
                        progress_callback=trans_progress.update,
                    )

                if bilingual:
                    merged = subtitle_processor.merge_bilingual(segments, translated)
                    output_path = output_dir / f"{video.stem}.{detected_lang}-{lang}.srt"
                    subtitle_processor.save(merged, output_path)
                else:
                    output_path = output_dir / f"{video.stem}.{lang}.srt"
                    subtitle_processor.save(translated, output_path)

                print_info(f"Translated subtitles saved: {output_path}")

            # Resume main progress bar
            tracker.resume()
            tracker.update("[3/4] Translation complete")

            # 5. Cleanup
            tracker.set_description("[4/4] Cleaning up")
            audio_path.unlink(missing_ok=True)
            transcriber.unload_model()
            tracker.update("[4/4] Complete")

        print_success(
            f"Processing complete!\n"
            f"  Video: {video.name}\n"
            f"  Detected language: {detected_lang} ({info.language_probability:.1%})\n"
            f"  Segments: {len(segments)}\n"
            f"  Output directory: {output_dir}"
        )

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
