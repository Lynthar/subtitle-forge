"""Transcribe command."""

from pathlib import Path
from typing import Optional

import typer

def transcribe_video(
    video: Path = typer.Argument(..., help="Video file path", exists=True),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output SRT file path",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Source language (auto-detect if not specified)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Whisper model name",
    ),
    vad_filter: bool = typer.Option(
        True,
        "--vad-filter/--no-vad-filter",
        help="Enable VAD filtering",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Batch size (uses BatchedInferencePipeline)",
    ),
    auto_model: bool = typer.Option(
        False,
        "--auto-model",
        help="Auto-select model based on GPU VRAM",
    ),
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
    save_debug_log: bool = typer.Option(
        False,
        "--save-debug-log",
        help="Save detailed debug logs (creates {video}_debug/ folder)",
    ),
):
    """
    Transcribe video audio to subtitles (no translation).

    Example:
        subtitle-forge transcribe video.mp4
        subtitle-forge transcribe video.mp4 --language en --model large-v3
    """
    from ...core.pipeline import PipelineHooks, build_vad_parameters, run_pipeline
    from ...core.transcriber import Transcriber
    from ...utils.progress import (
        SubtitleProgress,
        print_success,
        print_error,
        print_info,
        progress_disabled,
    )
    from ...utils.logger import setup_logging
    from ..app import get_config

    from rich.console import Console

    # get_config() (not a bare AppConfig.load()) so the root --config flag
    # actually reaches this command.
    config = get_config()
    progress = SubtitleProgress()
    console = Console()

    # Handle --save-debug-log option
    if save_debug_log:
        output_dir = video.parent
        debug_dir = output_dir / f"{video.stem}_debug"
        debug_dir.mkdir(exist_ok=True)
        debug_log_path = str(debug_dir / "run.log")
        # console_level="INFO" keeps third-party DEBUG stack traces out of the
        # terminal while the file still captures everything (same as process).
        setup_logging("DEBUG", debug_log_path, console_level="INFO")

    # Model selection
    if auto_model:
        model_name = Transcriber.select_optimal_model()
        print_info(f"Auto-selected model: {model_name}")
    else:
        model_name = model or config.whisper.model

    try:
        # ========== Phase 1: Prepare model (outside main progress bar) ==========

        # Determine WhisperX usage
        whisperx_enabled = use_whisperx if use_whisperx is not None else config.whisper.use_whisperx

        transcriber = Transcriber.from_config(
            config.whisper,
            model_name=model_name,
            use_whisperx=whisperx_enabled,
        )

        # Log which backend will be used
        if transcriber.use_whisperx:
            print_info("Using WhisperX for improved timestamp accuracy")

        # Check and download Whisper model if needed (separate progress bar)
        if not transcriber.is_model_cached():
            from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn
            import logging

            model_size_mb = transcriber.get_model_size() / (1024 * 1024)
            console.print(f"\n[cyan]Downloading Whisper model: {model_name}[/cyan]")
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

        # ========== Phase 2: Main processing (single progress bar) ==========

        # CLI overrides go onto the config, which is what run_pipeline reads.
        config.whisper.vad_filter = vad_filter
        if batch_size is not None:
            config.whisper.batch_size = batch_size

        with progress.track_video(video.name, total_steps=3) as tracker:
            tracker.set_description("Extracting audio...")

            def hook_audio_extracted(_audio_path):
                tracker.update("Audio extraction complete")
                tracker.set_description("Transcribing...")

            def hook_transcribe_complete(_segment_count, _detected_lang):
                tracker.update("Transcription complete")
                tracker.set_description("Saving subtitles...")

            def hook_original_saved(_path):
                tracker.update("Save complete")

            # No target languages, so no translator: this command never
            # translates. --output names the one file it does write.
            result = run_pipeline(
                video,
                config,
                transcriber=transcriber,
                target_languages=[],
                output_dir=output.parent if output else video.parent,
                source_language=language,
                keep_original=True,
                original_output_path=output,
                timestamp_mode=timestamp_mode,
                split_sentences=split_sentences,
                post_process=post_process,
                vad_parameters=build_vad_parameters(config),
                hooks=PipelineHooks(
                    on_audio_extracted=hook_audio_extracted,
                    on_transcribe_complete=hook_transcribe_complete,
                    on_original_saved=hook_original_saved,
                ),
            )

            transcriber.unload_model()

        print_success(
            f"Transcription complete!\n"
            f"  Detected language: {result.detected_language} "
            f"({result.language_probability:.1%})\n"
            f"  Segments: {result.segment_count}\n"
            f"  Output: {result.outputs[0].path}"
        )

    except Exception as e:
        print_error(str(e))
        raise typer.Exit(1)
