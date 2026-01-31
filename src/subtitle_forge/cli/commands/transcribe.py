"""Transcribe command."""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(no_args_is_help=True)


@app.command("run")
@app.command(hidden=True)  # Default command
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
):
    """
    Transcribe video audio to subtitles (no translation).

    Example:
        subtitle-forge transcribe video.mp4
        subtitle-forge transcribe video.mp4 --language en --model large-v3
    """
    from ...core.audio import AudioExtractor
    from ...core.transcriber import Transcriber
    from ...core.subtitle import SubtitleProcessor
    from ...models.config import AppConfig
    from ...utils.progress import SubtitleProgress, print_success, print_error, print_info

    from rich.console import Console

    config = AppConfig.load()
    progress = SubtitleProgress()
    console = Console()

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

        transcriber = Transcriber(
            model_name=model_name,
            device=config.whisper.device,
            compute_type=config.whisper.compute_type,
            use_whisperx=whisperx_enabled,
            whisperx_align=config.whisper.whisperx_align,
            hf_token=config.whisper.hf_token,
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

        with progress.track_video(video.name, total_steps=3) as tracker:
            # 1. Extract audio
            tracker.set_description("Extracting audio...")
            extractor = AudioExtractor()
            audio_path = extractor.extract(video)
            tracker.update("Audio extraction complete")

            # 2. Transcribe (model already prepared)
            tracker.set_description("Transcribing...")

            # Build timestamp config from settings
            timestamp_config = {
                "min_duration": config.timestamp.min_duration,
                "max_duration": config.timestamp.max_duration,
                "min_gap": config.timestamp.min_gap,
                "max_gap_warning": config.timestamp.max_gap_warning,
                "chars_per_second": config.timestamp.chars_per_second,
            } if post_process and config.timestamp.enabled else None

            segments, info = transcriber.transcribe(
                audio_path,
                language=language,
                beam_size=config.whisper.beam_size,
                vad_filter=vad_filter,
                batch_size=batch_size,
                post_process=post_process and config.timestamp.enabled,
                timestamp_config=timestamp_config,
            )
            tracker.update("Transcription complete")

            # 3. Save subtitles
            tracker.set_description("Saving subtitles...")

            # Output path
            if output is None:
                output = video.with_suffix("").with_suffix(f".{info.language}.srt")

            processor = SubtitleProcessor()
            processor.save(segments, output)
            tracker.update("Save complete")

            # Cleanup
            audio_path.unlink(missing_ok=True)
            transcriber.unload_model()

        print_success(
            f"Transcription complete!\n"
            f"  Detected language: {info.language} ({info.language_probability:.1%})\n"
            f"  Segments: {len(segments)}\n"
            f"  Output: {output}"
        )

    except Exception as e:
        print_error(str(e))
        raise typer.Exit(1)
