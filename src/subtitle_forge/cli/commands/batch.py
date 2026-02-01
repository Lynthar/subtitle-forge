"""Batch processing command."""

from pathlib import Path
from typing import Optional, List

import typer

app = typer.Typer(no_args_is_help=True)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}


def find_videos(path: Path, recursive: bool = False) -> List[Path]:
    """Find video files in directory."""
    videos = []

    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    elif path.is_dir():
        pattern = "**/*" if recursive else "*"
        for ext in VIDEO_EXTENSIONS:
            videos.extend(path.glob(f"{pattern}{ext}"))

    return sorted(videos)


@app.command("run")
@app.command(hidden=True)  # Default command
def batch_process(
    path: Path = typer.Argument(..., help="Directory or video file path", exists=True),
    target_lang: List[str] = typer.Option(
        ...,
        "--target-lang",
        "-t",
        help="Target language(s)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (default: same as video)",
    ),
    workers: int = typer.Option(
        2,
        "--workers",
        "-w",
        help="Number of concurrent workers",
        min=1,
        max=4,
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Recursively search for videos",
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
    file_list: Optional[Path] = typer.Option(
        None,
        "--file-list",
        help="File containing list of video paths",
    ),
):
    """
    Batch process multiple videos.

    Example:
        subtitle-forge batch ./videos/ --target-lang zh
        subtitle-forge batch ./videos/ -t zh -t ja --workers 2 --recursive
        subtitle-forge batch --file-list videos.txt -t zh
    """
    from ...core.audio import AudioExtractor
    from ...core.transcriber import Transcriber
    from ...core.translator import SubtitleTranslator, TranslationConfig
    from ...core.subtitle import SubtitleProcessor
    from ...core.queue import run_batch_sync
    from ...models.config import AppConfig
    from ...models.task import VideoTask
    from ...utils.progress import (
        SubtitleProgress,
        print_success,
        print_error,
        print_info,
        print_task_summary,
    )

    config = AppConfig.load()

    # Override config
    if whisper_model:
        config.whisper.model = whisper_model
    if ollama_model:
        config.ollama.model = ollama_model

    # Collect videos
    videos = []
    if file_list:
        with open(file_list, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    video_path = Path(line)
                    if video_path.exists():
                        videos.append(video_path)
                    else:
                        print_info(f"Skipping non-existent file: {line}")
    else:
        videos = find_videos(path, recursive)

    if not videos:
        print_error("No video files found")
        raise typer.Exit(1)

    print_info(f"Found {len(videos)} video(s) to process")

    # Create tasks
    tasks = [
        VideoTask(
            video_path=video,
            target_langs=list(target_lang),
            output_dir=output_dir or video.parent,
            options={
                "keep_original": keep_original,
            },
        )
        for video in videos
    ]

    # Initialize components (shared across workers for efficiency warning)
    extractor = AudioExtractor()
    transcriber = Transcriber(
        model_name=config.whisper.model,
        device=config.whisper.device,
        compute_type=config.whisper.compute_type,
        use_whisperx=config.whisper.use_whisperx,
        whisperx_align=config.whisper.whisperx_align,
        hf_token=config.whisper.hf_token,
        hf_endpoint=config.whisper.hf_endpoint,
    )

    # Build timestamp config
    timestamp_config = {
        "min_duration": config.timestamp.min_duration,
        "max_duration": config.timestamp.max_duration,
        "min_gap": config.timestamp.min_gap,
        "max_gap_warning": config.timestamp.max_gap_warning,
        "chars_per_second": config.timestamp.chars_per_second,
    } if config.timestamp.enabled else None
    translator = SubtitleTranslator(
        TranslationConfig(
            model=config.ollama.model,
            host=config.ollama.host,
            temperature=config.ollama.temperature,
            max_batch_size=config.ollama.max_batch_size,
        )
    )
    subtitle_processor = SubtitleProcessor()

    def process_task(task: VideoTask) -> None:
        """Process a single video task."""
        # Extract audio
        audio_path = extractor.extract(task.video_path)

        try:
            # Transcribe
            segments, info = transcriber.transcribe(
                audio_path,
                beam_size=config.whisper.beam_size,
                vad_filter=config.whisper.vad_filter,
                post_process=config.timestamp.enabled,
                timestamp_config=timestamp_config,
            )
            task.source_lang = info.language

            # Save original
            if task.options.get("keep_original", True):
                original_srt = task.output_dir / f"{task.video_path.stem}.{info.language}.srt"
                subtitle_processor.save(segments, original_srt)
                task.original_srt = original_srt

            # Translate to each target language
            for lang in task.target_langs:
                if lang == info.language:
                    continue

                translated = translator.translate(segments, info.language, lang)
                output_path = task.output_dir / f"{task.video_path.stem}.{lang}.srt"
                subtitle_processor.save(translated, output_path)
                task.translated_srts[lang] = output_path

        finally:
            # Cleanup audio
            audio_path.unlink(missing_ok=True)

    # Progress tracking
    progress = SubtitleProgress()

    with progress.track_batch(len(tasks)) as tracker:

        def on_start(task: VideoTask):
            tracker.start_video(task.video_path.name)

        def on_complete(task: VideoTask):
            tracker.complete_video()

        def on_error(task: VideoTask, error: Exception):
            tracker.complete_video()

        # Run batch processing
        results = run_batch_sync(
            tasks,
            process_task,
            max_workers=workers,
            on_task_start=on_start,
            on_task_complete=on_complete,
            on_task_error=on_error,
        )

    # Cleanup
    transcriber.unload_model()

    # Print summary
    print_task_summary(results)

    completed = sum(1 for t in results if t.status.value == "completed")
    failed = sum(1 for t in results if t.status.value == "failed")

    if failed > 0:
        print_error(f"{failed} task(s) failed")
        raise typer.Exit(1)
    else:
        print_success(f"All {completed} task(s) completed successfully")
