"""Batch processing command."""

from pathlib import Path
from typing import Optional, List

import typer

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}


def find_videos(path: Path, recursive: bool = False) -> List[Path]:
    """Find video files in directory."""
    videos = []

    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    elif path.is_dir():
        # Filter by lowercased suffix rather than globbing each extension: on
        # case-sensitive filesystems (Linux) `path.glob("*.mp4")` misses ".MP4",
        # so uppercase-extension files were silently skipped.
        entries = path.rglob("*") if recursive else path.glob("*")
        videos = [
            p for p in entries if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]

    return sorted(videos)


def batch_process(
    path: Optional[Path] = typer.Argument(
        None,
        help="Directory or video file path (omit when using --file-list)",
        exists=True,
    ),
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
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of concurrent workers (default: config max_workers)",
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
        help="Save detailed debug logs and failure reports (creates {video}_debug/ folders)",
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
    from ...core.pipeline import build_timestamp_config, build_vad_parameters, run_pipeline
    from ...core.transcriber import Transcriber
    from ...core.translator import SubtitleTranslator
    from ...core.subtitle import normalize_target_languages
    from ...core.queue import run_batch_sync
    from ...models.task import VideoTask
    from ...utils.logger import setup_logging
    from ...utils.progress import (
        SubtitleProgress,
        print_success,
        print_error,
        print_info,
        print_task_summary,
    )
    from ..app import get_config

    # get_config() (not a bare AppConfig.load()) so the root --config flag
    # actually reaches this command.
    config = get_config()

    try:
        # run_pipeline normalizes again per task; doing it once here turns an
        # unsafe -t value into one error before any video starts.
        target_lang = normalize_target_languages(target_lang)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # --workers falls back to config.max_workers — before this the config
    # field was displayed by `config show` but never read anywhere.
    if workers is None:
        workers = min(max(1, config.max_workers), 4)

    # Logging is process-wide, so concurrent tasks would write into each
    # other's run.log.
    if save_debug_log and workers > 1:
        print_info("--save-debug-log processes one video at a time")
        workers = 1

    # Override config
    if whisper_model:
        config.whisper.model = whisper_model
    if ollama_model:
        config.ollama.model = ollama_model
    keep_original = keep_original if keep_original is not None else config.output.keep_original
    bilingual = bilingual if bilingual is not None else config.output.bilingual

    # Fail once here rather than once per task on a typo'd --timestamp-mode.
    try:
        build_timestamp_config(
            config,
            mode_override=timestamp_mode,
            split_sentences_override=split_sentences,
        )
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Collect videos
    videos = []
    if file_list:
        with open(file_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    video_path = Path(line)
                    if video_path.exists():
                        videos.append(video_path)
                    else:
                        print_info(f"Skipping non-existent file: {line}")
    elif path is not None:
        videos = find_videos(path, recursive)
    else:
        print_error("Provide a directory/video PATH or --file-list")
        raise typer.Exit(1)

    # Drop exact duplicates (a path listed twice in --file-list), keep order.
    videos = list(dict.fromkeys(videos))

    if not videos:
        print_error("No video files found")
        raise typer.Exit(1)

    # Refuse silent overwrites before any work starts: two inputs writing the same
    # {output_dir}/{stem}.{lang}.srt — same-named episodes funneled into one --output-dir, or
    # movie.mp4 beside movie.mkv — would clobber each other mid-batch.
    targets: dict = {}
    for video in videos:
        key = ((output_dir or video.parent), video.stem)
        if key in targets:
            print_error(
                f"Output collision: '{targets[key]}' and '{video}' would both write "
                f"{key[0] / (video.stem + '.<lang>.srt')}\n"
                "Rename one, or drop --output-dir so outputs stay next to their videos."
            )
            raise typer.Exit(1)
        targets[key] = video

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

    # One Transcriber shared across workers — its model load is heavy, and a
    # lock inside serializes the actual inference.
    transcriber = Transcriber.from_config(config.whisper)
    vad_params = build_vad_parameters(config)

    def process_task(task: VideoTask) -> None:
        """Process a single video task."""
        failed_log_path = None
        if save_debug_log:
            debug_dir = task.output_dir / f"{task.video_path.stem}_debug"
            debug_dir.mkdir(exist_ok=True)
            failed_log_path = str(debug_dir / "translation_failures.json")
            # console_level="INFO" keeps third-party DEBUG stack traces out of
            # the terminal while the file still captures everything.
            setup_logging("DEBUG", str(debug_dir / "run.log"), console_level="INFO")

        # Fresh translator per task: it carries per-run failure tracking that
        # workers sharing one instance would clear out from under each other.
        translator = SubtitleTranslator.from_config(
            config.ollama,
            save_failed_log=save_debug_log,
            failed_log_path=failed_log_path,
        )

        result = run_pipeline(
            task.video_path,
            config,
            transcriber=transcriber,
            translator=translator,
            target_languages=task.target_langs,
            output_dir=task.output_dir,
            keep_original=task.options.get("keep_original", True),
            bilingual=bilingual,
            timestamp_mode=timestamp_mode,
            split_sentences=split_sentences,
            vad_parameters=vad_params,
        )

        task.source_lang = result.detected_language
        for produced in result.outputs:
            if produced.language == result.detected_language:
                task.original_srt = produced.path
            else:
                task.translated_srts[produced.language] = produced.path

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
