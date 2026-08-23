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
    from ...core.pipeline import build_timestamp_config, build_vad_parameters
    from ...core.transcriber import Transcriber
    from ...core.translator import SubtitleTranslator, TranslationConfig
    from ...core.subtitle import SubtitleProcessor, normalize_target_languages
    from ...core.queue import run_batch_sync
    from ...models.task import VideoTask
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
        # Validates for filename safety AND drops duplicate -t values (batch
        # doesn't go through run_pipeline, so it normalizes here itself).
        target_lang = normalize_target_languages(target_lang)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # --workers falls back to config.max_workers — before this the config
    # field was displayed by `config show` but never read anywhere.
    if workers is None:
        workers = min(max(1, config.max_workers), 4)

    # Override config
    if whisper_model:
        config.whisper.model = whisper_model
    if ollama_model:
        config.ollama.model = ollama_model
    keep_original = keep_original if keep_original is not None else config.output.keep_original

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

    # Initialize components (shared across workers for efficiency warning)
    extractor = AudioExtractor()
    transcriber = Transcriber(
        model_name=config.whisper.model,
        device=config.whisper.device,
        compute_type=config.whisper.compute_type,
        download_root=config.whisper.download_root,
        use_whisperx=config.whisper.use_whisperx,
        whisperx_align=config.whisper.whisperx_align,
        hf_token=config.whisper.hf_token,
        hf_endpoint=config.whisper.hf_endpoint,
    )

    timestamp_config = build_timestamp_config(config)
    vad_params = build_vad_parameters(config)
    translation_config = TranslationConfig(
        model=config.ollama.model,
        host=config.ollama.host,
        temperature=config.ollama.temperature,
        max_batch_size=config.ollama.max_batch_size,
        max_retries=config.ollama.max_retries,
        request_timeout=config.ollama.request_timeout,
        prompt_template=config.ollama.prompt_template,
        prompt_template_id=config.ollama.prompt_template_id,
    )
    subtitle_processor = SubtitleProcessor(encoding=config.output.encoding)

    def process_task(task: VideoTask) -> None:
        """Process a single video task."""
        # Fresh translator per task: it carries per-run failure tracking that
        # workers sharing one instance would clear out from under each other.
        # The Transcriber is shared — its model load is heavy, and a lock serializes it.
        translator = SubtitleTranslator(translation_config)

        # Extract audio
        audio_path = extractor.extract(task.video_path)

        try:
            # Transcribe
            segments, info = transcriber.transcribe(
                audio_path,
                beam_size=config.whisper.beam_size,
                vad_filter=config.whisper.vad_filter,
                batch_size=config.whisper.batch_size,
                vad_parameters=vad_params,
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
