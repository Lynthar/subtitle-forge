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

    config = AppConfig.load()
    progress = SubtitleProgress()

    # Model selection
    if auto_model:
        model_name = Transcriber.select_optimal_model()
        print_info(f"Auto-selected model: {model_name}")
    else:
        model_name = model or config.whisper.model

    try:
        with progress.track_video(video.name, total_steps=3) as tracker:
            # 1. Extract audio
            tracker.set_description("Extracting audio...")
            extractor = AudioExtractor()
            audio_path = extractor.extract(video)
            tracker.update("Audio extraction complete")

            # 2. Transcribe
            tracker.set_description("Transcribing...")
            transcriber = Transcriber(
                model_name=model_name,
                device=config.whisper.device,
                compute_type=config.whisper.compute_type,
            )
            segments, info = transcriber.transcribe(
                audio_path,
                language=language,
                beam_size=config.whisper.beam_size,
                vad_filter=vad_filter,
                batch_size=batch_size,
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
