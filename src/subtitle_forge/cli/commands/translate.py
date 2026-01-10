"""Translate command."""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(no_args_is_help=True)


@app.command("run")
@app.command(hidden=True)  # Default command
def translate_subtitle(
    subtitle: Path = typer.Argument(..., help="Subtitle file path (SRT)", exists=True),
    target_lang: str = typer.Option(
        ...,
        "--target-lang",
        "-t",
        help="Target language code",
    ),
    source_lang: Optional[str] = typer.Option(
        None,
        "--source-lang",
        "-s",
        help="Source language code",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output SRT file path",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Ollama model name",
    ),
    bilingual: bool = typer.Option(
        False,
        "--bilingual",
        help="Generate bilingual subtitles",
    ),
):
    """
    Translate existing subtitle file.

    Example:
        subtitle-forge translate video.en.srt --target-lang zh
        subtitle-forge translate video.srt -s en -t zh --bilingual
    """
    from ...core.translator import SubtitleTranslator, TranslationConfig
    from ...core.subtitle import SubtitleProcessor
    from ...models.config import AppConfig
    from ...utils.progress import SubtitleProgress, print_success, print_error, print_info

    config = AppConfig.load()

    # Try to detect source language from filename if not specified
    if source_lang is None:
        # Try pattern: name.lang.srt
        parts = subtitle.stem.split(".")
        if len(parts) >= 2 and len(parts[-1]) in (2, 5):  # 2 for 'en', 5 for 'zh-TW'
            source_lang = parts[-1]
            print_info(f"Detected source language from filename: {source_lang}")
        else:
            print_error(
                "Cannot detect source language. Please specify with --source-lang"
            )
            raise typer.Exit(1)

    # Output path
    if output is None:
        stem = subtitle.stem
        if stem.endswith(f".{source_lang}"):
            stem = stem[: -len(f".{source_lang}")]

        if bilingual:
            output = subtitle.parent / f"{stem}.{source_lang}-{target_lang}.srt"
        else:
            output = subtitle.parent / f"{stem}.{target_lang}.srt"

    progress = SubtitleProgress()

    try:
        with progress.track_video(subtitle.name, total_steps=2) as tracker:
            # 1. Load subtitles
            tracker.set_description("Loading subtitles...")
            processor = SubtitleProcessor()
            segments = processor.load(subtitle)
            tracker.update("Subtitles loaded")

            # 2. Translate
            tracker.set_description("Translating...")
            translator = SubtitleTranslator(
                TranslationConfig(
                    model=model or config.ollama.model,
                    host=config.ollama.host,
                    temperature=config.ollama.temperature,
                    max_batch_size=config.ollama.max_batch_size,
                )
            )

            translated = translator.translate(segments, source_lang, target_lang)

            # Save
            if bilingual:
                merged = processor.merge_bilingual(segments, translated)
                processor.save(merged, output)
            else:
                processor.save(translated, output)

            tracker.update("Translation complete")

        print_success(
            f"Translation complete!\n"
            f"  Source: {subtitle.name} ({source_lang})\n"
            f"  Target: {target_lang}\n"
            f"  Segments: {len(translated)}\n"
            f"  Output: {output}"
        )

    except Exception as e:
        print_error(str(e))
        raise typer.Exit(1)
