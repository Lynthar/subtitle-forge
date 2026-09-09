"""Translate command."""

import re
from pathlib import Path
from typing import Optional

import typer

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
    from ...core.translator import SubtitleTranslator
    from ...core.subtitle import SubtitleProcessor, validate_language_codes
    from ...utils.progress import SubtitleProgress, print_success, print_error, print_info
    from ..app import get_config

    # get_config() (not a bare AppConfig.load()) so the root --config flag
    # actually reaches this command.
    config = get_config()

    try:
        validate_language_codes([target_lang])
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Try to detect source language from filename if not specified
    if source_lang is None:
        # Try pattern: name.lang.srt — validate the *shape* of the last segment
        # ("en", "zh", "yue", "zh-TW") so a resolution/codec tag like the "1080p"
        # in "video.1080p.srt" isn't mistaken for a source language.
        parts = subtitle.stem.split(".")
        if len(parts) >= 2 and re.fullmatch(r"[a-z]{2,3}(-[A-Z]{2})?", parts[-1]):
            source_lang = parts[-1]
            print_info(f"Detected source language from filename: {source_lang}")
        else:
            print_error(
                "Cannot detect source language. Please specify with --source-lang"
            )
            raise typer.Exit(1)

    # Translating a file into its own language is a no-op — and with the
    # default output naming it would overwrite the input file with itself.
    if target_lang == source_lang:
        print_error(
            f"Target language '{target_lang}' is the same as the source — nothing to translate."
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

    if output.resolve() == subtitle.resolve():
        print_error(f"Output path equals the input file ({subtitle}) — refusing to overwrite it.")
        raise typer.Exit(1)

    progress = SubtitleProgress()

    try:
        with progress.track_video(subtitle.name, total_steps=2) as tracker:
            # 1. Load subtitles
            tracker.set_description("Loading subtitles...")
            processor = SubtitleProcessor(encoding=config.output.encoding)
            segments = processor.load(subtitle)
            tracker.update("Subtitles loaded")

            # 2. Translate
            tracker.set_description("Translating...")
            translator = SubtitleTranslator.from_config(
                config.ollama, model=model or config.ollama.model
            )

            translated = translator.translate(segments, source_lang, target_lang)

            # Save
            if bilingual:
                merged = processor.merge_bilingual(
                    segments, translated, original_on_top=config.output.original_on_top
                )
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
