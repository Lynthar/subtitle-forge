"""Configuration management command."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ...models.config import AppConfig
from ...utils.gpu import get_gpu_info, check_cuda_available
from ...utils.progress import print_success, print_error, print_info, print_warning

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def show():
    """Show current configuration."""
    config = AppConfig.load()

    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Whisper settings
    table.add_row("[bold]Whisper[/bold]", "")
    table.add_row("  Model", config.whisper.model)
    table.add_row("  Device", config.whisper.device)
    table.add_row("  Compute Type", config.whisper.compute_type)
    table.add_row("  Beam Size", str(config.whisper.beam_size))
    table.add_row("  VAD Filter", str(config.whisper.vad_filter))

    # Ollama settings
    table.add_row("[bold]Ollama[/bold]", "")
    table.add_row("  Model", config.ollama.model)
    table.add_row("  Host", config.ollama.host)
    table.add_row("  Temperature", str(config.ollama.temperature))
    table.add_row("  Max Batch Size", str(config.ollama.max_batch_size))

    # Output settings
    table.add_row("[bold]Output[/bold]", "")
    table.add_row("  Encoding", config.output.encoding)
    table.add_row("  Keep Original", str(config.output.keep_original))
    table.add_row("  Bilingual", str(config.output.bilingual))

    # General settings
    table.add_row("[bold]General[/bold]", "")
    table.add_row("  Max Workers", str(config.max_workers))
    table.add_row("  Log Level", config.log_level)

    console.print(table)
    console.print(f"\nConfig file: {AppConfig.get_config_path()}")


@app.command()
def set(
    key: str = typer.Argument(..., help="Setting key (e.g., whisper.model)"),
    value: str = typer.Argument(..., help="Setting value"),
):
    """
    Set a configuration value.

    Example:
        subtitle-forge config set whisper.model large-v3
        subtitle-forge config set ollama.model qwen2.5:32b
        subtitle-forge config set max_workers 4
    """
    config = AppConfig.load()

    parts = key.split(".")
    if len(parts) == 1:
        # Top-level setting
        if key == "max_workers":
            config.max_workers = int(value)
        elif key == "log_level":
            config.log_level = value
        else:
            print_error(f"Unknown setting: {key}")
            raise typer.Exit(1)
    elif len(parts) == 2:
        section, setting = parts
        if section == "whisper":
            if setting == "model":
                config.whisper.model = value
            elif setting == "device":
                config.whisper.device = value
            elif setting == "compute_type":
                config.whisper.compute_type = value
            elif setting == "beam_size":
                config.whisper.beam_size = int(value)
            elif setting == "vad_filter":
                config.whisper.vad_filter = value.lower() in ("true", "1", "yes")
            else:
                print_error(f"Unknown whisper setting: {setting}")
                raise typer.Exit(1)
        elif section == "ollama":
            if setting == "model":
                config.ollama.model = value
            elif setting == "host":
                config.ollama.host = value
            elif setting == "temperature":
                config.ollama.temperature = float(value)
            elif setting == "max_batch_size":
                config.ollama.max_batch_size = int(value)
            else:
                print_error(f"Unknown ollama setting: {setting}")
                raise typer.Exit(1)
        elif section == "output":
            if setting == "encoding":
                config.output.encoding = value
            elif setting == "keep_original":
                config.output.keep_original = value.lower() in ("true", "1", "yes")
            elif setting == "bilingual":
                config.output.bilingual = value.lower() in ("true", "1", "yes")
            else:
                print_error(f"Unknown output setting: {setting}")
                raise typer.Exit(1)
        else:
            print_error(f"Unknown section: {section}")
            raise typer.Exit(1)
    else:
        print_error(f"Invalid key format: {key}")
        raise typer.Exit(1)

    config.save()
    print_success(f"Set {key} = {value}")


@app.command()
def reset():
    """Reset configuration to defaults."""
    config = AppConfig()
    config.save()
    print_success("Configuration reset to defaults")


@app.command()
def check():
    """Check system status and dependencies."""
    console.print(Panel("System Check", style="bold blue"))

    # Check CUDA
    if check_cuda_available():
        gpu_info = get_gpu_info()
        console.print("[green]CUDA:[/green] Available")
        console.print(f"  GPU: {gpu_info.get('device_name', 'Unknown')}")
        console.print(f"  VRAM: {gpu_info.get('total_vram_mb', 0)}MB total")
        console.print(f"  Available: {gpu_info.get('available_vram_mb', 0)}MB")
    else:
        console.print("[yellow]CUDA:[/yellow] Not available (will use CPU)")

    # Check ffmpeg
    import shutil

    if shutil.which("ffmpeg"):
        console.print("[green]ffmpeg:[/green] Found")
    else:
        console.print("[red]ffmpeg:[/red] Not found - please install ffmpeg")

    # Check Ollama
    try:
        from ...core.translator import SubtitleTranslator, TranslationConfig

        config = AppConfig.load()
        translator = SubtitleTranslator(
            TranslationConfig(
                model=config.ollama.model,
                host=config.ollama.host,
            )
        )

        if translator.check_model_available():
            console.print(f"[green]Ollama:[/green] Connected, model {config.ollama.model} available")
        else:
            console.print(
                f"[yellow]Ollama:[/yellow] Connected, but model {config.ollama.model} not found"
            )
            console.print(f"  Run: ollama pull {config.ollama.model}")
    except Exception as e:
        console.print(f"[red]Ollama:[/red] Cannot connect ({e})")
        console.print("  Make sure Ollama is running: ollama serve")

    # Recommended model based on VRAM
    if check_cuda_available():
        from ...core.transcriber import Transcriber

        recommended = Transcriber.select_optimal_model()
        console.print(f"\nRecommended Whisper model: [cyan]{recommended}[/cyan]")


@app.command()
def export(
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file path",
    ),
):
    """Export configuration to file."""
    config = AppConfig.load()
    config.save(output)
    print_success(f"Configuration exported to: {output}")


@app.command("import")
def import_config(
    input_file: Path = typer.Argument(..., help="Configuration file to import", exists=True),
):
    """Import configuration from file."""
    config = AppConfig.load(input_file)
    config.save()
    print_success(f"Configuration imported from: {input_file}")
