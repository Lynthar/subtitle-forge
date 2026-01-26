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
def check(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed diagnostic information",
    ),
):
    """
    Check system status and dependencies.

    Example:
        subtitle-forge config check
        subtitle-forge config check --verbose
    """
    import shutil

    console.print(Panel("System Diagnostics", style="bold blue"))

    issues = []
    config = AppConfig.load()

    # Check CUDA
    console.print("\n[bold]GPU Status:[/bold]")
    if check_cuda_available():
        gpu_info = get_gpu_info()
        console.print(f"  [green]CUDA:[/green] Available")
        console.print(f"    GPU: {gpu_info.get('device_name', 'Unknown')}")
        console.print(f"    VRAM Total: {gpu_info.get('total_vram_mb', 0)}MB")
        console.print(f"    VRAM Available: {gpu_info.get('available_vram_mb', 0)}MB")

        if verbose:
            try:
                import torch
                console.print(f"    PyTorch Version: {torch.__version__}")
                console.print(f"    CUDA Version: {torch.version.cuda}")
            except ImportError:
                pass
    else:
        console.print("  [yellow]CUDA:[/yellow] Not available")
        console.print("    Transcription will use CPU (slower)")

        if verbose:
            console.print("    [dim]To enable GPU acceleration, install CUDA and PyTorch with CUDA support[/dim]")

    # Check ffmpeg
    console.print("\n[bold]FFmpeg Status:[/bold]")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        console.print(f"  [green]ffmpeg:[/green] Found")
        if verbose:
            console.print(f"    Path: {ffmpeg_path}")
            import subprocess
            try:
                result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
                version_line = result.stdout.split('\n')[0] if result.stdout else "Unknown"
                console.print(f"    Version: {version_line}")
            except Exception:
                pass
    else:
        console.print("  [red]ffmpeg:[/red] Not found")
        console.print("    [yellow]Install ffmpeg:[/yellow]")
        console.print("      macOS:   brew install ffmpeg")
        console.print("      Ubuntu:  sudo apt install ffmpeg")
        console.print("      Windows: choco install ffmpeg")
        issues.append("ffmpeg not installed")

    # Check Ollama
    console.print("\n[bold]Ollama Status:[/bold]")
    try:
        from ...core.model_manager import OllamaModelManager

        manager = OllamaModelManager(host=config.ollama.host)

        if manager.check_connection():
            console.print(f"  [green]Connection:[/green] OK")
            console.print(f"    Host: {config.ollama.host}")

            if verbose:
                available_models = manager.list_models()
                console.print(f"    Available models: {', '.join(available_models) if available_models else 'None'}")

            # Check configured model
            if manager.is_model_available(config.ollama.model):
                console.print(f"  [green]Model:[/green] {config.ollama.model} (ready)")
            else:
                console.print(f"  [yellow]Model:[/yellow] {config.ollama.model} (not downloaded)")
                console.print(f"    [yellow]Download with:[/yellow] subtitle-forge config pull-model")
                issues.append(f"Translation model '{config.ollama.model}' not downloaded")
        else:
            console.print("  [red]Connection:[/red] Failed")
            console.print(f"    Cannot connect to {config.ollama.host}")
            console.print("    [yellow]Start Ollama with:[/yellow] ollama serve")
            issues.append("Cannot connect to Ollama")

    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(f"    [dim]{traceback.format_exc()}[/dim]")
        issues.append(f"Ollama error: {e}")

    # Configuration summary
    if verbose:
        console.print("\n[bold]Current Configuration:[/bold]")
        console.print(f"  Whisper Model: {config.whisper.model}")
        console.print(f"  Whisper Device: {config.whisper.device}")
        console.print(f"  Ollama Model: {config.ollama.model}")
        console.print(f"  Max Workers: {config.max_workers}")
        console.print(f"  Config Path: {AppConfig.get_config_path()}")

    # Check Whisper model
    console.print("\n[bold]Whisper Status:[/bold]")
    try:
        from ...core.transcriber import Transcriber

        transcriber = Transcriber(model_name=config.whisper.model)
        if transcriber.is_model_cached():
            console.print(f"  [green]Model:[/green] {config.whisper.model} (ready)")
        else:
            model_size_mb = transcriber.get_model_size() / (1024 * 1024)
            console.print(f"  [yellow]Model:[/yellow] {config.whisper.model} (not downloaded, ~{model_size_mb:.0f}MB)")
            console.print(f"    [dim]Will be downloaded automatically on first use[/dim]")
            issues.append(f"Whisper model '{config.whisper.model}' not downloaded")

    except Exception as e:
        console.print(f"  [red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(f"    [dim]{traceback.format_exc()}[/dim]")

    # Recommended model based on VRAM
    if check_cuda_available():
        from ...core.transcriber import Transcriber

        recommended = Transcriber.select_optimal_model()
        console.print(f"\n[bold]Recommendation:[/bold]")
        console.print(f"  Whisper model: [cyan]{recommended}[/cyan] (based on available VRAM)")

    # Summary
    console.print()
    if issues:
        console.print(Panel(
            "[yellow]Issues Found:[/yellow]\n\n" +
            "\n".join(f"  - {issue}" for issue in issues) +
            "\n\n[dim]Run 'subtitle-forge quickstart' for guided setup[/dim]",
            title="Status",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            "[green]All systems operational![/green]\n\n"
            "Ready to process videos.",
            title="Status",
            border_style="green",
        ))


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


@app.command("pull-model")
def pull_model(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to download (default: configured model)",
    ),
):
    """
    Download Ollama translation model with progress display.

    Supports automatic resume - if download is interrupted,
    simply run this command again to continue from where it left off.

    Example:
        subtitle-forge config pull-model
        subtitle-forge config pull-model --model qwen2.5:32b
    """
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn

    from ...core.model_manager import OllamaModelManager, format_bytes

    config = AppConfig.load()
    target_model = model or config.ollama.model

    console.print(Panel(
        f"[cyan]Model:[/cyan] {target_model}\n"
        f"[cyan]Host:[/cyan] {config.ollama.host}",
        title="Download Configuration",
        border_style="blue",
    ))

    manager = OllamaModelManager(host=config.ollama.host)

    # Check connection first
    if not manager.check_connection():
        print_error("Cannot connect to Ollama service")
        console.print("\n[yellow]Please ensure Ollama is running:[/yellow]")
        console.print("  ollama serve")
        raise typer.Exit(1)

    # Check if already available
    if manager.is_model_available(target_model):
        print_success(f"Model {target_model} is already downloaded and ready to use")
        return

    console.print(f"\n[cyan]Downloading model: {target_model}[/cyan]")
    console.print("[dim]Tip: If interrupted, run this command again to resume download[/dim]\n")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            DownloadColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing...", total=None)

            for dp in manager.pull_model(target_model):
                # Safely check total_bytes (may be None or 0 during initialization)
                if dp.total_bytes and dp.total_bytes > 0:
                    progress.update(
                        task,
                        total=dp.total_bytes,
                        completed=dp.completed_bytes or 0,
                        description=dp.status.replace("_", " ").capitalize(),
                    )
                else:
                    progress.update(task, description=dp.status.replace("_", " ").capitalize())

        print_success(f"Model {target_model} downloaded successfully!")
        console.print("\n[green]You can now use subtitle-forge for translation.[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Download paused. Run this command again to resume.[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        print_error(f"Download failed: {e}")
        console.print("\n[yellow]You can try again - download will resume from where it stopped.[/yellow]")
        raise typer.Exit(1)
