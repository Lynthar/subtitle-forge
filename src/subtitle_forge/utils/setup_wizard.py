"""Interactive setup wizard for first-time users."""

import shutil
from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    from ..core.transcriber import Transcriber

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn

from .progress import print_success, print_error, print_info, print_warning
from ..models.config import AppConfig

console = Console()


def _build_transcriber(whisper_cfg) -> "Transcriber":
    """A Transcriber honouring the configured cache/mirror/token settings."""
    from ..core.transcriber import Transcriber

    return Transcriber.from_config(whisper_cfg)


def check_whisper_model(whisper_cfg) -> bool:
    """Check if the configured Whisper model is already downloaded."""
    return _build_transcriber(whisper_cfg).is_model_cached()


def download_whisper_model(
    whisper_cfg,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """Download Whisper model with progress display."""
    import logging

    transcriber = _build_transcriber(whisper_cfg)

    # Suppress logs during download to avoid interfering with progress bar
    hf_logger = logging.getLogger("huggingface_hub")
    sf_logger = logging.getLogger("subtitle_forge")
    original_hf_level = hf_logger.level
    original_sf_level = sf_logger.level

    try:
        model_size = transcriber.get_model_size()

        # Suppress logs during download
        hf_logger.setLevel(logging.ERROR)
        sf_logger.setLevel(logging.ERROR)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            DownloadColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=model_size)
            last_completed = 0

            def update_progress(downloaded: int, total: int):
                nonlocal last_completed
                # Only update completed, not total (avoid accumulation bug)
                if downloaded > last_completed:
                    progress.update(task, completed=downloaded)
                    last_completed = downloaded

            transcriber.download_model(progress_callback=update_progress)

        return True
    except KeyboardInterrupt:
        console.print("\n[yellow]Download paused. You can resume later when running subtitle-forge.[/yellow]")
        return False
    except Exception as e:
        console.print(f"\n[red]Download failed: {e}[/red]")
        return False
    finally:
        # Restore log levels
        hf_logger.setLevel(original_hf_level)
        sf_logger.setLevel(original_sf_level)


def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed."""
    return shutil.which("ffmpeg") is not None


def check_gpu() -> dict:
    """Check GPU availability and info."""
    from .gpu import check_cuda_available, get_gpu_info

    if check_cuda_available():
        return get_gpu_info()
    return {"cuda_available": False}


def check_ollama_connection(host: str) -> bool:
    """Check if Ollama service is running."""
    from ..core.model_manager import OllamaModelManager

    manager = OllamaModelManager(host=host)
    return manager.check_connection()


def check_ollama_model(host: str, model: str) -> bool:
    """Check if Ollama model is available."""
    from ..core.model_manager import OllamaModelManager

    manager = OllamaModelManager(host=host)
    return manager.is_model_available(model)


def download_ollama_model(host: str, model: str) -> bool:
    """Download Ollama model with progress display."""
    from ..core.model_manager import OllamaModelManager

    manager = OllamaModelManager(host=host)

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

            for dp in manager.pull_model(model):
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

        return True
    except KeyboardInterrupt:
        console.print("\n[yellow]Download paused. You can resume later with: subtitle-forge config pull-model[/yellow]")
        return False
    except Exception as e:
        console.print(f"\n[red]Download failed: {e}[/red]")
        return False


def run_setup_wizard() -> None:
    """Run the interactive setup wizard."""

    # Welcome message
    console.print(Panel(
        "[bold cyan]Welcome to subtitle-forge![/bold cyan]\n\n"
        "This tool generates and translates subtitles for your videos\n"
        "using local AI models - no internet required for processing.\n\n"
        "Let's make sure everything is set up correctly.",
        title="Quick Start Guide",
        border_style="cyan",
    ))

    console.print()
    issues_found = []

    # Step 1: Check ffmpeg
    console.print("[bold]Step 1/5: Checking ffmpeg...[/bold]")

    if check_ffmpeg():
        console.print("  [green]OK[/green] ffmpeg is installed")
    else:
        console.print("  [red]MISSING[/red] ffmpeg is not installed")
        console.print()
        console.print("  [yellow]Please install ffmpeg:[/yellow]")
        console.print("    macOS:   brew install ffmpeg")
        console.print("    Ubuntu:  sudo apt install ffmpeg")
        console.print("    Windows: choco install ffmpeg")
        console.print()
        issues_found.append("ffmpeg")

        if not typer.confirm("  Continue setup anyway?", default=False):
            raise typer.Exit(1)

    console.print()

    # Step 2: Check GPU
    console.print("[bold]Step 2/5: Checking GPU...[/bold]")

    gpu_info = check_gpu()
    if gpu_info.get("cuda_available"):
        console.print(f"  [green]OK[/green] GPU detected: {gpu_info.get('device_name', 'Unknown')}")
        console.print(f"      VRAM: {gpu_info.get('total_vram_mb', 0)}MB total, {gpu_info.get('available_vram_mb', 0)}MB available")

        # Recommend Whisper model based on VRAM
        from ..core.transcriber import Transcriber
        recommended_whisper = Transcriber.select_optimal_model()
        console.print(f"      Recommended Whisper model: [cyan]{recommended_whisper}[/cyan]")

        # Recommend translation model based on VRAM
        total_vram = gpu_info.get("total_vram_mb", 0)
        if total_vram >= 30000:
            recommended_ollama = "qwen2.5:32b"
        elif total_vram >= 16000:
            recommended_ollama = "qwen2.5:14b"
        else:
            recommended_ollama = "qwen2.5:7b"
        console.print(f"      Recommended translation model: [cyan]{recommended_ollama}[/cyan]")

        # Offer to persist the recommendation. Without this the wizard said
        # "recommended: 7b" and then step 3/5 prompted to download the
        # config defaults (large-v3 + 14b) anyway — an OOM trap on small GPUs.
        gpu_config = AppConfig.load()
        if (
            recommended_whisper != gpu_config.whisper.model
            or recommended_ollama != gpu_config.ollama.model
        ) and typer.confirm(
            f"  Use the recommended models ({recommended_whisper} + {recommended_ollama})?",
            default=True,
        ):
            gpu_config.whisper.model = recommended_whisper
            gpu_config.ollama.model = recommended_ollama
            gpu_config.save()
            console.print("      [dim]Saved recommended models to config[/dim]")
    else:
        console.print("  [yellow]INFO[/yellow] No NVIDIA GPU detected")
        console.print("      Transcription will use CPU (slower but functional)")
        console.print("      Recommended Whisper model for CPU: [cyan]small[/cyan]")
        console.print("      Recommended translation model: [cyan]qwen2.5:7b[/cyan]")

        # Persist CPU settings now: the default is device=cuda and the base install has no CUDA
        # auto-fallback, so the first run would crash loading the model. int8 is the CPU compute
        # type.
        cpu_config = AppConfig.load()
        if cpu_config.whisper.device != "cpu":
            cpu_config.whisper.device = "cpu"
            cpu_config.whisper.compute_type = "int8"
            cpu_config.save()
            console.print("      [dim]Saved device=cpu, compute_type=int8 to config[/dim]")

        # Models behind a confirm — unlike the device fix above this is a
        # quality/speed tradeoff, not a crash fix, so the user decides.
        if (
            cpu_config.whisper.model != "small" or cpu_config.ollama.model != "qwen2.5:7b"
        ) and typer.confirm(
            "  Use the recommended CPU models (small + qwen2.5:7b)? "
            "The defaults (large-v3 + 14b) are painfully slow on CPU",
            default=True,
        ):
            cpu_config.whisper.model = "small"
            cpu_config.ollama.model = "qwen2.5:7b"
            cpu_config.save()
            console.print("      [dim]Saved recommended models to config[/dim]")

    console.print()

    # Step 3: Check Whisper model (transcription)
    console.print("[bold]Step 3/5: Checking Whisper model (transcription engine)...[/bold]")

    config = AppConfig.load()

    if check_whisper_model(config.whisper):
        console.print(f"  [green]OK[/green] Whisper model '{config.whisper.model}' is ready")
    else:
        from ..core.transcriber import WHISPER_MODEL_SIZES

        model_size = WHISPER_MODEL_SIZES.get(config.whisper.model, 1_000_000_000)
        model_size_mb = model_size / (1024 * 1024)

        console.print(f"  [yellow]MISSING[/yellow] Whisper model '{config.whisper.model}' not downloaded")
        console.print()
        console.print(f"  The Whisper model is needed for speech recognition.")
        console.print(f"  Size: ~{model_size_mb:.0f}MB (one-time download)")
        console.print()

        if typer.confirm(f"  Download '{config.whisper.model}' now?", default=True):
            console.print()
            console.print("  [dim]Tip: If interrupted, it will auto-download when you run subtitle-forge[/dim]")
            console.print()

            if download_whisper_model(config.whisper):
                console.print()
                console.print(f"  [green]OK[/green] Whisper model '{config.whisper.model}' downloaded successfully!")
            else:
                issues_found.append("whisper-model")
        else:
            console.print()
            console.print("  [dim]Model will be downloaded automatically when you first run subtitle-forge[/dim]")

    console.print()

    # Step 4: Check Ollama
    console.print("[bold]Step 4/5: Checking Ollama (translation engine)...[/bold]")

    if check_ollama_connection(config.ollama.host):
        console.print(f"  [green]OK[/green] Ollama is running at {config.ollama.host}")
    else:
        console.print("  [red]NOT RUNNING[/red] Cannot connect to Ollama")
        console.print()
        console.print("  [yellow]Please install and start Ollama:[/yellow]")
        console.print("    1. Download from: https://ollama.ai/download")
        console.print("    2. Start the service: ollama serve")
        console.print()
        issues_found.append("ollama")

        if not typer.confirm("  Continue setup anyway?", default=False):
            raise typer.Exit(1)

    console.print()

    # Step 5: Check/download translation model
    console.print("[bold]Step 5/5: Checking translation model...[/bold]")

    if "ollama" in issues_found:
        console.print("  [yellow]SKIPPED[/yellow] Ollama not running, cannot check model")
    elif check_ollama_model(config.ollama.host, config.ollama.model):
        console.print(f"  [green]OK[/green] Model '{config.ollama.model}' is ready")
    else:
        console.print(f"  [yellow]MISSING[/yellow] Model '{config.ollama.model}' not downloaded")
        console.print()
        console.print(f"  The translation model ({config.ollama.model}) needs to be downloaded.")
        console.print("  This is a one-time download and may take a while for large models.")
        console.print()

        if typer.confirm(f"  Download '{config.ollama.model}' now?", default=True):
            console.print()
            console.print("  [dim]Tip: If interrupted, run 'subtitle-forge config pull-model' to resume[/dim]")
            console.print()

            if download_ollama_model(config.ollama.host, config.ollama.model):
                console.print()
                console.print(f"  [green]OK[/green] Model '{config.ollama.model}' downloaded successfully!")
            else:
                issues_found.append("model")
        else:
            console.print()
            console.print("  [dim]You can download later with: subtitle-forge config pull-model[/dim]")
            issues_found.append("model")

    console.print()

    # Summary
    if issues_found:
        console.print(Panel(
            "[yellow]Setup completed with some issues.[/yellow]\n\n"
            f"Issues found: {', '.join(issues_found)}\n\n"
            "Please resolve the issues above before using subtitle-forge.\n"
            "Run 'subtitle-forge config check' to verify your setup.",
            title="Setup Status",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            "[bold green]Setup Complete![/bold green]\n\n"
            "Everything is configured and ready to use!\n\n"
            "[cyan]Quick Start Examples:[/cyan]\n\n"
            "  # Generate Chinese subtitles for a video\n"
            "  subtitle-forge process video.mp4 -t zh\n\n"
            "  # Generate subtitles in multiple languages\n"
            "  subtitle-forge process video.mp4 -t zh -t ja -t ko\n\n"
            "  # Create bilingual subtitles (original + translation)\n"
            "  subtitle-forge process video.mp4 -t zh --bilingual\n\n"
            "  # Process all videos in a folder\n"
            "  subtitle-forge batch ./videos/ -t zh\n\n"
            "[dim]For more options: subtitle-forge --help[/dim]",
            title="Ready!",
            border_style="green",
        ))
