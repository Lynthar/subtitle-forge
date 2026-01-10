"""Progress display utilities using Rich."""

from typing import Optional, List
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    Progress,
    TaskID,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.panel import Panel
from rich.table import Table

from ..models.task import VideoTask, TaskStatus

console = Console()


class SubtitleProgress:
    """Subtitle processing progress manager."""

    def __init__(self, disable: bool = False):
        self.disable = disable

    def create_progress(self) -> Progress:
        """Create a progress bar instance."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("["),
            TimeElapsedColumn(),
            TextColumn("/"),
            TimeRemainingColumn(),
            TextColumn("]"),
            console=console,
            disable=self.disable,
        )

    @contextmanager
    def track_video(self, video_name: str, total_steps: int = 4):
        """
        Track progress for a single video.

        Args:
            video_name: Video file name.
            total_steps: Total number of steps.
        """
        with self.create_progress() as progress:
            task = progress.add_task(f"Processing {video_name}", total=total_steps)

            class ProgressTracker:
                def __init__(self, prog: Progress, task_id: TaskID):
                    self.progress = prog
                    self.task_id = task_id
                    self.current_step = 0

                def update(self, description: str, advance: int = 1):
                    self.progress.update(
                        self.task_id,
                        description=description,
                        advance=advance,
                    )
                    self.current_step += advance

                def set_description(self, description: str):
                    self.progress.update(self.task_id, description=description)

            yield ProgressTracker(progress, task)

    @contextmanager
    def track_batch(self, total_videos: int):
        """
        Track progress for batch processing.

        Args:
            total_videos: Total number of videos.
        """
        with self.create_progress() as progress:
            overall_task = progress.add_task(
                "[cyan]Overall progress",
                total=total_videos,
            )
            current_task = progress.add_task(
                "[green]Current video",
                total=100,
                visible=False,
            )

            class BatchTracker:
                def __init__(self, prog: Progress, overall: TaskID, current: TaskID):
                    self.progress = prog
                    self.overall = overall
                    self.current = current

                def start_video(self, name: str):
                    self.progress.update(
                        self.current,
                        description=f"[green]{name}",
                        completed=0,
                        visible=True,
                    )

                def update_current(self, percent: float, description: str = ""):
                    desc = f"[green]{description}" if description else None
                    self.progress.update(
                        self.current,
                        completed=int(percent),
                        description=desc,
                    )

                def complete_video(self):
                    self.progress.update(self.overall, advance=1)
                    self.progress.update(self.current, visible=False)

            yield BatchTracker(progress, overall_task, current_task)


def print_task_summary(tasks: List[VideoTask]) -> None:
    """Print task summary table."""
    table = Table(title="Processing Results")

    table.add_column("File", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right")
    table.add_column("Note")

    for task in tasks:
        if task.status == TaskStatus.COMPLETED:
            status = "[green]Success[/green]"
        elif task.status == TaskStatus.FAILED:
            status = "[red]Failed[/red]"
        else:
            status = "[yellow]Processing[/yellow]"

        duration = ""
        if task.elapsed_time:
            duration = f"{task.elapsed_time:.1f}s"

        note = task.error if task.error else ""
        if len(note) > 50:
            note = note[:50] + "..."

        table.add_row(
            task.video_path.name,
            status,
            duration,
            note,
        )

    console.print(table)


def print_error(message: str) -> None:
    """Print error message."""
    console.print(Panel(message, title="Error", border_style="red"))


def print_success(message: str) -> None:
    """Print success message."""
    console.print(Panel(message, title="Complete", border_style="green"))


def print_info(message: str) -> None:
    """Print info message."""
    console.print(f"[cyan]i[/cyan] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]![/yellow] {message}")
