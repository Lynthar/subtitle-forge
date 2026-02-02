"""Progress display utilities using Rich."""

from typing import Optional, List
from contextlib import contextmanager
from datetime import datetime

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
                    self._paused = False

                def update(self, description: str, advance: int = 1):
                    self.progress.update(
                        self.task_id,
                        description=description,
                        advance=advance,
                    )
                    self.current_step += advance

                def set_description(self, description: str):
                    self.progress.update(self.task_id, description=description)

                def pause(self):
                    """Pause the progress bar to allow nested progress bars."""
                    if not self._paused:
                        self.progress.stop()
                        self._paused = True

                def resume(self):
                    """Resume the progress bar after nested operations."""
                    if self._paused:
                        self.progress.start()
                        self._paused = False

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


class TranslationProgressTracker:
    """
    Track translation progress with detailed batch information.

    Shows progress like: "Translating subtitles... 45/150 [batch 5/15] 1:30 ~ 3:00"
    """

    def __init__(
        self,
        total_segments: int,
        batch_size: int = 10,
        target_lang: str = "",
        disable: bool = False,
    ):
        """
        Initialize translation progress tracker.

        Args:
            total_segments: Total number of subtitle segments to translate.
            batch_size: Number of segments per translation batch.
            target_lang: Target language name for display.
            disable: Disable progress display.
        """
        self.total_segments = total_segments
        self.batch_size = batch_size
        self.target_lang = target_lang
        self.disable = disable

        self.completed_segments = 0
        self.current_batch = 0
        self.total_batches = (total_segments + batch_size - 1) // batch_size
        self.start_time: Optional[datetime] = None

        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None

    def __enter__(self):
        self.start_time = datetime.now()

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[dim]batch {task.fields[batch]}/{task.fields[total_batches]}[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]~[/dim]"),
            TimeRemainingColumn(),
            console=console,
            disable=self.disable,
        )
        self._progress.start()

        desc = f"Translating to {self.target_lang}..." if self.target_lang else "Translating..."
        self._task_id = self._progress.add_task(
            desc,
            total=self.total_segments,
            batch=0,
            total_batches=self.total_batches,
        )
        return self

    def __exit__(self, *args):
        if self._progress:
            self._progress.stop()

    def update(self, completed: int, total: int):
        """
        Update progress - designed to be used as translator callback.

        Args:
            completed: Number of segments completed.
            total: Total number of segments.
        """
        self.completed_segments = completed
        self.current_batch = (completed + self.batch_size - 1) // self.batch_size

        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=completed,
                batch=self.current_batch,
            )

    def get_stats(self) -> dict:
        """Get translation statistics."""
        elapsed = 0.0
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        rate = self.completed_segments / elapsed if elapsed > 0 else 0

        return {
            "completed": self.completed_segments,
            "total": self.total_segments,
            "batches_done": self.current_batch,
            "total_batches": self.total_batches,
            "elapsed_seconds": elapsed,
            "rate_per_second": rate,
        }


def print_translation_explainer(show_once: bool = True) -> None:
    """
    Print user-friendly explanation of the translation process.

    Args:
        show_once: Only show once per session (uses module-level flag).
    """
    # Module-level flag to track if already shown
    if show_once and getattr(print_translation_explainer, "_shown", False):
        return

    console.print(Panel(
        "[cyan]Translation Process[/cyan]\n\n"
        "Your subtitles are being translated using a local AI model.\n"
        "This runs entirely on your computer - no internet required.\n\n"
        "[dim]Progress shows: completed subtitles / total subtitles[/dim]\n"
        "[dim]Batch processing: subtitles are translated in groups for efficiency[/dim]",
        title="What's happening?",
        border_style="blue",
    ))

    if show_once:
        print_translation_explainer._shown = True


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
