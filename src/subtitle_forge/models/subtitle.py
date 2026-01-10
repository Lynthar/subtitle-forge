"""Subtitle data model."""

from dataclasses import dataclass


@dataclass
class SubtitleSegment:
    """A single subtitle segment with timing information."""

    index: int
    start: float  # Start time in seconds
    end: float  # End time in seconds
    text: str

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start

    def __str__(self) -> str:
        return f"[{self.start:.2f}-{self.end:.2f}] {self.text}"
