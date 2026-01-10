"""Audio extraction module using ffmpeg."""

from pathlib import Path
from typing import Optional
import tempfile
import logging

import ffmpeg

from ..exceptions import AudioExtractionError

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Extract audio from video files using ffmpeg."""

    SUPPORTED_VIDEO_FORMATS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
    DEFAULT_AUDIO_FORMAT = "wav"
    DEFAULT_SAMPLE_RATE = 16000  # Recommended for Whisper

    def __init__(
        self,
        output_format: str = DEFAULT_AUDIO_FORMAT,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mono: bool = True,
    ):
        self.output_format = output_format
        self.sample_rate = sample_rate
        self.mono = mono

    def extract(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Extract audio from video file.

        Args:
            video_path: Path to video file.
            output_path: Output audio path. Uses temp file if None.

        Returns:
            Path to extracted audio file.
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise AudioExtractionError(f"Video file not found: {video_path}")

        if video_path.suffix.lower() not in self.SUPPORTED_VIDEO_FORMATS:
            raise AudioExtractionError(
                f"Unsupported video format: {video_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_VIDEO_FORMATS)}"
            )

        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=f".{self.output_format}"))

        logger.info(f"Extracting audio from {video_path.name}...")

        try:
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec="pcm_s16le" if self.output_format == "wav" else "libmp3lame",
                ac=1 if self.mono else 2,
                ar=self.sample_rate,
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise AudioExtractionError(f"Audio extraction failed: {error_msg}")

        logger.info(f"Audio extracted: {output_path}")
        return output_path

    def get_duration(self, video_path: Path) -> float:
        """Get video duration in seconds."""
        try:
            probe = ffmpeg.probe(str(video_path))
            return float(probe["format"]["duration"])
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
            return 0.0

    def get_video_info(self, video_path: Path) -> dict:
        """Get video information."""
        try:
            probe = ffmpeg.probe(str(video_path))
            format_info = probe.get("format", {})

            info = {
                "duration": float(format_info.get("duration", 0)),
                "size_mb": int(format_info.get("size", 0)) / (1024 * 1024),
                "format": format_info.get("format_name", "unknown"),
            }

            # Get video stream info
            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "video":
                    info["width"] = stream.get("width")
                    info["height"] = stream.get("height")
                    info["video_codec"] = stream.get("codec_name")
                    break

            return info
        except Exception as e:
            logger.warning(f"Could not get video info: {e}")
            return {}
