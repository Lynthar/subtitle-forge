"""Subtitle processing module for SRT file handling."""

from pathlib import Path
from typing import List, Optional
from datetime import timedelta
import logging

import pysrt

from ..models.subtitle import SubtitleSegment
from ..exceptions import SubtitleError

logger = logging.getLogger(__name__)


class SubtitleProcessor:
    """Subtitle file processor."""

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    @staticmethod
    def seconds_to_time(seconds: float) -> pysrt.SubRipTime:
        """Convert seconds to SRT time format."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, secs = divmod(remainder, 60)
        milliseconds = int((secs - int(secs)) * 1000)

        return pysrt.SubRipTime(
            hours=int(hours),
            minutes=int(minutes),
            seconds=int(secs),
            milliseconds=milliseconds,
        )

    @staticmethod
    def time_to_seconds(time: pysrt.SubRipTime) -> float:
        """Convert SRT time format to seconds."""
        return time.hours * 3600 + time.minutes * 60 + time.seconds + time.milliseconds / 1000

    def segments_to_srt(self, segments: List[SubtitleSegment]) -> pysrt.SubRipFile:
        """Convert subtitle segments to pysrt object."""
        srt_file = pysrt.SubRipFile()

        for seg in segments:
            item = pysrt.SubRipItem(
                index=seg.index,
                start=self.seconds_to_time(seg.start),
                end=self.seconds_to_time(seg.end),
                text=seg.text,
            )
            srt_file.append(item)

        return srt_file

    def srt_to_segments(self, srt_file: pysrt.SubRipFile) -> List[SubtitleSegment]:
        """Convert pysrt object to subtitle segments."""
        segments = []

        for item in srt_file:
            segments.append(
                SubtitleSegment(
                    index=item.index,
                    start=self.time_to_seconds(item.start),
                    end=self.time_to_seconds(item.end),
                    text=item.text,
                )
            )

        return segments

    def save(
        self,
        segments: List[SubtitleSegment],
        output_path: Path,
        encoding: Optional[str] = None,
    ) -> None:
        """
        Save subtitles to SRT file.

        Args:
            segments: Subtitle segments to save.
            output_path: Output file path.
            encoding: File encoding.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        srt_file = self.segments_to_srt(segments)
        srt_file.save(str(output_path), encoding=encoding or self.encoding)

        logger.info(f"Subtitles saved: {output_path}")

    def load(self, input_path: Path, encoding: Optional[str] = None) -> List[SubtitleSegment]:
        """
        Load subtitles from SRT file.

        Args:
            input_path: Input file path.
            encoding: File encoding.

        Returns:
            List of subtitle segments.
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise SubtitleError(f"Subtitle file not found: {input_path}")

        # Try different encodings
        encodings_to_try = [encoding or self.encoding, "utf-8-sig", "gbk", "gb2312", "iso-8859-1"]

        for enc in encodings_to_try:
            try:
                srt_file = pysrt.open(str(input_path), encoding=enc)
                if enc != (encoding or self.encoding):
                    logger.info(f"Loaded file with encoding: {enc}")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise SubtitleError(f"Cannot decode subtitle file: {input_path}")

        segments = self.srt_to_segments(srt_file)
        logger.info(f"Loaded subtitles: {input_path} ({len(segments)} segments)")

        return segments

    def merge_bilingual(
        self,
        original: List[SubtitleSegment],
        translated: List[SubtitleSegment],
        original_on_top: bool = True,
    ) -> List[SubtitleSegment]:
        """
        Merge original and translated subtitles into bilingual format.

        Args:
            original: Original subtitles.
            translated: Translated subtitles.
            original_on_top: Place original text on top.

        Returns:
            Merged bilingual subtitles.
        """
        merged = []
        trans_dict = {seg.index: seg for seg in translated}

        for orig in original:
            trans = trans_dict.get(orig.index)
            if trans:
                if original_on_top:
                    text = f"{orig.text}\n{trans.text}"
                else:
                    text = f"{trans.text}\n{orig.text}"
            else:
                text = orig.text

            merged.append(
                SubtitleSegment(
                    index=orig.index,
                    start=orig.start,
                    end=orig.end,
                    text=text,
                )
            )

        return merged
