"""Timestamp post-processing module for improving subtitle timing accuracy."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import re

from ..models.subtitle import SubtitleSegment, WordTiming

logger = logging.getLogger(__name__)

# Sentence-ending patterns for different languages
SENTENCE_ENDINGS = re.compile(r'([。！？!?\.\n]+)')
# Japanese/Chinese specific sentence endings
CJK_SENTENCE_ENDINGS = re.compile(r'([。！？」』）]+)')
# Pattern to detect sentence-ending punctuation (for word matching)
SENTENCE_END_CHARS = set('。！？!?.')


@dataclass
class TimestampIssue:
    """Record of a detected timestamp issue."""

    segment_index: int
    issue_type: str  # 'overlap', 'long_duration', 'large_gap', 'short_duration', 'exceeds_duration'
    severity: str  # 'info', 'warning', 'error'
    details: str


@dataclass
class GapInfo:
    """Information about a detected gap that may indicate missing speech."""

    start: float
    end: float
    duration: float
    before_segment_index: int
    after_segment_index: int


class TimestampProcessor:
    """
    Post-processor for subtitle timestamps.

    Supports three processing modes:
    - "off": No processing, trust WhisperX output
    - "minimal": Only fix overlaps and ensure minimum duration
    - "full": Complete processing (split, extend, fix all issues)
    """

    # CJK language codes
    CJK_LANGUAGES = {'zh', 'ja', 'ko', 'chinese', 'japanese', 'korean', 'yue', 'wuu'}

    def __init__(
        self,
        mode: str = "minimal",
        language: Optional[str] = None,
        min_duration: float = 0.5,
        max_duration: float = 8.0,
        min_gap: float = 0.05,
        max_gap_warning: float = 50.0,
        chars_per_second: float = 15.0,
        cjk_chars_per_second: float = 10.0,
        split_threshold: int = 30,
        split_long_segments: bool = True,
        extend_end_times: bool = True,
        split_sentences: bool = False,
    ):
        """
        Initialize timestamp processor.

        Args:
            mode: Processing mode - "off", "minimal", or "full".
            language: Detected language code for CJK optimization.
            min_duration: Minimum subtitle duration in seconds.
            max_duration: Maximum subtitle duration in seconds.
            min_gap: Minimum gap between subtitles in seconds.
            max_gap_warning: Gap threshold for potential missed speech warning.
            chars_per_second: Reading speed for Western languages.
            cjk_chars_per_second: Reading speed for CJK languages.
            split_threshold: Minimum characters before attempting split.
            split_long_segments: Split segments containing multiple sentences (full mode only).
            extend_end_times: Extend end times based on text length (full mode only).
            split_sentences: Split segments by sentences using word-level timestamps.
        """
        self.mode = mode
        self.language = language
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_gap = min_gap
        self.max_gap_warning = max_gap_warning
        self.chars_per_second = chars_per_second
        self.cjk_chars_per_second = cjk_chars_per_second
        self.split_threshold = split_threshold
        self.split_long_segments = split_long_segments
        self.extend_end_times = extend_end_times
        self.split_sentences = split_sentences
        self._issues: List[TimestampIssue] = []
        self._gaps: List[GapInfo] = []

        # Select effective reading speed based on language
        if self._is_cjk_language(language):
            self._effective_cps = cjk_chars_per_second
            logger.debug(f"Using CJK reading speed: {cjk_chars_per_second} chars/sec")
        else:
            self._effective_cps = chars_per_second

    @classmethod
    def _is_cjk_language(cls, language: Optional[str]) -> bool:
        """Check if language is CJK (Chinese, Japanese, Korean)."""
        if not language:
            return False
        return language.lower() in cls.CJK_LANGUAGES

    def _get_split_threshold(self) -> int:
        """Get language-appropriate split threshold."""
        if self._is_cjk_language(self.language):
            # CJK characters are denser, use lower threshold
            return max(15, self.split_threshold // 2)
        return self.split_threshold

    def process(
        self,
        segments: List[SubtitleSegment],
        audio_duration: Optional[float] = None,
    ) -> List[SubtitleSegment]:
        """
        Execute post-processing pipeline based on mode.

        Args:
            segments: List of subtitle segments to process.
            audio_duration: Total audio duration for boundary checking.

        Returns:
            Processed subtitle segments with corrected timestamps.
        """
        self._issues = []
        self._gaps = []

        if not segments:
            return segments

        # Sentence splitting runs independently of mode if enabled
        if self.split_sentences:
            original_count = len(segments)
            segments = self._split_by_sentences(segments)
            if len(segments) != original_count:
                logger.info(
                    f"Sentence splitting: {original_count} segments -> {len(segments)} segments"
                )

        # Mode: off - trust WhisperX output completely
        if self.mode == "off":
            logger.debug("Timestamp processing mode: off (no processing)")
            return self._reindex(segments)

        # Mode: minimal - only essential fixes
        if self.mode == "minimal":
            logger.debug("Timestamp processing mode: minimal")
            segments = self._fix_overlaps(segments)
            segments = self._ensure_minimum_duration(segments)
            if audio_duration:
                segments = self._clamp_to_duration(segments, audio_duration)
            return self._reindex(segments)

        # Mode: full - complete processing (original behavior)
        logger.debug("Timestamp processing mode: full")

        # 0. Pre-processing for segments without word-level timestamps
        if self.split_long_segments:
            segments = self._split_multi_sentence_segments(segments)

        if self.extend_end_times:
            segments = self._extend_segment_end_times(segments)

        # 1. Validate and record issues
        self._validate(segments, audio_duration)

        # 2. Apply fixes
        segments = self._fix_overlaps(segments)
        segments = self._fix_long_displays(segments, audio_duration)
        segments = self._ensure_minimum_duration(segments)
        segments = self._ensure_minimum_gap(segments)

        # 3. Clamp to audio duration if known
        if audio_duration:
            segments = self._clamp_to_duration(segments, audio_duration)

        # 4. Detect potential gaps (missing speech)
        self._detect_large_gaps(segments, audio_duration)

        # 5. Reindex
        segments = self._reindex(segments)

        # Log summary
        self._log_summary()

        return segments

    def get_issues(self) -> List[TimestampIssue]:
        """Get list of detected timestamp issues."""
        return self._issues.copy()

    def get_potential_gaps(self) -> List[GapInfo]:
        """Get list of large gaps that may indicate missed speech."""
        return self._gaps.copy()

    def _validate(
        self,
        segments: List[SubtitleSegment],
        audio_duration: Optional[float],
    ) -> None:
        """Validate segments and record issues."""
        for i, seg in enumerate(segments):
            # Check negative/zero duration
            if seg.end <= seg.start:
                self._issues.append(
                    TimestampIssue(
                        segment_index=seg.index,
                        issue_type="negative_duration",
                        severity="error",
                        details=f"End ({seg.end:.2f}s) <= Start ({seg.start:.2f}s)",
                    )
                )

            # Check excessively long duration
            duration = seg.end - seg.start
            if duration > self.max_duration:
                self._issues.append(
                    TimestampIssue(
                        segment_index=seg.index,
                        issue_type="long_duration",
                        severity="warning",
                        details=f"Duration {duration:.2f}s exceeds max {self.max_duration}s",
                    )
                )

            # Check overlap with previous
            if i > 0:
                prev = segments[i - 1]
                if seg.start < prev.end:
                    overlap = prev.end - seg.start
                    self._issues.append(
                        TimestampIssue(
                            segment_index=seg.index,
                            issue_type="overlap",
                            severity="warning" if overlap < 0.5 else "error",
                            details=f"Overlaps with previous segment by {overlap:.2f}s",
                        )
                    )

            # Check exceeds audio duration
            if audio_duration and seg.end > audio_duration:
                self._issues.append(
                    TimestampIssue(
                        segment_index=seg.index,
                        issue_type="exceeds_duration",
                        severity="error",
                        details=f"End ({seg.end:.2f}s) exceeds audio duration ({audio_duration:.2f}s)",
                    )
                )

    def _fix_overlaps(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Fix overlapping timestamps between segments."""
        if len(segments) <= 1:
            return segments

        result = [segments[0]]
        for seg in segments[1:]:
            prev = result[-1]

            if seg.start < prev.end:
                # Calculate midpoint for fair distribution
                midpoint = (prev.end + seg.start) / 2

                # Adjust previous segment's end
                adjusted_prev_end = midpoint - self.min_gap / 2
                if adjusted_prev_end > prev.start + 0.1:
                    result[-1] = SubtitleSegment(
                        index=prev.index,
                        start=prev.start,
                        end=adjusted_prev_end,
                        text=prev.text,
                    )

                # Adjust current segment's start
                adjusted_start = midpoint + self.min_gap / 2
                seg = SubtitleSegment(
                    index=seg.index,
                    start=adjusted_start,
                    end=seg.end,
                    text=seg.text,
                )

            result.append(seg)
        return result

    def _fix_long_displays(
        self,
        segments: List[SubtitleSegment],
        audio_duration: Optional[float],
    ) -> List[SubtitleSegment]:
        """Fix segments with excessively long display duration."""
        result = []
        for i, seg in enumerate(segments):
            duration = seg.end - seg.start

            if duration > self.max_duration:
                # Estimate reasonable duration based on text length
                chars = len(seg.text)
                estimated_duration = max(
                    self.min_duration, min(chars / self._effective_cps, self.max_duration)
                )

                new_end = seg.start + estimated_duration

                # Don't exceed next segment's start
                if i < len(segments) - 1:
                    next_start = segments[i + 1].start
                    new_end = min(new_end, next_start - self.min_gap)

                # Don't exceed audio duration
                if audio_duration:
                    new_end = min(new_end, audio_duration)

                # Ensure minimum duration
                new_end = max(new_end, seg.start + self.min_duration)

                seg = SubtitleSegment(
                    index=seg.index,
                    start=seg.start,
                    end=new_end,
                    text=seg.text,
                )

            result.append(seg)
        return result

    def _ensure_minimum_duration(
        self, segments: List[SubtitleSegment]
    ) -> List[SubtitleSegment]:
        """Ensure all segments have minimum display duration."""
        result = []
        for seg in segments:
            duration = seg.end - seg.start
            if duration < self.min_duration:
                seg = SubtitleSegment(
                    index=seg.index,
                    start=seg.start,
                    end=seg.start + self.min_duration,
                    text=seg.text,
                )
            result.append(seg)
        return result

    def _ensure_minimum_gap(
        self, segments: List[SubtitleSegment]
    ) -> List[SubtitleSegment]:
        """Ensure minimum gap between consecutive segments."""
        if len(segments) <= 1:
            return segments

        result = [segments[0]]
        for seg in segments[1:]:
            prev = result[-1]
            gap = seg.start - prev.end

            if gap < self.min_gap:
                # Adjust previous segment's end
                new_prev_end = seg.start - self.min_gap
                if new_prev_end > prev.start + 0.1:
                    result[-1] = SubtitleSegment(
                        index=prev.index,
                        start=prev.start,
                        end=new_prev_end,
                        text=prev.text,
                    )

            result.append(seg)
        return result

    def _clamp_to_duration(
        self,
        segments: List[SubtitleSegment],
        audio_duration: float,
    ) -> List[SubtitleSegment]:
        """Ensure all timestamps are within audio duration."""
        result = []
        for seg in segments:
            if seg.end > audio_duration:
                seg = SubtitleSegment(
                    index=seg.index,
                    start=min(seg.start, audio_duration - 0.1),
                    end=audio_duration,
                    text=seg.text,
                )
            result.append(seg)
        return result

    def _detect_large_gaps(
        self,
        segments: List[SubtitleSegment],
        audio_duration: Optional[float],
    ) -> None:
        """Detect large gaps that may indicate missed speech."""
        # Check gap at beginning
        if segments and segments[0].start > self.max_gap_warning:
            self._gaps.append(
                GapInfo(
                    start=0,
                    end=segments[0].start,
                    duration=segments[0].start,
                    before_segment_index=0,
                    after_segment_index=segments[0].index,
                )
            )
            logger.info(
                f"Large gap at start: 0.00s - {segments[0].start:.2f}s "
                f"(duration: {segments[0].start:.2f}s)"
            )

        # Check gaps between segments
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            gap = curr.start - prev.end

            if gap > self.max_gap_warning:
                self._gaps.append(
                    GapInfo(
                        start=prev.end,
                        end=curr.start,
                        duration=gap,
                        before_segment_index=prev.index,
                        after_segment_index=curr.index,
                    )
                )
                logger.info(
                    f"Large gap detected: {prev.end:.2f}s - {curr.start:.2f}s "
                    f"(duration: {gap:.2f}s) - may indicate missed speech"
                )

        # Check gap at end
        if audio_duration and segments:
            last = segments[-1]
            end_gap = audio_duration - last.end
            if end_gap > self.max_gap_warning:
                self._gaps.append(
                    GapInfo(
                        start=last.end,
                        end=audio_duration,
                        duration=end_gap,
                        before_segment_index=last.index,
                        after_segment_index=-1,  # -1 indicates end of audio
                    )
                )
                logger.info(
                    f"Large gap at end: {last.end:.2f}s - {audio_duration:.2f}s "
                    f"(duration: {end_gap:.2f}s)"
                )

    def _reindex(self, segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
        """Reindex segments starting from 1."""
        return [
            SubtitleSegment(
                index=i + 1,
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )
            for i, seg in enumerate(segments)
        ]

    def _split_multi_sentence_segments(
        self, segments: List[SubtitleSegment]
    ) -> List[SubtitleSegment]:
        """
        Split segments containing multiple sentences into separate segments.

        This is particularly useful when forced alignment fails and Whisper
        returns long segments with multiple sentences grouped together.
        """
        result = []

        for seg in segments:
            # Skip if segment has word-level timestamps (alignment worked)
            if seg.has_word_timestamps():
                result.append(seg)
                continue

            # Skip short segments (use language-aware threshold)
            threshold = self._get_split_threshold()
            if len(seg.text) < threshold:
                result.append(seg)
                continue

            # Try to split by sentence endings
            sentences = self._split_into_sentences(seg.text)

            if len(sentences) <= 1:
                result.append(seg)
                continue

            # Distribute time proportionally across sentences
            total_chars = sum(len(s) for s in sentences)
            total_duration = seg.end - seg.start
            current_time = seg.start

            for i, sentence in enumerate(sentences):
                # Calculate duration based on character ratio
                char_ratio = len(sentence) / total_chars if total_chars > 0 else 1.0 / len(sentences)
                duration = total_duration * char_ratio

                # Ensure minimum duration
                duration = max(duration, self.min_duration)

                # Calculate end time
                end_time = current_time + duration

                # Don't exceed original segment end for last sentence
                if i == len(sentences) - 1:
                    end_time = seg.end

                result.append(
                    SubtitleSegment(
                        index=len(result) + 1,
                        start=current_time,
                        end=end_time,
                        text=sentence.strip(),
                    )
                )

                current_time = end_time

            logger.debug(f"Split segment {seg.index} into {len(sentences)} parts")

        return result

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences based on punctuation.

        Handles both Western and CJK punctuation.
        """
        # First try CJK sentence endings
        if any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in text):
            # Contains CJK characters, use CJK pattern
            parts = CJK_SENTENCE_ENDINGS.split(text)
        else:
            # Use standard sentence endings
            parts = SENTENCE_ENDINGS.split(text)

        # Recombine parts (pattern split separates delimiters)
        sentences = []
        current = ""

        for part in parts:
            current += part
            # Check if this part is a sentence ending
            if SENTENCE_ENDINGS.match(part) or CJK_SENTENCE_ENDINGS.match(part):
                if current.strip():
                    sentences.append(current.strip())
                current = ""

        # Add remaining text if any
        if current.strip():
            sentences.append(current.strip())

        # Filter out empty sentences and very short ones
        sentences = [s for s in sentences if len(s) >= 2]

        return sentences if sentences else [text]

    def _extend_segment_end_times(
        self, segments: List[SubtitleSegment]
    ) -> List[SubtitleSegment]:
        """
        Extend segment end times based on text length.

        When forced alignment fails, segment end times may be too short.
        This method extends end times based on estimated reading time,
        while respecting the next segment's start time.
        """
        if not segments:
            return segments

        result = []

        for i, seg in enumerate(segments):
            # Skip if segment has word-level timestamps (alignment worked)
            if seg.has_word_timestamps():
                result.append(seg)
                continue

            # Calculate minimum required duration based on text length
            # Using language-appropriate reading speed
            chars = len(seg.text)
            min_required_duration = max(
                self.min_duration,
                chars / self._effective_cps
            )

            current_duration = seg.end - seg.start

            # Only extend if current duration is too short
            if current_duration < min_required_duration:
                new_end = seg.start + min_required_duration

                # Don't exceed next segment's start time (with gap)
                if i < len(segments) - 1:
                    next_start = segments[i + 1].start
                    max_end = next_start - self.min_gap
                    new_end = min(new_end, max_end)

                # Ensure we don't make it shorter than original
                new_end = max(new_end, seg.end)

                if new_end != seg.end:
                    seg = SubtitleSegment(
                        index=seg.index,
                        start=seg.start,
                        end=new_end,
                        text=seg.text,
                        words=seg.words,
                        confidence=seg.confidence,
                    )

            result.append(seg)

        return result

    def _split_by_sentences(
        self, segments: List[SubtitleSegment]
    ) -> List[SubtitleSegment]:
        """
        Split segments by sentence boundaries using word-level timestamps.

        This method detects sentence-ending punctuation and uses word timestamps
        to calculate precise timing for each sentence.
        """
        result = []

        for seg in segments:
            # If no word timestamps, use fallback proportional splitting
            if not seg.has_word_timestamps():
                split_segs = self._split_segment_proportionally(seg)
                result.extend(split_segs)
                continue

            # Find sentence boundaries using word timestamps
            sentences = self._extract_sentences_with_timing(seg)

            if len(sentences) <= 1:
                # No splitting needed
                result.append(seg)
                continue

            # Create new segments for each sentence
            for sentence_text, start_time, end_time in sentences:
                if not sentence_text.strip():
                    continue

                # Ensure minimum duration
                if end_time - start_time < self.min_duration:
                    end_time = start_time + self.min_duration

                result.append(
                    SubtitleSegment(
                        index=len(result) + 1,
                        start=start_time,
                        end=end_time,
                        text=sentence_text.strip(),
                    )
                )

        return result

    def _extract_sentences_with_timing(
        self, seg: SubtitleSegment
    ) -> List[Tuple[str, float, float]]:
        """
        Extract sentences from segment with their timing using word timestamps.

        Returns:
            List of (sentence_text, start_time, end_time) tuples.
        """
        if not seg.words:
            return [(seg.text, seg.start, seg.end)]

        sentences = []
        current_sentence_words: List[WordTiming] = []
        current_text_parts: List[str] = []

        for word in seg.words:
            word_text = word.word.strip()
            if not word_text:
                continue

            current_sentence_words.append(word)
            current_text_parts.append(word_text)

            # Check if this word ends with sentence-ending punctuation
            if self._is_sentence_end(word_text):
                if current_sentence_words:
                    sentence_text = self._join_words(current_text_parts)
                    start_time = current_sentence_words[0].start
                    end_time = current_sentence_words[-1].end
                    sentences.append((sentence_text, start_time, end_time))

                current_sentence_words = []
                current_text_parts = []

        # Handle remaining words (sentence without ending punctuation)
        if current_sentence_words:
            sentence_text = self._join_words(current_text_parts)
            start_time = current_sentence_words[0].start
            end_time = current_sentence_words[-1].end
            sentences.append((sentence_text, start_time, end_time))

        return sentences if sentences else [(seg.text, seg.start, seg.end)]

    def _is_sentence_end(self, word: str) -> bool:
        """Check if a word ends with sentence-ending punctuation."""
        if not word:
            return False
        # Check the last character (or last few for multi-char endings)
        for char in reversed(word):
            if char in SENTENCE_END_CHARS:
                return True
            if not char.isspace():
                break
        return False

    def _join_words(self, words: List[str]) -> str:
        """
        Join words into text, handling CJK vs Western spacing.
        """
        if not words:
            return ""

        # Check if primarily CJK
        is_cjk = self._is_cjk_language(self.language)

        if is_cjk:
            # CJK: no space between characters
            return "".join(words)
        else:
            # Western: space between words
            return " ".join(words)

    def _split_segment_proportionally(
        self, seg: SubtitleSegment
    ) -> List[SubtitleSegment]:
        """
        Fallback: split segment proportionally when no word timestamps available.
        """
        # Try to split by sentence endings
        sentences = self._split_into_sentences(seg.text)

        if len(sentences) <= 1:
            return [seg]

        # Distribute time proportionally across sentences
        total_chars = sum(len(s) for s in sentences)
        total_duration = seg.end - seg.start
        current_time = seg.start
        result = []

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Calculate duration based on character ratio
            char_ratio = len(sentence) / total_chars if total_chars > 0 else 1.0 / len(sentences)
            duration = total_duration * char_ratio

            # Ensure minimum duration
            duration = max(duration, self.min_duration)

            # Calculate end time
            end_time = current_time + duration

            # Don't exceed original segment end for last sentence
            if i == len(sentences) - 1:
                end_time = seg.end

            result.append(
                SubtitleSegment(
                    index=len(result) + 1,
                    start=current_time,
                    end=end_time,
                    text=sentence.strip(),
                )
            )

            current_time = end_time

        return result if result else [seg]

    def _log_summary(self) -> None:
        """Log processing summary."""
        error_count = sum(1 for i in self._issues if i.severity == "error")
        warning_count = sum(1 for i in self._issues if i.severity == "warning")
        gap_count = len(self._gaps)

        if error_count or warning_count or gap_count:
            logger.info(
                f"Timestamp processing: {error_count} errors, {warning_count} warnings, "
                f"{gap_count} potential gaps detected"
            )
        else:
            logger.info("Timestamp processing: No issues detected")
