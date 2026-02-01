"""Translation module using Ollama."""

from typing import List, Optional, Callable
from dataclasses import dataclass
import re
import logging

import ollama
from ollama import Client, ResponseError

from ..models.subtitle import SubtitleSegment
from ..exceptions import TranslationError
from .model_manager import OllamaModelManager, DownloadProgress

logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    """Translation configuration."""

    model: str = "qwen2.5:14b"
    host: str = "http://localhost:11434"
    temperature: float = 0.3
    max_batch_size: int = 10
    max_retries: int = 3
    prompt_template: Optional[str] = None  # Custom prompt template (None = use default)
    prompt_template_id: Optional[str] = None  # Reference to prompt library template
    save_failed_log: bool = False  # Save failed translations to a log file
    failed_log_path: Optional[str] = None  # Path for failed translations log


class SubtitleTranslator:
    """Subtitle translator using Ollama LLM."""

    # Default translation prompt template
    # Available placeholders: {source_lang}, {target_lang}, {context_before}, {segments}, {context_after}
    DEFAULT_PROMPT_TEMPLATE = """You are an expert subtitle translator for movies and TV dramas.

TRANSLATION GUIDELINES:
1. Preserve natural dialogue flow and conversational tone
2. Capture emotional nuance, character voice, and speaker intent
3. Use appropriate register (formal/informal) based on context
4. Keep translations concise for subtitle readability
5. Maintain consistency with surrounding dialogue
6. Preserve the [number] prefix for each line
7. Output ONLY the translated lines, no explanations
{context_before}
TRANSLATE THESE LINES from {source_lang} to {target_lang}:
{segments}
{context_after}
Translated subtitles:"""

    LANGUAGE_NAMES = {
        "zh": "Simplified Chinese",
        "zh-TW": "Traditional Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ru": "Russian",
        "pt": "Portuguese",
        "it": "Italian",
        "ar": "Arabic",
        "vi": "Vietnamese",
        "th": "Thai",
    }

    # Approximate VRAM requirements for Ollama models (MB)
    OLLAMA_MODEL_VRAM = {
        "qwen2.5:7b": 5000,    # ~5GB
        "qwen2.5:14b": 9000,   # ~9GB
        "qwen2.5:32b": 20000,  # ~20GB
        "qwen2.5:72b": 45000,  # ~45GB
    }

    @classmethod
    def select_optimal_model(cls, prefer_quality: bool = True) -> str:
        """
        Select optimal translation model based on available VRAM.

        Args:
            prefer_quality: Prefer larger/higher quality models if VRAM allows.

        Returns:
            Recommended model name.
        """
        from ..utils.gpu import get_available_vram

        available_vram = get_available_vram()

        if available_vram <= 0:
            logger.warning("Cannot detect GPU VRAM, using default model")
            return "qwen2.5:14b"

        # Sort models by VRAM requirement (larger first if prefer_quality)
        sorted_models = sorted(
            cls.OLLAMA_MODEL_VRAM.items(),
            key=lambda x: x[1],
            reverse=prefer_quality,
        )

        for model_name, required_vram in sorted_models:
            if available_vram >= required_vram * 1.1:  # 10% headroom
                logger.info(
                    f"Selected translation model: {model_name} "
                    f"(requires {required_vram}MB, available {available_vram}MB)"
                )
                return model_name

        return "qwen2.5:7b"  # Fallback to smallest model

    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
        self._client: Optional[Client] = None
        self._model_manager: Optional[OllamaModelManager] = None
        self._failed_translations: List[dict] = []  # Track failed translations

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(host=self.config.host)
        return self._client

    @property
    def model_manager(self) -> OllamaModelManager:
        """Get or create model manager instance."""
        if self._model_manager is None:
            self._model_manager = OllamaModelManager(host=self.config.host)
        return self._model_manager

    def ensure_model_ready(
        self,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        auto_pull: bool = True,
    ) -> bool:
        """
        Ensure the translation model is ready, optionally downloading it.

        Args:
            progress_callback: Optional callback for download progress updates.
            auto_pull: If True, automatically download missing model.

        Returns:
            True if model is ready to use.
        """
        return self.model_manager.ensure_model(
            self.config.model,
            progress_callback=progress_callback,
            auto_pull=auto_pull,
        )

    def check_connection(self) -> bool:
        """Check if Ollama service is reachable."""
        return self.model_manager.check_connection()

    def check_model_available(self) -> bool:
        """Check if Ollama model is available."""
        return self.model_manager.is_model_available(self.config.model)

    def get_prompt_template(self) -> str:
        """
        Get the current prompt template with priority:
        1. Custom prompt template (direct string)
        2. Library template by ID
        3. Default template
        """
        # Custom prompt takes precedence (backward compatible)
        if self.config.prompt_template:
            return self.config.prompt_template

        # Library template by ID
        if self.config.prompt_template_id:
            from .prompt_library import get_prompt_library

            library = get_prompt_library()
            template = library.get_template(self.config.prompt_template_id)
            if template:
                return template.template
            else:
                logger.warning(
                    f"Prompt template '{self.config.prompt_template_id}' not found, "
                    "using default"
                )

        # Default
        return self.DEFAULT_PROMPT_TEMPLATE

    def get_prompt_template_info(self) -> Optional[str]:
        """
        Get info about current prompt source.

        Returns:
            String describing prompt source, or None for default.
        """
        if self.config.prompt_template:
            return "custom"
        if self.config.prompt_template_id:
            return f"library:{self.config.prompt_template_id}"
        return None

    def _build_translation_prompt(
        self,
        segments: List[SubtitleSegment],
        source_lang: str,
        target_lang: str,
        context_before: Optional[List[SubtitleSegment]] = None,
        context_after: Optional[List[SubtitleSegment]] = None,
    ) -> str:
        """
        Build translation prompt with context for natural translation.

        Args:
            segments: Segments to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            context_before: Previous segments for context (not translated).
            context_after: Following segments for context (not translated).

        Returns:
            Formatted prompt string.
        """
        source_name = self.LANGUAGE_NAMES.get(source_lang, source_lang)
        target_name = self.LANGUAGE_NAMES.get(target_lang, target_lang)

        # Format context sections
        context_before_text = ""
        if context_before:
            context_lines = [f"  [{seg.index}] {seg.text}" for seg in context_before[-3:]]
            context_before_text = f"""
PREVIOUS DIALOGUE (for context, DO NOT translate):
{chr(10).join(context_lines)}"""

        context_after_text = ""
        if context_after:
            context_lines = [f"  [{seg.index}] {seg.text}" for seg in context_after[:2]]
            context_after_text = f"""
FOLLOWING DIALOGUE (for context, DO NOT translate):
{chr(10).join(context_lines)}"""

        # Format segments
        lines = [f"[{seg.index}] {seg.text}" for seg in segments]
        segments_text = chr(10).join(lines)

        # Use custom template or default
        template = self.get_prompt_template()

        return template.format(
            source_lang=source_name,
            target_lang=target_name,
            context_before=context_before_text,
            segments=segments_text,
            context_after=context_after_text,
        )

    def _parse_translation_response(
        self,
        response: str,
        original_segments: List[SubtitleSegment],
    ) -> List[SubtitleSegment]:
        """Parse translation response with multiple fallback patterns."""
        translated = []

        # Log raw response for debugging (truncated)
        response_preview = response[:500] + "..." if len(response) > 500 else response
        logger.debug(f"Raw LLM response:\n{response_preview}")

        # Try multiple patterns to extract translations
        index_to_translation = {}

        # Pattern 1: [number] text (standard format)
        pattern1 = r"\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)"
        matches = re.findall(pattern1, response, re.DOTALL)
        for idx_str, text in matches:
            idx = int(idx_str)
            text = text.strip()
            if text and idx not in index_to_translation:
                index_to_translation[idx] = text

        # Pattern 2: number. text or number: text (alternative formats)
        if len(index_to_translation) < len(original_segments):
            pattern2 = r"(?:^|\n)\s*(\d+)[.:\)]\s*(.+?)(?=\n\s*\d+[.:\)]|$)"
            matches = re.findall(pattern2, response, re.DOTALL)
            for idx_str, text in matches:
                idx = int(idx_str)
                text = text.strip()
                if text and idx not in index_to_translation:
                    index_to_translation[idx] = text

        # Pattern 3: Line-by-line matching (if same number of lines)
        if len(index_to_translation) < len(original_segments):
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            # Remove any lines that look like headers or instructions
            lines = [line for line in lines if not line.startswith(('#', '-', '*', '翻译', 'Translation'))]

            if len(lines) == len(original_segments):
                for i, (seg, line) in enumerate(zip(original_segments, lines)):
                    if seg.index not in index_to_translation:
                        # Remove any leading index markers
                        cleaned = re.sub(r'^[\[\(]?\d+[\]\)]?[.:\s]*', '', line).strip()
                        if cleaned:
                            index_to_translation[seg.index] = cleaned

        # Build translated segments
        for seg in original_segments:
            if seg.index in index_to_translation:
                trans_text = index_to_translation[seg.index]
                # Clean up common issues
                trans_text = self._clean_translation(trans_text, seg.text)
                translated.append(
                    SubtitleSegment(
                        index=seg.index,
                        start=seg.start,
                        end=seg.end,
                        text=trans_text,
                    )
                )
            else:
                # Detailed failure analysis
                failure_reason = self._analyze_translation_failure(
                    seg, response, index_to_translation
                )
                logger.warning(
                    f"Segment {seg.index} translation not found: {failure_reason}"
                )

                # Track failed translation for later analysis
                self._failed_translations.append({
                    "index": seg.index,
                    "original": seg.text,
                    "reason": failure_reason,
                    "response_snippet": response[:200] if response else "(empty)",
                })

                translated.append(seg)

        return translated

    def _analyze_translation_failure(
        self,
        segment: SubtitleSegment,
        response: str,
        found_translations: dict,
    ) -> str:
        """Analyze why a translation failed and return a descriptive reason."""
        original_text = segment.text

        # Check for common refusal patterns
        refusal_patterns = [
            ("I cannot", "model_refusal"),
            ("I can't", "model_refusal"),
            ("I'm sorry", "model_refusal"),
            ("不能翻译", "model_refusal"),
            ("无法翻译", "model_refusal"),
            ("不适合", "content_filter"),
            ("inappropriate", "content_filter"),
            ("sensitive", "content_filter"),
            ("违规", "content_filter"),
        ]

        response_lower = response.lower()
        for pattern, reason_type in refusal_patterns:
            if pattern.lower() in response_lower:
                if reason_type == "model_refusal":
                    return f"模型拒绝翻译 (检测到: '{pattern}')"
                else:
                    return f"内容被过滤 (检测到: '{pattern}')"

        # Check if response is empty or too short
        if not response.strip():
            return "LLM返回空响应"

        if len(response.strip()) < 10:
            return f"LLM响应过短: '{response.strip()[:50]}'"

        # Check if the segment index pattern is missing
        if f"[{segment.index}]" not in response and str(segment.index) not in response:
            return f"响应中未找到段落索引 [{segment.index}]"

        # Check if original text might be problematic
        if len(original_text) < 3:
            return f"原文过短: '{original_text}'"

        # Check for interjections that models often skip
        interjection_patterns = ["あぁ", "うん", "ああ", "えっ", "んん", "はぁ"]
        if any(p in original_text for p in interjection_patterns):
            return f"可能是语气词被跳过: '{original_text[:20]}'"

        # Default - parsing issue
        return f"解析失败 (已解析 {len(found_translations)} 个, 原文: '{original_text[:30]}...')"

    def _clean_translation(self, translated: str, original: str) -> str:
        """Clean up translation text."""
        # Remove trailing punctuation duplicates
        translated = translated.strip()

        # If translation is suspiciously short compared to original, might be an error
        if len(translated) < 2 and len(original) > 10:
            return original

        # Remove any remaining index markers at the start
        translated = re.sub(r'^[\[\(]?\d+[\]\)]?\s*', '', translated)

        return translated.strip()

    def translate_batch(
        self,
        segments: List[SubtitleSegment],
        source_lang: str,
        target_lang: str,
        context_before: Optional[List[SubtitleSegment]] = None,
        context_after: Optional[List[SubtitleSegment]] = None,
    ) -> List[SubtitleSegment]:
        """
        Translate a batch of subtitles with context.

        Args:
            segments: Subtitle segments to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            context_before: Previous segments for context.
            context_after: Following segments for context.

        Returns:
            Translated subtitle segments.
        """
        if not segments:
            return []

        prompt = self._build_translation_prompt(
            segments, source_lang, target_lang,
            context_before=context_before,
            context_after=context_after,
        )

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": self.config.temperature},
                )

                translated = self._parse_translation_response(
                    response["message"]["content"],
                    segments,
                )

                # Check for failed translations and retry individually
                failed_segments = [
                    (i, seg) for i, seg in enumerate(translated)
                    if seg.text == segments[i].text  # Translation same as original
                ]

                # If more than half failed, retry individual segments
                if len(failed_segments) > len(segments) // 2:
                    logger.debug(
                        f"Batch had {len(failed_segments)} failures, retrying individually"
                    )
                    translated = self._retry_failed_individually(
                        translated, failed_segments, source_lang, target_lang
                    )

                return translated

            except ResponseError as e:
                logger.warning(
                    f"Translation request failed "
                    f"(attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt == self.config.max_retries - 1:
                    raise TranslationError(f"Translation failed: {e}")

        return segments  # Fallback to original

    def _retry_failed_individually(
        self,
        translated: List[SubtitleSegment],
        failed: List[tuple],
        source_lang: str,
        target_lang: str,
    ) -> List[SubtitleSegment]:
        """Retry translating failed segments one by one."""
        result = list(translated)
        source_name = self.LANGUAGE_NAMES.get(source_lang, source_lang)
        target_name = self.LANGUAGE_NAMES.get(target_lang, target_lang)

        for idx, seg in failed:
            try:
                # Simple single-segment prompt
                simple_prompt = (
                    f"Translate this subtitle from {source_name} to {target_name}. "
                    f"Output ONLY the translation, nothing else:\n\n{seg.text}"
                )

                response = self.client.chat(
                    model=self.config.model,
                    messages=[{"role": "user", "content": simple_prompt}],
                    options={"temperature": self.config.temperature},
                )

                trans_text = response["message"]["content"].strip()

                # Clean up response
                trans_text = self._clean_translation(trans_text, seg.text)

                if trans_text and trans_text != seg.text:
                    result[idx] = SubtitleSegment(
                        index=seg.index,
                        start=seg.start,
                        end=seg.end,
                        text=trans_text,
                    )
                    logger.debug(f"Successfully retried segment {seg.index}")

            except Exception as e:
                logger.debug(f"Individual retry failed for segment {seg.index}: {e}")
                # Keep original on failure

        return result

    def translate(
        self,
        segments: List[SubtitleSegment],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        context_window: int = 3,
    ) -> List[SubtitleSegment]:
        """
        Translate all subtitles with context awareness.

        Args:
            segments: Subtitle segments to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            progress_callback: Progress callback function (completed, total).
            context_window: Number of surrounding segments to include as context.

        Returns:
            Translated subtitle segments.
        """
        if source_lang == target_lang:
            logger.warning("Source and target languages are the same, skipping translation")
            return segments

        logger.info(f"Starting translation: {len(segments)} segments ({source_lang} -> {target_lang})")

        translated_segments = []
        batch_size = self.config.max_batch_size

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]

            # Get context from surrounding segments
            context_before = segments[max(0, i - context_window):i]
            context_after = segments[i + batch_size:i + batch_size + context_window]

            translated_batch = self.translate_batch(
                batch, source_lang, target_lang,
                context_before=context_before,
                context_after=context_after,
            )
            translated_segments.extend(translated_batch)

            if progress_callback:
                progress_callback(len(translated_segments), len(segments))

        # Count and report failures
        failed_count = sum(
            1 for orig, trans in zip(segments, translated_segments)
            if orig.text == trans.text
        )

        if failed_count > 0:
            logger.warning(
                f"Translation complete: {len(translated_segments)} segments, "
                f"{failed_count} failed ({failed_count * 100 // len(segments)}%)"
            )

            # Save failed translations log if enabled
            if self.config.save_failed_log and self._failed_translations:
                self._save_failed_log()
        else:
            logger.info(f"Translation complete: {len(translated_segments)} segments")

        return translated_segments

    def _save_failed_log(self) -> None:
        """Save failed translations to a log file for analysis."""
        import json
        from pathlib import Path
        from datetime import datetime

        # Determine log path
        if self.config.failed_log_path:
            log_path = Path(self.config.failed_log_path)
        else:
            log_path = Path.cwd() / f"translation_failures_{datetime.now():%Y%m%d_%H%M%S}.json"

        try:
            # Categorize failures
            categorized = {
                "model_refusal": [],
                "content_filter": [],
                "parsing_error": [],
                "other": [],
            }

            for failure in self._failed_translations:
                reason = failure.get("reason", "")
                if "拒绝" in reason:
                    categorized["model_refusal"].append(failure)
                elif "过滤" in reason or "content" in reason.lower():
                    categorized["content_filter"].append(failure)
                elif "解析" in reason or "parsing" in reason.lower():
                    categorized["parsing_error"].append(failure)
                else:
                    categorized["other"].append(failure)

            report = {
                "timestamp": datetime.now().isoformat(),
                "model": self.config.model,
                "total_failures": len(self._failed_translations),
                "summary": {
                    "model_refusal": len(categorized["model_refusal"]),
                    "content_filter": len(categorized["content_filter"]),
                    "parsing_error": len(categorized["parsing_error"]),
                    "other": len(categorized["other"]),
                },
                "failures": self._failed_translations,
            }

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"Failed translations log saved: {log_path}")

        except Exception as e:
            logger.error(f"Failed to save translation log: {e}")

    def get_failed_translations(self) -> List[dict]:
        """Get list of failed translations for external analysis."""
        return self._failed_translations.copy()

    def clear_failed_translations(self) -> None:
        """Clear the failed translations list."""
        self._failed_translations = []
