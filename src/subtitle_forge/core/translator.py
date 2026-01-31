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
        """Parse translation response."""
        translated = []

        # Extract translations by index
        pattern = r"\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)"
        matches = re.findall(pattern, response, re.DOTALL)

        index_to_translation = {}
        for idx_str, text in matches:
            idx = int(idx_str)
            index_to_translation[idx] = text.strip()

        # Build translated segments
        for seg in original_segments:
            if seg.index in index_to_translation:
                translated.append(
                    SubtitleSegment(
                        index=seg.index,
                        start=seg.start,
                        end=seg.end,
                        text=index_to_translation[seg.index],
                    )
                )
            else:
                logger.warning(f"Segment {seg.index} translation not found, keeping original")
                translated.append(seg)

        return translated

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

                return translated

            except ResponseError as e:
                logger.warning(
                    f"Translation request failed "
                    f"(attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt == self.config.max_retries - 1:
                    raise TranslationError(f"Translation failed: {e}")

        return segments  # Fallback to original

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

        logger.info(f"Translation complete: {len(translated_segments)} segments")
        return translated_segments
