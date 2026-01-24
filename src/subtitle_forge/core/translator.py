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


class SubtitleTranslator:
    """Subtitle translator using Ollama LLM."""

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
        try:
            models = self.client.list()
            available_models = [m["name"] for m in models.get("models", [])]

            model_base = self.config.model.split(":")[0]
            for available in available_models:
                if self.config.model == available or model_base in available:
                    return True

            logger.warning(
                f"Model {self.config.model} not available. "
                f"Available models: {', '.join(available_models)}"
            )
            return False

        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False

    def _build_translation_prompt(
        self,
        segments: List[SubtitleSegment],
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Build translation prompt with structured format."""
        source_name = self.LANGUAGE_NAMES.get(source_lang, source_lang)
        target_name = self.LANGUAGE_NAMES.get(target_lang, target_lang)

        lines = [f"[{seg.index}] {seg.text}" for seg in segments]

        prompt = f"""You are a professional subtitle translator. Translate the following subtitle lines from {source_name} to {target_name}.

IMPORTANT RULES:
1. Preserve the [number] prefix for each line
2. Keep translations concise and suitable for subtitles
3. Maintain the original meaning and tone
4. Do not add explanations or notes
5. Output ONLY the translated lines, nothing else

Source subtitles:
{chr(10).join(lines)}

Translated subtitles:"""

        return prompt

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
    ) -> List[SubtitleSegment]:
        """
        Translate a batch of subtitles.

        Args:
            segments: Subtitle segments to translate.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            Translated subtitle segments.
        """
        if not segments:
            return []

        prompt = self._build_translation_prompt(segments, source_lang, target_lang)

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
    ) -> List[SubtitleSegment]:
        """
        Translate all subtitles.

        Args:
            segments: Subtitle segments to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            progress_callback: Progress callback function (completed, total).

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
            batch = segments[i : i + batch_size]
            translated_batch = self.translate_batch(batch, source_lang, target_lang)
            translated_segments.extend(translated_batch)

            if progress_callback:
                progress_callback(len(translated_segments), len(segments))

        logger.info(f"Translation complete: {len(translated_segments)} segments")
        return translated_segments
