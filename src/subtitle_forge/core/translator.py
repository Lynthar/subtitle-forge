"""Translation module using Ollama."""

from typing import List, Optional, Callable
from dataclasses import dataclass
import json
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
    # Translation is deterministic — temperature=0 sharply reduces dropped
    # / renumbered segments. Raise only for explicit creative paraphrasing.
    temperature: float = 0.0
    max_batch_size: int = 10
    max_retries: int = 3
    request_timeout: float = 180.0  # Per-request timeout (seconds)
    prompt_template: Optional[str] = None  # Custom prompt template (None = use default)
    prompt_template_id: Optional[str] = None  # Reference to prompt library template
    save_failed_log: bool = False  # Save failed translations to a log file
    failed_log_path: Optional[str] = None  # Path for failed translations log


class SubtitleTranslator:
    """Subtitle translator using Ollama LLM."""

    # Default JSON prompt template. Used when the user has NOT picked a
    # library template — drives Ollama's `format="json"` mode for reliable
    # parsing. The legacy [N]-line text format is still used when a custom
    # or library template is selected (see _is_json_mode / _parse_response).
    DEFAULT_PROMPT_TEMPLATE = """You are a professional subtitle translator. Translate dialogue from {source_lang} into {target_lang}.

GUIDELINES
- Preserve natural conversational tone, emotional nuance, and register.
- Keep translations concise for subtitle readability.
- For very short utterances (interjections, "uh", "嗯", "ah"), still translate them — do NOT drop them.
- Treat the context blocks as background only; never include them in your output.
{context_before}
SUBTITLES TO TRANSLATE — every key in the JSON below MUST appear in your output:
{segments}
{context_after}
OUTPUT — return STRICTLY a JSON object of the form:
{{"translations": {{"<index>": "<translated text>", ...}}}}

Do not include any text outside the JSON object. Do not omit any indices. Do not renumber.
"""

    # Legacy [N]-line prompt — kept for prompt_library templates that still
    # rely on the original format. Available placeholders are the same set:
    # {source_lang}, {target_lang}, {context_before}, {segments}, {context_after}
    LEGACY_PROMPT_TEMPLATE = """You are an expert subtitle translator for movies and TV dramas.

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
        self._batch_failure_indices: dict = {}  # Track failed indices per reason for compressed logging

    @property
    def client(self) -> Client:
        if self._client is None:
            # httpx is a transitive dependency of ollama-python, so it's
            # always installed when ollama is — but defer the import so the
            # module can still load in environments that haven't pip
            # installed dependencies yet (e.g. during static checks).
            try:
                import httpx
                timeout = httpx.Timeout(self.config.request_timeout, connect=10.0)
            except ImportError:
                timeout = self.config.request_timeout
            self._client = Client(host=self.config.host, timeout=timeout)
        return self._client

    def _is_json_mode(self) -> bool:
        """
        Whether to drive the LLM in JSON mode + parse JSON output.

        We only enable JSON mode for the default prompt — library / custom
        templates were authored against the legacy [N]-line format and would
        misbehave under format="json".
        """
        return (
            self.config.prompt_template is None
            and self.config.prompt_template_id is None
        )

    def _effective_batch_size(self) -> int:
        """
        Cap batch size based on model capability.

        Larger batches degrade translation reliability for smaller models
        (the LLM "forgets" segments mid-batch). Hard cap below the user's
        configured size for 7B / 14B class models. 32B+ uses the user value.
        """
        cfg_max = max(1, self.config.max_batch_size)
        model = self.config.model.lower()
        # 32B / 70B / 72B and similarly large — trust user config.
        if any(tag in model for tag in (":32b", ":34b", ":70b", ":72b", ":110b")):
            return cfg_max
        # 14B class — clamp to 6.
        if ":14b" in model or ":13b" in model:
            return min(cfg_max, 6)
        # Anything 8B and below — clamp to 4.
        return min(cfg_max, 4)

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

        In JSON mode (default), context is rendered as prose so the LLM cannot
        confuse it with the indexed segments-to-translate. In legacy mode the
        original [N] format is preserved for backward compatibility with
        prompt-library templates.
        """
        source_name = self.LANGUAGE_NAMES.get(source_lang, source_lang)
        target_name = self.LANGUAGE_NAMES.get(target_lang, target_lang)
        json_mode = self._is_json_mode()

        if json_mode:
            # Prose context — visually distinct from the JSON-shaped segments
            # block, so the LLM can't accidentally include context indices in
            # its output.
            context_before_text = ""
            if context_before:
                snippet = " ".join(seg.text.strip() for seg in context_before[-3:])
                context_before_text = (
                    f'\nEARLIER DIALOGUE (background only, do NOT include in output): "{snippet}"\n'
                )

            context_after_text = ""
            if context_after:
                snippet = " ".join(seg.text.strip() for seg in context_after[:2])
                context_after_text = (
                    f'\nLATER DIALOGUE (background only, do NOT include in output): "{snippet}"\n'
                )

            # Segments rendered as a JSON object — primes the model to
            # respond with a JSON object of the same shape.
            seg_obj = {str(seg.index): seg.text for seg in segments}
            segments_text = json.dumps(seg_obj, ensure_ascii=False, indent=2)
        else:
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

            lines = [f"[{seg.index}] {seg.text}" for seg in segments]
            segments_text = chr(10).join(lines)

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
        """Parse translation response. Tries JSON first, falls back to regex."""
        translated = []

        # Log raw response for debugging (truncated)
        response_preview = response[:500] + "..." if len(response) > 500 else response
        logger.debug(f"Raw LLM response:\n{response_preview}")

        index_to_translation: dict = {}

        # Strategy 1: JSON parsing (the new default path).
        # The LLM may wrap the JSON in markdown fences or add a stray prefix
        # despite format="json"; pull out the first {...} block and parse it.
        if not index_to_translation:
            json_blob = self._extract_json_object(response)
            if json_blob is not None:
                try:
                    parsed = json.loads(json_blob)
                    translations = parsed.get("translations") if isinstance(parsed, dict) else None
                    if isinstance(translations, dict):
                        for key, value in translations.items():
                            try:
                                idx = int(key)
                            except (TypeError, ValueError):
                                continue
                            if isinstance(value, str) and value.strip():
                                index_to_translation[idx] = value.strip()
                    elif isinstance(parsed, dict):
                        # Some models return {"5": "...", "6": "..."} directly
                        for key, value in parsed.items():
                            try:
                                idx = int(key)
                            except (TypeError, ValueError):
                                continue
                            if isinstance(value, str) and value.strip():
                                index_to_translation[idx] = value.strip()
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parse failed, falling back to regex: {e}")

        # Strategy 2: [N] prefixed lines (legacy format, also catches stray
        # markdown like **[1]** text)
        if len(index_to_translation) < len(original_segments):
            pattern1 = r"\[(\d+)\]\s*[:：]?\s*(.+?)(?=\[\d+\]|$)"
            matches = re.findall(pattern1, response, re.DOTALL)
            for idx_str, text in matches:
                idx = int(idx_str)
                text = text.strip()
                if text and idx not in index_to_translation:
                    index_to_translation[idx] = text

        # Strategy 3: bare numbered lines — "1.", "1:", "1)", "(1)"
        if len(index_to_translation) < len(original_segments):
            pattern2 = r"(?:^|\n)\s*[\(\[]?(\d+)[\]\)]?\s*[.:：\)]\s*(.+?)(?=\n\s*[\(\[]?\d+[\]\)]?\s*[.:：\)]|$)"
            matches = re.findall(pattern2, response, re.DOTALL)
            for idx_str, text in matches:
                idx = int(idx_str)
                text = text.strip()
                if text and idx not in index_to_translation:
                    index_to_translation[idx] = text

        # Strategy 4: positional fallback — same number of non-empty lines
        if len(index_to_translation) < len(original_segments):
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            lines = [line for line in lines if not line.startswith(('#', '-', '*', '翻译', 'Translation'))]

            if len(lines) == len(original_segments):
                for i, (seg, line) in enumerate(zip(original_segments, lines)):
                    if seg.index not in index_to_translation:
                        cleaned = re.sub(r'^[\[\(]?\d+[\]\)]?[.:：\s]*', '', line).strip()
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

                # Track failed indices by reason (for compressed batch logging)
                if failure_reason not in self._batch_failure_indices:
                    self._batch_failure_indices[failure_reason] = []
                self._batch_failure_indices[failure_reason].append(seg.index)

                # Track failed translation for later analysis/log file
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
                    return "模型拒绝翻译"
                else:
                    return "内容被过滤"

        # Check if response is empty or too short
        if not response.strip():
            return "LLM返回空响应"

        if len(response.strip()) < 10:
            return "LLM响应过短"

        # Check if the segment index pattern is missing
        if f"[{segment.index}]" not in response and str(segment.index) not in response:
            return "响应中未找到段落索引"

        # Check if original text might be problematic
        if len(original_text) < 3:
            return "原文过短"

        # Check for interjections that models often skip
        interjection_patterns = ["あぁ", "うん", "ああ", "えっ", "んん", "はぁ"]
        if any(p in original_text for p in interjection_patterns):
            return "可能是语气词被跳过"

        # Default - parsing issue
        return "解析失败"

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

    @staticmethod
    def _extract_json_object(response: str) -> Optional[str]:
        """
        Extract the first balanced JSON object from a response string.

        Handles markdown fences (```json ... ```) and stray prefixes like
        "Here is the translation:" that some models emit despite format="json".
        Returns None if no balanced object is found.
        """
        if not response:
            return None

        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            # Drop the opening fence (and optional language tag)
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            # Drop trailing fence
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]

        # Walk the string finding a balanced { ... } block, respecting strings
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

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

        # Reset batch failure indices for this batch
        self._batch_failure_indices = {}

        prompt = self._build_translation_prompt(
            segments, source_lang, target_lang,
            context_before=context_before,
            context_after=context_after,
        )

        json_mode = self._is_json_mode()
        chat_kwargs = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": self.config.temperature},
        }
        if json_mode:
            # Ollama's structured-output mode — guarantees parseable JSON
            # whenever the model honours the constraint.
            chat_kwargs["format"] = "json"

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat(**chat_kwargs)

                translated = self._parse_translation_response(
                    response["message"]["content"],
                    segments,
                )

                # Log compressed summary of failures for this batch.
                # INFO not WARNING: batch-level issues are usually fixed by
                # the individual retry that runs immediately after — the
                # final WARNING (if any) is emitted from translate() once
                # all batches have settled.
                if self._batch_failure_indices:
                    summary_parts = []
                    for reason, indices in self._batch_failure_indices.items():
                        if len(indices) <= 5:
                            idx_str = ", ".join(str(i) for i in indices)
                        else:
                            # Show first 3 and last 2 with ellipsis
                            idx_str = f"{indices[0]}, {indices[1]}, {indices[2]}...{indices[-2]}, {indices[-1]}"
                        summary_parts.append(f"{reason} [{idx_str}]")
                    logger.info(f"Batch parse issues, retrying individually: {'; '.join(summary_parts)}")

                # Check for failed translations and retry individually.
                # Detection: translated text is identical to the original AND
                # the source/target are not the same language (which would
                # make identical text a legitimate outcome).
                failed_segments = [
                    (i, seg) for i, seg in enumerate(translated)
                    if seg.text == segments[i].text
                    and source_lang != target_lang
                ]

                # Retry ANY failures individually — the prior "only if >50%
                # failed" threshold left small batches with 1–2 untranslated
                # segments unfixed, which is the most common failure mode in
                # practice.
                if failed_segments:
                    logger.debug(
                        f"Batch had {len(failed_segments)} failure(s), retrying individually"
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
            except Exception as e:
                # Catches httpx.TimeoutException + transport errors without
                # a hard import-time dep on httpx. ResponseError is handled
                # above for retry-on-server-error semantics.
                err_name = type(e).__name__
                if "Timeout" not in err_name and "Connect" not in err_name:
                    raise
                logger.warning(
                    f"Translation request failed ({err_name}) "
                    f"(attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt == self.config.max_retries - 1:
                    raise TranslationError(f"Translation failed after {self.config.max_retries} attempts: {e}")

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

        batch_size = self._effective_batch_size()
        if batch_size != self.config.max_batch_size:
            logger.info(
                f"Auto-clamped translation batch size: {self.config.max_batch_size} -> "
                f"{batch_size} (model: {self.config.model})"
            )

        logger.info(
            f"Starting translation: {len(segments)} segments ({source_lang} -> {target_lang}), "
            f"batch_size={batch_size}"
        )

        translated_segments = []

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
