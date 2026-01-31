"""Speech recognition module using faster-whisper and WhisperX."""

from pathlib import Path
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
import logging
import os

from faster_whisper import WhisperModel, BatchedInferencePipeline

from ..models.subtitle import SubtitleSegment, WordTiming
from ..utils.gpu import get_optimal_compute_type, get_available_vram
from ..exceptions import TranscriptionError

logger = logging.getLogger(__name__)

# Check if WhisperX is available
WHISPERX_AVAILABLE = False
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    pass

# Model name to HuggingFace repo mapping
WHISPER_HF_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "Systran/faster-whisper-large-v3-turbo",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}

# Approximate model sizes in bytes for progress display
WHISPER_MODEL_SIZES = {
    "tiny": 75_000_000,       # ~75MB
    "tiny.en": 75_000_000,
    "base": 145_000_000,      # ~145MB
    "base.en": 145_000_000,
    "small": 488_000_000,     # ~488MB
    "small.en": 488_000_000,
    "medium": 1_530_000_000,  # ~1.5GB
    "medium.en": 1_530_000_000,
    "large-v1": 3_100_000_000,  # ~3.1GB
    "large-v2": 3_100_000_000,
    "large-v3": 3_100_000_000,
    "large-v3-turbo": 1_600_000_000,  # ~1.6GB
    "distil-large-v2": 1_500_000_000,
    "distil-large-v3": 1_500_000_000,
}


@dataclass
class TranscriptionInfo:
    """Transcription metadata."""

    language: str
    language_probability: float
    duration: float


class Transcriber:
    """Speech-to-text processor using faster-whisper."""

    # Model VRAM requirements (MB)
    MODEL_VRAM_REQUIREMENTS = {
        "tiny": 1000,
        "base": 1500,
        "small": 2500,
        "medium": 5000,
        "large-v2": 6000,
        "large-v3": 6000,
        "distil-large-v3": 4000,
    }

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: Optional[str] = None,
        download_root: Optional[str] = None,
        use_whisperx: bool = True,
        whisperx_align: bool = True,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize transcriber.

        Args:
            model_name: Whisper model name.
            device: Device type (cuda/cpu).
            compute_type: Compute precision (float16/int8_float16/int8).
            download_root: Model download directory.
            use_whisperx: Use WhisperX for better timestamp accuracy.
            whisperx_align: Enable forced alignment with wav2vec2.
            hf_token: HuggingFace token for pyannote models.
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type or get_optimal_compute_type(device)
        self.download_root = download_root
        self.use_whisperx = use_whisperx and WHISPERX_AVAILABLE
        self.whisperx_align = whisperx_align
        self.hf_token = hf_token

        if use_whisperx and not WHISPERX_AVAILABLE:
            logger.warning(
                "WhisperX not available. Install with: pip install whisperx. "
                "Falling back to faster-whisper."
            )

        self._model: Optional[WhisperModel] = None
        self._batched_pipeline: Optional[BatchedInferencePipeline] = None
        self._whisperx_model = None
        self._whisperx_align_model = None
        self._whisperx_metadata = None

    @classmethod
    def select_optimal_model(cls, prefer_large: bool = True) -> str:
        """
        Select optimal model based on available VRAM.

        Args:
            prefer_large: Prefer larger models if VRAM allows.

        Returns:
            Recommended model name.
        """
        available_vram = get_available_vram()

        if available_vram <= 0:
            logger.warning("Cannot detect GPU VRAM, using small model on CPU")
            return "small"

        # Sort models by VRAM requirement
        sorted_models = sorted(
            cls.MODEL_VRAM_REQUIREMENTS.items(),
            key=lambda x: x[1],
            reverse=prefer_large,
        )

        for model_name, required_vram in sorted_models:
            if available_vram >= required_vram * 1.2:  # 20% headroom
                logger.info(
                    f"Selected model: {model_name} "
                    f"(requires {required_vram}MB, available {available_vram}MB)"
                )
                return model_name

        return "tiny"

    def is_model_cached(self) -> bool:
        """
        Check if Whisper model is already downloaded.

        Returns:
            True if model files exist locally.
        """
        try:
            from huggingface_hub import try_to_load_from_cache

            # Import _CACHED_NO_EXIST separately as it might not exist in older versions
            try:
                from huggingface_hub import _CACHED_NO_EXIST
            except ImportError:
                _CACHED_NO_EXIST = None

            repo_id = WHISPER_HF_REPOS.get(self.model_name, self.model_name)

            # Check for the main model file
            result = try_to_load_from_cache(
                repo_id,
                "model.bin",
                cache_dir=self.download_root,
            )

            # If result is a string path, the file is cached
            if isinstance(result, str):
                return True

            # If _CACHED_NO_EXIST exists and result equals it, file definitely doesn't exist
            if _CACHED_NO_EXIST is not None and result is _CACHED_NO_EXIST:
                return False

            # Otherwise (result is None), file is not cached
            return False

        except Exception as e:
            logger.debug(f"Could not check model cache: {e}")
            # If we can't check, assume not cached
            return False

    def get_model_size(self) -> int:
        """Get approximate model size in bytes."""
        return WHISPER_MODEL_SIZES.get(self.model_name, 1_000_000_000)

    def download_model(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Download Whisper model with progress tracking.

        Args:
            progress_callback: Callback(downloaded_bytes, total_bytes) for progress updates.
        """
        from huggingface_hub import snapshot_download

        repo_id = WHISPER_HF_REPOS.get(self.model_name, self.model_name)

        logger.info(f"Downloading Whisper model: {self.model_name} from {repo_id}")

        try:
            if progress_callback:
                # Create a custom tqdm class that captures progress
                # Must be a proper class (not lambda) because tqdm_class needs class methods like get_lock()
                from tqdm.auto import tqdm as base_tqdm
                import time

                # Use a class factory to inject the callback
                class ProgressTqdm(base_tqdm):
                    """Custom tqdm class to capture download progress with throttling."""

                    # Class-level storage (set before instantiation)
                    _progress_callback = progress_callback
                    _last_update_time = 0.0
                    _update_interval = 0.1  # Throttle: max 10 updates per second

                    def __init__(self, *args, **kwargs):
                        # Disable tqdm's own output to avoid console conflicts with Rich
                        kwargs.setdefault("disable", True)
                        super().__init__(*args, **kwargs)

                    def update(self, n=1):
                        result = super().update(n)
                        if n and ProgressTqdm._progress_callback and self.total:
                            # Throttle callback frequency to prevent flickering
                            now = time.time()
                            if now - ProgressTqdm._last_update_time >= ProgressTqdm._update_interval:
                                ProgressTqdm._last_update_time = now
                                # Use self.n (tqdm's built-in cumulative counter)
                                ProgressTqdm._progress_callback(self.n, self.total)
                        return result

                # Set environment variable to enable tqdm class instantiation
                original_tqdm_disable = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

                try:
                    snapshot_download(
                        repo_id,
                        cache_dir=self.download_root,
                        local_files_only=False,
                        tqdm_class=ProgressTqdm,
                    )
                finally:
                    if original_tqdm_disable is None:
                        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                    else:
                        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = original_tqdm_disable
            else:
                snapshot_download(
                    repo_id,
                    cache_dir=self.download_root,
                    local_files_only=False,
                )

            logger.info(f"Model {self.model_name} downloaded successfully")

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise TranscriptionError(f"Failed to download Whisper model: {e}")

    def ensure_model_downloaded(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """
        Ensure Whisper model is downloaded, with optional progress callback.

        Args:
            progress_callback: Callback(downloaded_bytes, total_bytes) for progress.

        Returns:
            True if model was downloaded, False if already cached.
        """
        if self.is_model_cached():
            logger.info(f"Whisper model {self.model_name} is already cached")
            return False

        self.download_model(progress_callback)
        return True

    def load_model(self) -> None:
        """Load Whisper model."""
        if self._model is not None:
            return

        logger.info(f"Loading Whisper model: {self.model_name} ({self.compute_type})...")

        try:
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )
        except Exception as e:
            raise TranscriptionError(f"Failed to load model: {e}")

        logger.info("Model loaded successfully")

    # Optimized VAD parameters for subtitle timing accuracy
    DEFAULT_VAD_PARAMETERS = {
        "speech_pad_ms": 100,            # Reduced from 400ms to minimize early subtitle appearance
        "min_silence_duration_ms": 500,  # Reduced from 2000ms for better segment breaks
    }

    # Preset VAD modes for different use cases
    VAD_PRESETS = {
        "default": {
            "speech_pad_ms": 100,
            "min_silence_duration_ms": 500,
        },
        "aggressive": {
            "speech_pad_ms": 50,
            "min_silence_duration_ms": 300,
        },
        "relaxed": {
            "speech_pad_ms": 200,
            "min_silence_duration_ms": 800,
        },
        "precise": {
            "speech_pad_ms": 30,
            "min_silence_duration_ms": 200,
        },
    }

    @classmethod
    def get_vad_parameters(
        cls,
        mode: Optional[str] = None,
        speech_pad_ms: Optional[int] = None,
        min_silence_duration_ms: Optional[int] = None,
    ) -> dict:
        """
        Get VAD parameters based on mode or custom values.

        Args:
            mode: Preset mode name ("default", "aggressive", "relaxed", "precise").
            speech_pad_ms: Custom speech padding (overrides mode).
            min_silence_duration_ms: Custom min silence duration (overrides mode).

        Returns:
            VAD parameters dictionary.
        """
        # Start with default or mode preset
        if mode and mode in cls.VAD_PRESETS:
            params = cls.VAD_PRESETS[mode].copy()
        else:
            params = cls.DEFAULT_VAD_PARAMETERS.copy()

        # Override with custom values if provided
        if speech_pad_ms is not None:
            params["speech_pad_ms"] = speech_pad_ms
        if min_silence_duration_ms is not None:
            params["min_silence_duration_ms"] = min_silence_duration_ms

        return params

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True,
        batch_size: Optional[int] = None,
        vad_parameters: Optional[dict] = None,
        post_process: bool = True,
        timestamp_config: Optional[dict] = None,
    ) -> Tuple[List[SubtitleSegment], TranscriptionInfo]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file.
            language: Source language code. Auto-detect if None.
            beam_size: Beam search size.
            vad_filter: Enable VAD filtering.
            word_timestamps: Generate word-level timestamps for better timing.
            batch_size: Batch size for BatchedInferencePipeline.
            vad_parameters: Custom VAD parameters. Uses optimized defaults if None.
            post_process: Enable timestamp post-processing.
            timestamp_config: TimestampProcessor configuration dict.

        Returns:
            Tuple of (subtitle segments, transcription info).
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting transcription: {audio_path.name}")

        # Use WhisperX if available and enabled
        if self.use_whisperx:
            return self._transcribe_whisperx(
                audio_path=audio_path,
                language=language,
                beam_size=beam_size,
                post_process=post_process,
                timestamp_config=timestamp_config,
            )

        # Fall back to faster-whisper
        return self._transcribe_faster_whisper(
            audio_path=audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            batch_size=batch_size,
            vad_parameters=vad_parameters,
            post_process=post_process,
            timestamp_config=timestamp_config,
        )

    def _transcribe_whisperx(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        beam_size: int = 5,
        post_process: bool = True,
        timestamp_config: Optional[dict] = None,
    ) -> Tuple[List[SubtitleSegment], TranscriptionInfo]:
        """Transcribe using WhisperX with forced alignment."""
        import whisperx
        import torch

        logger.info("Using WhisperX for transcription with forced alignment")

        # Fix for PyTorch 2.6+ weights_only security change
        # WhisperX models use omegaconf which needs to be allowlisted
        try:
            from omegaconf import DictConfig, ListConfig
            torch.serialization.add_safe_globals([DictConfig, ListConfig])
        except (ImportError, AttributeError):
            # omegaconf not available or older PyTorch version
            pass

        try:
            # Determine device
            device = self.device
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                logger.warning("CUDA not available, using CPU for WhisperX")

            # Load WhisperX model
            if self._whisperx_model is None:
                logger.info(f"Loading WhisperX model: {self.model_name}")
                self._whisperx_model = whisperx.load_model(
                    self.model_name,
                    device=device,
                    compute_type=self.compute_type,
                    download_root=self.download_root,
                )

            # Load audio
            audio = whisperx.load_audio(str(audio_path))

            # Transcribe
            result = self._whisperx_model.transcribe(
                audio,
                language=language,
                beam_size=beam_size,
            )

            detected_language = result.get("language", language or "en")
            audio_duration = len(audio) / 16000  # WhisperX uses 16kHz

            # Forced alignment for word-level timestamps
            if self.whisperx_align and result.get("segments"):
                logger.info("Performing forced alignment with wav2vec2")
                try:
                    # Load alignment model
                    if self._whisperx_align_model is None or self._whisperx_metadata is None:
                        self._whisperx_align_model, self._whisperx_metadata = whisperx.load_align_model(
                            language_code=detected_language,
                            device=device,
                        )

                    # Align
                    result = whisperx.align(
                        result["segments"],
                        self._whisperx_align_model,
                        self._whisperx_metadata,
                        audio,
                        device,
                        return_char_alignments=False,
                    )
                except Exception as e:
                    logger.warning(f"Forced alignment failed: {e}. Using original timestamps.")

            # Convert to SubtitleSegments
            segments = []
            for i, seg in enumerate(result.get("segments", [])):
                # Extract word-level timestamps if available
                words = None
                if "words" in seg:
                    words = [
                        WordTiming(
                            word=w.get("word", ""),
                            start=w.get("start", 0),
                            end=w.get("end", 0),
                            probability=w.get("score", 1.0),
                        )
                        for w in seg["words"]
                        if "start" in w and "end" in w
                    ]

                # Use word timestamps for more accurate boundaries if available
                if words:
                    start = words[0].start
                    end = words[-1].end
                else:
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)

                segments.append(
                    SubtitleSegment(
                        index=i + 1,
                        start=start,
                        end=end,
                        text=seg.get("text", "").strip(),
                        words=words,
                        confidence=seg.get("score", 1.0) if "score" in seg else 1.0,
                    )
                )

            transcription_info = TranscriptionInfo(
                language=detected_language,
                language_probability=result.get("language_probability", 1.0) if "language_probability" in result else 1.0,
                duration=audio_duration,
            )

            # Apply post-processing
            if post_process:
                segments = self._apply_post_processing(
                    segments, audio_duration, timestamp_config
                )

            logger.info(
                f"WhisperX transcription complete: {len(segments)} segments, "
                f"language: {transcription_info.language}"
            )

            return segments, transcription_info

        except Exception as e:
            logger.error(f"WhisperX transcription failed: {e}")
            raise TranscriptionError(f"WhisperX transcription failed: {e}")

    def _transcribe_faster_whisper(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True,
        batch_size: Optional[int] = None,
        vad_parameters: Optional[dict] = None,
        post_process: bool = True,
        timestamp_config: Optional[dict] = None,
    ) -> Tuple[List[SubtitleSegment], TranscriptionInfo]:
        """Transcribe using faster-whisper."""
        self.load_model()

        logger.info("Using faster-whisper for transcription")

        # Use optimized VAD parameters for better subtitle timing
        vad_params = vad_parameters if vad_parameters is not None else self.DEFAULT_VAD_PARAMETERS

        try:
            # Select inference method
            if batch_size and batch_size > 1:
                if self._batched_pipeline is None:
                    self._batched_pipeline = BatchedInferencePipeline(model=self._model)

                segments_iter, info = self._batched_pipeline.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=beam_size,
                    batch_size=batch_size,
                    vad_filter=vad_filter,
                    word_timestamps=word_timestamps,
                    vad_parameters=vad_params if vad_filter else None,
                )
            else:
                segments_iter, info = self._model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    word_timestamps=word_timestamps,
                    vad_parameters=vad_params if vad_filter else None,
                )

            # Collect all segments with word-level timestamps
            segments = []
            for segment in segments_iter:
                # Extract word-level timestamps if available
                words = None
                if word_timestamps and hasattr(segment, "words") and segment.words:
                    words = [
                        WordTiming(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                        for w in segment.words
                    ]

                # Use word timestamps for more accurate segment boundaries
                if words:
                    start = words[0].start
                    end = words[-1].end
                else:
                    start = segment.start
                    end = segment.end

                segments.append(
                    SubtitleSegment(
                        index=len(segments) + 1,
                        start=start,
                        end=end,
                        text=segment.text.strip(),
                        words=words,
                    )
                )

            audio_duration = info.duration if hasattr(info, "duration") else 0.0

            transcription_info = TranscriptionInfo(
                language=info.language,
                language_probability=info.language_probability,
                duration=audio_duration,
            )

            # Apply post-processing
            if post_process:
                segments = self._apply_post_processing(
                    segments, audio_duration, timestamp_config
                )

            logger.info(
                f"Transcription complete: {len(segments)} segments, "
                f"language: {transcription_info.language} "
                f"(confidence: {transcription_info.language_probability:.2%})"
            )

            return segments, transcription_info

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}")

    def _apply_post_processing(
        self,
        segments: List[SubtitleSegment],
        audio_duration: float,
        timestamp_config: Optional[dict] = None,
    ) -> List[SubtitleSegment]:
        """Apply timestamp post-processing."""
        from .timestamp_processor import TimestampProcessor

        config = timestamp_config or {}
        processor = TimestampProcessor(
            min_duration=config.get("min_duration", 0.5),
            max_duration=config.get("max_duration", 8.0),
            min_gap=config.get("min_gap", 0.05),
            max_gap_warning=config.get("max_gap_warning", 2.0),
            chars_per_second=config.get("chars_per_second", 15.0),
        )

        return processor.process(segments, audio_duration)

    def unload_model(self) -> None:
        """Unload model to free VRAM."""
        self._model = None
        self._batched_pipeline = None
        self._whisperx_model = None
        self._whisperx_align_model = None
        self._whisperx_metadata = None

        import gc

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
