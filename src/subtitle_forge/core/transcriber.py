"""Speech recognition module using faster-whisper."""

from pathlib import Path
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
import logging
import os

from faster_whisper import WhisperModel, BatchedInferencePipeline

from ..models.subtitle import SubtitleSegment
from ..utils.gpu import get_optimal_compute_type, get_available_vram
from ..exceptions import TranscriptionError

logger = logging.getLogger(__name__)

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
    ):
        """
        Initialize transcriber.

        Args:
            model_name: Whisper model name.
            device: Device type (cuda/cpu).
            compute_type: Compute precision (float16/int8_float16/int8).
            download_root: Model download directory.
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type or get_optimal_compute_type(device)
        self.download_root = download_root

        self._model: Optional[WhisperModel] = None
        self._batched_pipeline: Optional[BatchedInferencePipeline] = None

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
            from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

            repo_id = WHISPER_HF_REPOS.get(self.model_name, self.model_name)

            # Check for the main model file
            result = try_to_load_from_cache(
                repo_id,
                "model.bin",
                cache_dir=self.download_root,
            )

            return result is not None and result != _CACHED_NO_EXIST

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
        from tqdm import tqdm

        repo_id = WHISPER_HF_REPOS.get(self.model_name, self.model_name)
        model_size = self.get_model_size()

        logger.info(f"Downloading Whisper model: {self.model_name} from {repo_id}")

        class DownloadProgressCallback(tqdm):
            """Custom tqdm class to capture download progress."""

            def __init__(self, *args, callback=None, **kwargs):
                super().__init__(*args, **kwargs)
                self._callback = callback
                self._downloaded = 0

            def update(self, n=1):
                super().update(n)
                self._downloaded += n
                if self._callback and self.total:
                    self._callback(self._downloaded, self.total)

        try:
            # Use snapshot_download with custom tqdm class for progress
            if progress_callback:
                # Set environment variable to enable progress bars
                original_tqdm_disable = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

                try:
                    snapshot_download(
                        repo_id,
                        cache_dir=self.download_root,
                        local_files_only=False,
                        tqdm_class=lambda *args, **kwargs: DownloadProgressCallback(
                            *args, callback=progress_callback, **kwargs
                        ),
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

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = False,
        batch_size: Optional[int] = None,
    ) -> Tuple[List[SubtitleSegment], TranscriptionInfo]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file.
            language: Source language code. Auto-detect if None.
            beam_size: Beam search size.
            vad_filter: Enable VAD filtering.
            word_timestamps: Generate word-level timestamps.
            batch_size: Batch size for BatchedInferencePipeline.

        Returns:
            Tuple of (subtitle segments, transcription info).
        """
        self.load_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting transcription: {audio_path.name}")

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
                )
            else:
                segments_iter, info = self._model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    word_timestamps=word_timestamps,
                )

            # Collect all segments
            segments = []
            for segment in segments_iter:
                segments.append(
                    SubtitleSegment(
                        index=len(segments) + 1,
                        start=segment.start,
                        end=segment.end,
                        text=segment.text.strip(),
                    )
                )

            transcription_info = TranscriptionInfo(
                language=info.language,
                language_probability=info.language_probability,
                duration=info.duration if hasattr(info, "duration") else 0.0,
            )

            logger.info(
                f"Transcription complete: {len(segments)} segments, "
                f"language: {transcription_info.language} "
                f"(confidence: {transcription_info.language_probability:.2%})"
            )

            return segments, transcription_info

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}")

    def unload_model(self) -> None:
        """Unload model to free VRAM."""
        self._model = None
        self._batched_pipeline = None

        import gc

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
