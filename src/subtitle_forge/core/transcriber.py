"""Speech recognition module using faster-whisper."""

from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import logging

from faster_whisper import WhisperModel, BatchedInferencePipeline

from ..models.subtitle import SubtitleSegment
from ..utils.gpu import get_optimal_compute_type, get_available_vram
from ..exceptions import TranscriptionError

logger = logging.getLogger(__name__)


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
