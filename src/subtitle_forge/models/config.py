"""Configuration data model."""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml


@dataclass
class TimestampConfig:
    """Timestamp post-processing configuration."""

    enabled: bool = True  # Enable timestamp post-processing
    mode: str = "minimal"  # Processing mode: "off", "minimal", "full"
    min_duration: float = 1.0  # Minimum subtitle duration (seconds)
    max_duration: float = 8.0  # Maximum subtitle duration (seconds)
    min_gap: float = 0.05  # Minimum gap between subtitles (seconds)
    max_gap_warning: float = 10.0  # Gap threshold for missed speech warning (seconds)
    chars_per_second: float = 15.0  # Reading speed for Western languages
    cjk_chars_per_second: float = 10.0  # Reading speed for CJK languages
    split_threshold: int = 30  # Minimum characters before attempting split
    split_sentences: bool = True  # Split segments by sentence using word timestamps
    # Lead-in and linger to compensate for acoustic-alignment timestamps not being
    # the same thing as on-screen subtitle timing. Without these, subtitles tend
    # to feel like they "chase" the audio and disappear too fast.
    lead_in_ms: int = 80   # Show subtitle this many ms BEFORE the first word's onset
    linger_ms: int = 300   # Keep subtitle this many ms AFTER the last word's offset


@dataclass
class WhisperConfig:
    """Whisper transcription configuration."""

    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True
    batch_size: Optional[int] = None
    download_root: Optional[str] = None
    # VAD parameters for subtitle timing optimization.
    # 250ms padding (vs silero default 400ms) keeps subtitles tight without
    # eating word onsets. 700ms min silence avoids merging adjacent utterances
    # in fast dialogue while still letting the VAD bridge natural breath pauses.
    speech_pad_ms: int = 250
    min_silence_duration_ms: int = 700
    # WhisperX options
    use_whisperx: bool = True  # Use WhisperX for better timestamp accuracy
    whisperx_align: bool = True  # Enable forced alignment with wav2vec2
    hf_token: Optional[str] = None  # HuggingFace token for pyannote models
    # HuggingFace mirror for users in regions with limited access
    hf_endpoint: Optional[str] = None  # e.g., "https://hf-mirror.com"


@dataclass
class OllamaConfig:
    """Ollama translation configuration."""

    model: str = "qwen2.5:14b"
    host: str = "http://localhost:11434"
    # Translation is a deterministic task — temperature=0 dramatically reduces
    # the rate of skipped/renumbered segments compared to 0.3.
    temperature: float = 0.0
    max_batch_size: int = 10
    max_retries: int = 3
    request_timeout: float = 180.0  # Per-request timeout (seconds)
    prompt_template: Optional[str] = None  # Custom translation prompt (None = use default)
    prompt_template_id: Optional[str] = None  # Reference to prompt library template


@dataclass
class OutputConfig:
    """Output configuration."""

    encoding: str = "utf-8"
    keep_original: bool = True
    bilingual: bool = False
    original_on_top: bool = True


@dataclass
class AppConfig:
    """Application configuration."""

    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    timestamp: TimestampConfig = field(default_factory=TimestampConfig)
    max_workers: int = 2
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def get_config_path(cls) -> Path:
        """Get user config file path.

        Windows uses the %APPDATA% location the user guide documents;
        everything else uses ~/.config.
        """
        if sys.platform == "win32":
            appdata = os.environ.get("APPDATA")
            if appdata:
                return Path(appdata) / "subtitle-forge" / "config.yaml"
        return Path.home() / ".config" / "subtitle-forge" / "config.yaml"

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        """Load configuration from YAML file."""
        if path is None:
            path = cls.get_config_path()

        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            whisper=WhisperConfig(**data.get("whisper", {})),
            ollama=OllamaConfig(**data.get("ollama", {})),
            output=OutputConfig(**data.get("output", {})),
            timestamp=TimestampConfig(**data.get("timestamp", {})),
            max_workers=data.get("max_workers", 2),
            log_level=data.get("log_level", "INFO"),
            log_file=data.get("log_file"),
        )

    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to YAML file.

        Serializes EVERY dataclass field via ``asdict`` so hand-edited values
        survive a rewrite. The previous version used a hand-maintained whitelist
        that silently omitted several fields (e.g. ``ollama.request_timeout`` /
        ``max_retries``, ``whisper.batch_size`` / ``download_root``), so running
        any ``config set`` erased them and reverted them to defaults — a
        data-loss bug. Field order matches the dataclass definitions.
        """
        if path is None:
            path = self.get_config_path()

        path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
