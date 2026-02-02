"""Configuration data model."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class TimestampConfig:
    """Timestamp post-processing configuration."""

    enabled: bool = True  # Enable timestamp post-processing
    mode: str = "minimal"  # Processing mode: "off", "minimal", "full"
    min_duration: float = 0.5  # Minimum subtitle duration (seconds)
    max_duration: float = 8.0  # Maximum subtitle duration (seconds)
    min_gap: float = 0.05  # Minimum gap between subtitles (seconds)
    max_gap_warning: float = 50.0  # Gap threshold for missed speech warning (seconds)
    chars_per_second: float = 15.0  # Reading speed for Western languages
    cjk_chars_per_second: float = 10.0  # Reading speed for CJK languages
    split_threshold: int = 30  # Minimum characters before attempting split
    split_sentences: bool = True  # Split segments by sentence using word timestamps


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
    # VAD parameters for subtitle timing optimization
    speech_pad_ms: int = 100  # Padding around detected speech (ms)
    min_silence_duration_ms: int = 500  # Minimum silence duration for segment breaks (ms)
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
    temperature: float = 0.3
    max_batch_size: int = 10
    max_retries: int = 3
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
        """Get user config file path."""
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
        """Save configuration to YAML file."""
        if path is None:
            path = self.get_config_path()

        path.parent.mkdir(parents=True, exist_ok=True)

        ollama_data = {
            "model": self.ollama.model,
            "host": self.ollama.host,
            "temperature": self.ollama.temperature,
            "max_batch_size": self.ollama.max_batch_size,
        }
        if self.ollama.prompt_template:
            ollama_data["prompt_template"] = self.ollama.prompt_template
        if self.ollama.prompt_template_id:
            ollama_data["prompt_template_id"] = self.ollama.prompt_template_id

        whisper_data = {
            "model": self.whisper.model,
            "device": self.whisper.device,
            "compute_type": self.whisper.compute_type,
            "beam_size": self.whisper.beam_size,
            "vad_filter": self.whisper.vad_filter,
            "speech_pad_ms": self.whisper.speech_pad_ms,
            "min_silence_duration_ms": self.whisper.min_silence_duration_ms,
            "use_whisperx": self.whisper.use_whisperx,
            "whisperx_align": self.whisper.whisperx_align,
        }
        if self.whisper.hf_token:
            whisper_data["hf_token"] = self.whisper.hf_token
        if self.whisper.hf_endpoint:
            whisper_data["hf_endpoint"] = self.whisper.hf_endpoint

        data = {
            "whisper": whisper_data,
            "ollama": ollama_data,
            "output": {
                "encoding": self.output.encoding,
                "keep_original": self.output.keep_original,
                "bilingual": self.output.bilingual,
                "original_on_top": self.output.original_on_top,
            },
            "timestamp": {
                "enabled": self.timestamp.enabled,
                "mode": self.timestamp.mode,
                "min_duration": self.timestamp.min_duration,
                "max_duration": self.timestamp.max_duration,
                "min_gap": self.timestamp.min_gap,
                "max_gap_warning": self.timestamp.max_gap_warning,
                "chars_per_second": self.timestamp.chars_per_second,
                "cjk_chars_per_second": self.timestamp.cjk_chars_per_second,
                "split_threshold": self.timestamp.split_threshold,
                "split_sentences": self.timestamp.split_sentences,
            },
            "max_workers": self.max_workers,
            "log_level": self.log_level,
        }

        if self.log_file:
            data["log_file"] = self.log_file

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        ollama_dict = {
            "model": self.ollama.model,
            "host": self.ollama.host,
            "temperature": self.ollama.temperature,
            "max_batch_size": self.ollama.max_batch_size,
        }
        if self.ollama.prompt_template:
            ollama_dict["prompt_template"] = self.ollama.prompt_template
        if self.ollama.prompt_template_id:
            ollama_dict["prompt_template_id"] = self.ollama.prompt_template_id

        return {
            "whisper": {
                "model": self.whisper.model,
                "device": self.whisper.device,
                "compute_type": self.whisper.compute_type,
                "beam_size": self.whisper.beam_size,
                "vad_filter": self.whisper.vad_filter,
                "speech_pad_ms": self.whisper.speech_pad_ms,
                "min_silence_duration_ms": self.whisper.min_silence_duration_ms,
                "use_whisperx": self.whisper.use_whisperx,
                "whisperx_align": self.whisper.whisperx_align,
            },
            "ollama": ollama_dict,
            "output": {
                "encoding": self.output.encoding,
                "keep_original": self.output.keep_original,
                "bilingual": self.output.bilingual,
            },
            "timestamp": {
                "enabled": self.timestamp.enabled,
                "mode": self.timestamp.mode,
                "min_duration": self.timestamp.min_duration,
                "max_duration": self.timestamp.max_duration,
                "min_gap": self.timestamp.min_gap,
                "max_gap_warning": self.timestamp.max_gap_warning,
                "chars_per_second": self.timestamp.chars_per_second,
                "cjk_chars_per_second": self.timestamp.cjk_chars_per_second,
                "split_threshold": self.timestamp.split_threshold,
                "split_sentences": self.timestamp.split_sentences,
            },
            "max_workers": self.max_workers,
            "log_level": self.log_level,
        }
