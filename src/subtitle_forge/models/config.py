"""Configuration data model."""

import os
import sys
import tempfile
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
    # VAD tuned for subtitle timing: 250ms padding (silero default 400) keeps subtitles tight
    # without eating word onsets; 700ms min silence avoids merging adjacent utterances in fast
    # dialogue while still bridging natural breath pauses.
    speech_pad_ms: int = 250
    min_silence_duration_ms: int = 700
    # WhisperX options
    use_whisperx: bool = True  # Use WhisperX for better timestamp accuracy
    whisperx_align: bool = True  # Enable forced alignment with wav2vec2
    hf_token: Optional[str] = None  # HuggingFace token for gated/private model downloads
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

        config = cls(
            whisper=WhisperConfig(**data.get("whisper", {})),
            ollama=OllamaConfig(**data.get("ollama", {})),
            output=OutputConfig(**data.get("output", {})),
            timestamp=TimestampConfig(**data.get("timestamp", {})),
            max_workers=data.get("max_workers", 2),
            log_level=data.get("log_level", "INFO"),
            log_file=data.get("log_file"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Raise ValueError listing every invalid field value.

        Called from load() (and by `config set` before saving) so a bad value
        fails before any work starts — chars_per_second=0 used to save fine
        and surface as a ZeroDivisionError mid-transcription.
        """

        def _num(value) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        ts, wh, ol = self.timestamp, self.whisper, self.ollama
        errors = []

        if ts.mode not in ("off", "minimal", "full"):
            errors.append(f"timestamp.mode must be one of off/minimal/full, got {ts.mode!r}")
        if wh.device not in ("cuda", "cpu", "auto"):
            errors.append(f"whisper.device must be one of cuda/cpu/auto, got {wh.device!r}")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append(
                f"log_level must be one of DEBUG/INFO/WARNING/ERROR/CRITICAL, got {self.log_level!r}"
            )

        positive = [
            ("timestamp.min_duration", ts.min_duration),
            ("timestamp.max_duration", ts.max_duration),
            ("timestamp.max_gap_warning", ts.max_gap_warning),
            ("timestamp.chars_per_second", ts.chars_per_second),
            ("timestamp.cjk_chars_per_second", ts.cjk_chars_per_second),
            ("ollama.request_timeout", ol.request_timeout),
        ]
        non_negative = [
            ("timestamp.min_gap", ts.min_gap),
            ("timestamp.lead_in_ms", ts.lead_in_ms),
            ("timestamp.linger_ms", ts.linger_ms),
            ("whisper.speech_pad_ms", wh.speech_pad_ms),
            ("whisper.min_silence_duration_ms", wh.min_silence_duration_ms),
            ("ollama.temperature", ol.temperature),
        ]
        # max_retries >= 1 matters: the translator's attempt loop is
        # `range(max_retries)`, so 0 would silently skip translation entirely.
        at_least_one = [
            ("timestamp.split_threshold", ts.split_threshold),
            ("whisper.beam_size", wh.beam_size),
            ("ollama.max_batch_size", ol.max_batch_size),
            ("ollama.max_retries", ol.max_retries),
            ("max_workers", self.max_workers),
        ]
        for name, value in positive:
            if not _num(value) or value <= 0:
                errors.append(f"{name} must be a number > 0, got {value!r}")
        for name, value in non_negative:
            if not _num(value) or value < 0:
                errors.append(f"{name} must be a number >= 0, got {value!r}")
        for name, value in at_least_one:
            if not _num(value) or value < 1:
                errors.append(f"{name} must be an integer >= 1, got {value!r}")
        if wh.batch_size is not None and (not _num(wh.batch_size) or wh.batch_size < 1):
            errors.append(f"whisper.batch_size must be an integer >= 1 or null, got {wh.batch_size!r}")
        if _num(ts.min_duration) and _num(ts.max_duration) and ts.max_duration < ts.min_duration:
            errors.append(
                f"timestamp.max_duration ({ts.max_duration}) must be >= "
                f"timestamp.min_duration ({ts.min_duration})"
            )

        if errors:
            raise ValueError("Invalid configuration:\n  " + "\n  ".join(errors))

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

        content = yaml.safe_dump(
            asdict(self), default_flow_style=False, allow_unicode=True, sort_keys=False
        )

        # Same-directory temp file, then replace: a crash mid-write must not
        # leave a truncated config. 0600 because the file can hold hf_token in
        # plaintext, and the umask default 0644 is world-readable on POSIX.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, str(path))
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
