"""Custom exceptions for subtitle-forge."""


class SubtitleForgeError(Exception):
    """Base exception class."""

    pass


class AudioExtractionError(SubtitleForgeError):
    """Audio extraction error."""

    pass


class TranscriptionError(SubtitleForgeError):
    """Speech recognition error."""

    pass


class TranslationError(SubtitleForgeError):
    """Translation error."""

    pass


class SubtitleError(SubtitleForgeError):
    """Subtitle processing error."""

    pass


class ConfigError(SubtitleForgeError):
    """Configuration error."""

    pass


class ModelNotFoundError(SubtitleForgeError):
    """Model not found error."""

    pass
