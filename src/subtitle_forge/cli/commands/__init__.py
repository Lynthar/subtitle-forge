"""CLI commands."""

from ...models.config import ASR_BACKENDS, TRANSLATION_BACKENDS

ASR_BACKEND_HELP = f"Speech recognition backend: {', '.join(ASR_BACKENDS)} (default: config)"
TRANSLATION_BACKEND_HELP = (
    f"Translation backend: {', '.join(TRANSLATION_BACKENDS)} (default: config translation.backend)"
)
