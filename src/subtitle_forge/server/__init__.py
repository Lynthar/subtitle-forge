"""HTTP server exposing subtitle-forge as a job-based API."""

from .app import create_app

__all__ = ["create_app"]
