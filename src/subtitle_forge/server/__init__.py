"""HTTP server exposing subtitle-forge as a job-based API.

`create_app` is intentionally not re-exported here — importing it eagerly
would chain into the transcribe pipeline (ffmpeg, faster-whisper, ollama)
just to look at request schemas. Import directly when you need it:

    from subtitle_forge.server.app import create_app
"""
