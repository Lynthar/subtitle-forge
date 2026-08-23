# subtitle-forge

**English** | [简体中文](README.zh-CN.md)

Generate and translate video subtitles locally using AI — no cloud services required.

## Features

- **Local inference** — transcription and translation run on your machine; no audio or subtitle text is sent to any cloud service ([network boundaries](docs/user-guide.md#五隐私与网络边界))
- **Speech Recognition** — faster-whisper with support for 99+ languages, optional WhisperX forced alignment
- **AI Translation** — context-aware translation using a local LLM (Ollama)
- **GPU Acceleration** — CUDA support for fast processing (optional)
- **Batch Processing** — process multiple videos with configurable concurrency
- **HTTP Server Mode** — optional REST API for media servers and automation

## Quick Start

### 1. Install dependencies

```bash
# macOS
brew install ffmpeg ollama

# Ubuntu/Debian
sudo apt install ffmpeg
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: install from https://ffmpeg.org and https://ollama.ai
```

### 2. Start Ollama

```bash
ollama serve
```

### 3. Install subtitle-forge

```bash
# Base install
pip install -e .

# Strongly recommended: WhisperX provides forced wav2vec2 alignment for
# accurate word-level timestamps. Without it, subtitle timing falls back
# to faster-whisper's lower-precision word timestamps.
pip install -e '.[whisperx]'
```

### 4. Run the setup wizard

```bash
subtitle-forge quickstart
```

The wizard checks ffmpeg / Ollama / GPU availability and downloads the default
Whisper + Ollama models, so the first real run isn't blocked on multi-GB
downloads.

### 5. Generate subtitles

```bash
subtitle-forge process video.mp4 -t zh
```

## Common Use Cases

```bash
# Generate subtitles in one language
subtitle-forge process video.mp4 -t zh        # Chinese
subtitle-forge process video.mp4 -t ja        # Japanese

# Generate subtitles in multiple languages
subtitle-forge process video.mp4 -t zh -t ja -t ko

# Create bilingual subtitles
subtitle-forge process video.mp4 -t zh --bilingual

# Batch process videos
subtitle-forge batch ./videos/ -t zh
subtitle-forge batch ./videos/ -t zh --recursive

# Transcribe only (no translation)
subtitle-forge transcribe video.mp4

# Translate existing subtitles
subtitle-forge translate video.en.srt -t zh

# When something feels off — saves run.log + translation_failures.json
subtitle-forge process video.mp4 -t zh --save-debug-log
```

## Configuration

```bash
subtitle-forge config show                          # view every setting
subtitle-forge config set whisper.model large-v3    # change one
subtitle-forge config check --verbose               # system diagnostics
```

Config lives at `~/.config/subtitle-forge/config.yaml` (Windows:
`%APPDATA%\subtitle-forge\config.yaml`). Full field reference with defaults and
the reasoning behind them: [`config/default.yaml`](config/default.yaml).

## Privacy & Network

Inference is local, but "local-first" is not the same as "never connects":

- **Model downloads need network access the first time** — Whisper weights from
  HuggingFace, translation models from Ollama. Everything afterwards is offline.
- **`ollama.host` can point at another machine.** It defaults to `localhost`; if
  you change it, subtitle text goes to that host.
- **`--save-debug-log` / `--save-failed-log` write dialogue to disk** — the
  failure report contains original subtitle text. Read before sharing.
- **Output is written next to the source video** by default, inheriting your
  OS file permissions.

Details, telemetry opt-out, and a fully-offline deployment checklist:
[隐私与网络边界](docs/user-guide.md#五隐私与网络边界).

## Documentation

The **[Usage Guide](docs/user-guide.md)** (Chinese) is the complete reference —
per-platform installation, GPU/CUDA setup per RTX series, command reference,
HTTP server mode, privacy boundaries, and troubleshooting.

## Requirements

- Python 3.9+
- ffmpeg
- Ollama
- NVIDIA GPU (optional, for acceleration)

## License

Apache License 2.0 — see [LICENSE](LICENSE).
