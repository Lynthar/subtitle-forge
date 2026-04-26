# subtitle-forge

Generate and translate video subtitles locally using AI - no cloud services required.

## Features

- **100% Local Processing** - All processing runs on your computer, no data leaves your machine
- **Speech Recognition** - Uses faster-whisper with support for 99+ languages
- **AI Translation** - Context-aware translation using local LLM (Ollama)
- **GPU Acceleration** - CUDA support for fast processing (optional)
- **Batch Processing** - Process multiple videos with configurable concurrency
- **HTTP Server Mode** - Optional REST API for integration with media servers and automation

## Quick Start

### 1. Install Dependencies

```bash
# macOS
brew install ffmpeg ollama

# Ubuntu/Debian
sudo apt install ffmpeg
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Install from https://ffmpeg.org and https://ollama.ai
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

### 4. Run Setup Wizard

```bash
subtitle-forge quickstart
```

The wizard checks ffmpeg / Ollama / GPU availability and downloads the
default Whisper + Ollama models so the first real run isn't blocked on
multi-GB downloads.

### 5. Generate Subtitles

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
```

## HTTP Server Mode

For integration with media servers (Jellyfin, etc.) or any automation that
needs to submit subtitle jobs from another machine, run subtitle-forge as an
HTTP server.

### Install with the `serve` extra

```bash
pip install -e '.[serve]'        # adds fastapi + uvicorn
```

### Start the server

```bash
# Pick a long random token; clients must send it as Bearer auth.
export SUBTITLE_FORGE_TOKEN="$(openssl rand -hex 32)"

subtitle-forge serve --host 0.0.0.0 --port 8765
```

Visit `http://<host>:8765/docs` for the auto-generated OpenAPI UI.

### API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `POST` | `/jobs` | Bearer | Submit a subtitle generation job |
| `GET`  | `/jobs/{id}` | Bearer | Check job status / get output paths |
| `GET`  | `/health` | none | Liveness + queue stats |

`POST /jobs` body:
```json
{
  "video_path": "/absolute/path/as/this/server/sees/it.mp4",
  "target_languages": ["zh"],
  "source_language": null,
  "bilingual": false,
  "keep_original": true
}
```

The server validates `video_path` exists at submit time — you get an immediate
`400` instead of a delayed worker failure if the path is wrong.

Output SRT files are written next to the source video as
`<video_stem>.<lang>.srt` (or `<video_stem>.<src>-<tgt>.srt` for bilingual).

### Run as a daemon

**macOS (launchd)** — save as `~/Library/LaunchAgents/com.subtitle-forge.serve.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.subtitle-forge.serve</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/local/bin/subtitle-forge</string>
    <string>serve</string>
    <string>--host</string><string>0.0.0.0</string>
    <string>--port</string><string>8765</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>SUBTITLE_FORGE_TOKEN</key><string>YOUR-LONG-HEX-TOKEN</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/subtitle-forge.log</string>
  <key>StandardErrorPath</key><string>/tmp/subtitle-forge.err</string>
</dict>
</plist>
```
Then `launchctl load ~/Library/LaunchAgents/com.subtitle-forge.serve.plist`.

**Linux (systemd)** — save as `/etc/systemd/system/subtitle-forge.service`:

```ini
[Unit]
Description=subtitle-forge HTTP server
After=network.target ollama.service

[Service]
Type=simple
User=YOUR_USER
Environment=SUBTITLE_FORGE_TOKEN=YOUR-LONG-HEX-TOKEN
ExecStart=/usr/local/bin/subtitle-forge serve --host 0.0.0.0 --port 8765
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
Then `systemctl enable --now subtitle-forge`.

### Server-mode caveats

- **Models must be cached before first job.** The server refuses to run
  interactive downloads. Run `subtitle-forge transcribe <any-video>` once on
  the host so Whisper downloads its weights, then start the server.
- **Ollama must be running.** Translation calls Ollama; check with
  `subtitle-forge config check --verbose` before going live.
- **Single GPU worker by default** (`--workers 1`). Concurrent Whisper
  invocations thrash VRAM. Don't raise this unless you've measured.
- **First job is slow** (~10–30s). Whisper model is lazy-loaded on first
  request and stays resident afterwards — subsequent jobs start instantly.
- **Jobs are in-memory only.** Restarting the server drops the job log;
  in-flight jobs are marked `failed` so they don't sit in `processing` forever.
  Resubmit from the client side. (Persistence wasn't added on purpose — it's
  yagni for a single-user setup.)
- **Auth is required by default.** `--no-auth` is only safe when binding to
  `127.0.0.1` on a single-user host; the CLI will warn you if you combine
  `--no-auth` with a non-loopback bind.
- **Path handling is strict.** `video_path` must be absolute and visible to
  *this* server's filesystem. If the client and server see the storage at
  different paths (e.g. `/media/jav` vs `/Volumes/nas-jav`), do the rewrite
  on the client.

## Configuration

```bash
# View settings
subtitle-forge config show

# Change settings
subtitle-forge config set whisper.model large-v3
subtitle-forge config set ollama.model qwen2.5:32b

# System check
subtitle-forge config check --verbose
```

Full field reference with defaults and explanations: see `config/default.yaml`.

### Troubleshooting flag

```bash
# Saves run.log (DEBUG) and translation_failures.json next to the output:
subtitle-forge process video.mp4 -t zh --save-debug-log
```

This is the first thing to try when something feels off — the saved log
is detailed enough to diagnose timing issues, missed translations, model
download failures, etc.

## Supported Languages

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `zh` | Chinese |
| `ja` | Japanese | `ko` | Korean |
| `es` | Spanish | `fr` | French |
| `de` | German | `ru` | Russian |

And 90+ more languages supported by Whisper.

## Documentation

For detailed installation instructions, GPU setup, troubleshooting, and configuration options, see the **[Usage Guide](GUIDE.md)**.

## Requirements

- Python 3.9+
- ffmpeg
- Ollama
- NVIDIA GPU (optional, for acceleration)

## License

MIT License
