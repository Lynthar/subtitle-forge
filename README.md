# subtitle-forge

Generate and translate video subtitles locally using AI - no cloud services required.

## Features

- **100% Local Processing** - All processing runs on your computer, no data leaves your machine
- **Speech Recognition** - Uses faster-whisper with support for 99+ languages
- **AI Translation** - Context-aware translation using local LLM (Ollama)
- **GPU Acceleration** - CUDA support for fast processing (optional)
- **Batch Processing** - Process multiple videos with configurable concurrency

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
pip install -e .
```

### 4. Run Setup Wizard

```bash
subtitle-forge quickstart
```

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
