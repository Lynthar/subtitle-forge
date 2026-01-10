# subtitle-forge

Local video subtitle generation and translation CLI tool.

## Features

- **Local Processing**: All processing runs locally, no cloud services required
- **Speech Recognition**: Uses faster-whisper with CUDA acceleration for fast, accurate transcription
- **Multi-language Translation**: Translates to any language using Ollama local LLM
- **Batch Processing**: Process multiple videos with configurable concurrency
- **User-friendly CLI**: Rich progress display and helpful prompts

## Requirements

- Python 3.9+
- NVIDIA GPU with CUDA support (recommended) or CPU
- ffmpeg
- Ollama (for translation)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/subtitle-forge.git
cd subtitle-forge

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### System Dependencies

```bash
# Install ffmpeg (macOS)
brew install ffmpeg

# Install ffmpeg (Ubuntu/Debian)
sudo apt install ffmpeg

# Install and start Ollama
# See: https://ollama.ai/download
ollama serve

# Pull a translation model
ollama pull qwen2.5:14b
```

## Usage

### Process a Single Video

```bash
# Basic usage - transcribe and translate
subtitle-forge process video.mp4 --target-lang zh

# Multiple target languages
subtitle-forge process video.mp4 -t zh -t ja -t ko

# Keep original and generate bilingual subtitles
subtitle-forge process video.mp4 -t zh --bilingual

# Specify output directory
subtitle-forge process video.mp4 -t zh -o ./subtitles/
```

### Transcribe Only (No Translation)

```bash
# Basic transcription
subtitle-forge transcribe video.mp4

# Specify source language
subtitle-forge transcribe video.mp4 --language en

# Auto-select model based on GPU VRAM
subtitle-forge transcribe video.mp4 --auto-model
```

### Translate Existing Subtitles

```bash
# Translate SRT file
subtitle-forge translate video.en.srt --target-lang zh

# Generate bilingual subtitles
subtitle-forge translate video.en.srt -t zh --bilingual
```

### Batch Processing

```bash
# Process all videos in a directory
subtitle-forge batch ./videos/ --target-lang zh

# Recursive search with multiple workers
subtitle-forge batch ./videos/ -t zh --recursive --workers 2

# Process from file list
subtitle-forge batch --file-list videos.txt -t zh
```

### Configuration

```bash
# Show current configuration
subtitle-forge config show

# Set configuration values
subtitle-forge config set whisper.model large-v3
subtitle-forge config set ollama.model qwen2.5:32b
subtitle-forge config set max_workers 4

# Check system status
subtitle-forge config check

# Reset to defaults
subtitle-forge config reset
```

## Configuration File

Configuration is stored at `~/.config/subtitle-forge/config.yaml`:

```yaml
whisper:
  model: large-v3
  device: cuda
  compute_type: float16
  beam_size: 5
  vad_filter: true

ollama:
  model: qwen2.5:14b
  host: http://localhost:11434
  temperature: 0.3
  max_batch_size: 10

output:
  encoding: utf-8
  keep_original: true
  bilingual: false

max_workers: 2
log_level: INFO
```

## Supported Languages

The tool supports all languages recognized by Whisper (99+ languages) and can translate to any language supported by the Ollama model.

Common language codes:
- `en` - English
- `zh` - Chinese (Simplified)
- `zh-TW` - Chinese (Traditional)
- `ja` - Japanese
- `ko` - Korean
- `es` - Spanish
- `fr` - French
- `de` - German

## Whisper Model Selection

| Model | VRAM Required | Speed | Accuracy |
|-------|---------------|-------|----------|
| tiny | ~1GB | Fastest | Lower |
| base | ~1.5GB | Fast | Good |
| small | ~2.5GB | Medium | Better |
| medium | ~5GB | Slower | High |
| large-v3 | ~6GB | Slowest | Highest |

Use `--auto-model` to automatically select the best model for your GPU.

## License

MIT License
