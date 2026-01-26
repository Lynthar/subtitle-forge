# subtitle-forge

Generate and translate video subtitles locally using AI - no cloud services required.

## Quick Start (5 minutes)

### 1. Install subtitle-forge

```bash
pip install -e .
```

### 2. Install System Dependencies

**macOS:**
```bash
brew install ffmpeg
brew install ollama
ollama serve  # Keep this running in a separate terminal
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve  # Keep this running in a separate terminal
```

**Windows:**
```bash
# Install ffmpeg from https://ffmpeg.org/download.html
# Install Ollama from https://ollama.ai/download
ollama serve  # Keep this running in a separate terminal
```

### 3. Run Setup Wizard

```bash
subtitle-forge quickstart
```

This interactive wizard will:
- Check your system requirements
- Download the AI translation model (~8GB, one-time)
- Verify everything is working

### 4. Generate Your First Subtitles

```bash
subtitle-forge process your-video.mp4 --target-lang zh
```

---

## Common Use Cases

### Generate subtitles in one language
```bash
subtitle-forge process video.mp4 -t zh        # Chinese
subtitle-forge process video.mp4 -t ja        # Japanese
subtitle-forge process video.mp4 -t ko        # Korean
subtitle-forge process video.mp4 -t en        # English
```

### Generate subtitles in multiple languages
```bash
subtitle-forge process video.mp4 -t zh -t ja -t ko
```

### Create bilingual subtitles (original + translation)
```bash
subtitle-forge process video.mp4 -t zh --bilingual
```

### Process multiple videos at once
```bash
subtitle-forge batch ./my-videos/ -t zh
subtitle-forge batch ./my-videos/ -t zh --recursive  # Include subfolders
subtitle-forge batch --file-list videos.txt -t zh    # From a file list
```

### Transcribe only (no translation)
```bash
subtitle-forge transcribe video.mp4
subtitle-forge transcribe video.mp4 --language en    # Specify source language
subtitle-forge transcribe video.mp4 --auto-model     # Auto-select best model
```

### Translate existing subtitles
```bash
subtitle-forge translate video.en.srt -t zh
subtitle-forge translate video.srt -s en -t zh --bilingual
```

---

## Troubleshooting

### "Cannot connect to Ollama"

Ollama is the AI engine that powers translation. Make sure it's running:
```bash
ollama serve
```

### "Model not found" or download keeps failing

Download the translation model manually (supports resume if interrupted):
```bash
subtitle-forge config pull-model
```

If download fails, just run the command again - it will resume from where it stopped.

### Whisper model download is slow (China/Asia users)

If the Whisper model (~3GB) downloads very slowly, configure a HuggingFace mirror:

**Temporary (current session only):**
```bash
export HF_ENDPOINT=https://hf-mirror.com
subtitle-forge process video.mp4 -t zh
```

**Permanent configuration:**
```bash
# For zsh (macOS default)
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.zshrc
source ~/.zshrc

# For bash (Linux default)
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

Alternative mirrors:
- `https://hf-mirror.com` (recommended)
- `https://huggingface.sukaka.top`

### "Translation is very slow"

- **With GPU**: ~10-20 subtitles per second
- **Without GPU**: 5-10x slower

Check your GPU status:
```bash
subtitle-forge config check
```

### Full system diagnostic
```bash
subtitle-forge config check --verbose
```

---

## Configuration

### View current settings
```bash
subtitle-forge config show
```

### Change settings
```bash
subtitle-forge config set whisper.model large-v3
subtitle-forge config set ollama.model qwen2.5:32b
subtitle-forge config set max_workers 4
```

### Reset to defaults
```bash
subtitle-forge config reset
```

### Configuration file location
`~/.config/subtitle-forge/config.yaml`

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

---

## Supported Languages

The tool supports all languages recognized by Whisper (99+ languages) and can translate to any language supported by Ollama.

| Code | Language |
|------|----------|
| `en` | English |
| `zh` | Chinese (Simplified) |
| `zh-TW` | Chinese (Traditional) |
| `ja` | Japanese |
| `ko` | Korean |
| `es` | Spanish |
| `fr` | French |
| `de` | German |
| `ru` | Russian |
| `pt` | Portuguese |

---

## Whisper Model Selection

| Model | VRAM Required | Speed | Accuracy |
|-------|---------------|-------|----------|
| tiny | ~1GB | Fastest | Lower |
| base | ~1.5GB | Fast | Good |
| small | ~2.5GB | Medium | Better |
| medium | ~5GB | Slower | High |
| large-v3 | ~6GB | Slowest | Highest |

Use `--auto-model` to automatically select the best model for your GPU:
```bash
subtitle-forge transcribe video.mp4 --auto-model
```

---

## Features

- **100% Local Processing**: All processing runs on your computer, no cloud services
- **Speech Recognition**: Uses faster-whisper with CUDA acceleration
- **AI Translation**: Uses Ollama with local LLM models
- **Batch Processing**: Process multiple videos with configurable concurrency
- **Resume Support**: Model downloads can be resumed if interrupted
- **User-friendly CLI**: Rich progress display and interactive setup

---

## Updating

### If installed from GitHub (git clone)

```bash
cd subtitle-forge
git pull origin main

# Clean old cache files (recommended)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Reinstall (only needed if dependencies changed)
pip install -e .
```

Or use this one-liner:
```bash
cd subtitle-forge && git pull && find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null; pip install -e .
```

### If installed from PyPI (future)

```bash
pip install --upgrade subtitle-forge
```

### Check your version

```bash
subtitle-forge version
```

---

## Requirements

- Python 3.9+
- NVIDIA GPU with CUDA support (recommended) or CPU
- ffmpeg
- Ollama (for translation)

---

## License

MIT License
