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

**Linux/macOS (temporary):**
```bash
export HF_ENDPOINT=https://hf-mirror.com
subtitle-forge process video.mp4 -t zh
```

**Linux/macOS (permanent):**
```bash
# For zsh (macOS default)
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.zshrc
source ~/.zshrc

# For bash (Linux default)
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

**Windows PowerShell (temporary):**
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
subtitle-forge process video.mp4 -t zh
```

**Windows (permanent):**
```powershell
# Add to system environment variables
[Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "User")
# Restart PowerShell after running this command
```

Alternative mirrors:
- `https://hf-mirror.com` (recommended)
- `https://huggingface.sukaka.top`

### GPU not detected / "CUDA not available"

GPU acceleration requires PyTorch with CUDA support. The correct version depends on your GPU:

**Check your current setup:**
```bash
subtitle-forge config check --verbose
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"
```

**Install PyTorch based on your GPU:**

| GPU Series | Architecture | Required PyTorch |
|------------|--------------|------------------|
| RTX 50xx (5090, 5080, 5070...) | Blackwell (sm_120) | **cu128** or newer |
| RTX 40xx (4090, 4080, 4070...) | Ada Lovelace (sm_89) | cu118, cu121, cu124 |
| RTX 30xx (3090, 3080, 3070...) | Ampere (sm_86) | cu118, cu121, cu124 |
| RTX 20xx / GTX 16xx | Turing (sm_75) | cu118, cu121, cu124 |
| Older GPUs | - | cu118 |

**Installation commands:**

```bash
# For RTX 50 series (5090, 5080, 5070, etc.) - MUST use cu128
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# For RTX 40/30/20 series - cu124 recommended
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CPU only (no NVIDIA GPU)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### "sm_120 is not compatible" error (RTX 50 series)

If you see this error:
```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation
```

This means you have an RTX 50 series GPU but installed the wrong PyTorch version. RTX 50 series (Blackwell architecture) requires **CUDA 12.8** or newer:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### "Translation is very slow"

- **With GPU**: ~10-20 subtitles per second
- **Without GPU**: 5-10x slower

Check your GPU status:
```bash
subtitle-forge config check
```

### "Transcription is very slow" (taking 10+ minutes for short videos)

This usually means GPU is not being used. Check:

1. **Verify CUDA is working:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
   Should print `True`. If `False`, reinstall PyTorch with correct CUDA version (see above).

2. **Check for architecture mismatch:**
   ```bash
   subtitle-forge config check --verbose
   ```
   Look for warnings about CUDA capability.

3. **Expected transcription times (with GPU):**
   - 1-hour video: ~2-5 minutes
   - 3-hour video: ~5-15 minutes

   If it's taking much longer, GPU acceleration is likely not working.

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

### Required
- **Python**: 3.9 or higher (3.11+ recommended)
- **ffmpeg**: For audio extraction from videos
- **Ollama**: For AI translation ([download](https://ollama.ai/download))

### Optional (for GPU acceleration)
- **NVIDIA GPU**: Any CUDA-capable GPU (recommended for faster processing)
- **PyTorch with CUDA**: Must match your GPU architecture

### PyTorch Installation by GPU

| Your GPU | Install Command |
|----------|-----------------|
| **RTX 50xx** (5090, 5080, 5070) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` |
| **RTX 40xx** (4090, 4080, 4070) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| **RTX 30xx** (3090, 3080, 3070) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| **RTX 20xx / GTX 16xx** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| **No GPU / CPU only** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` |

> **Important for RTX 50 series users**: You MUST use `cu128` or newer. Using `cu124` or older will result in "sm_120 not compatible" errors and fall back to slow CPU mode.

### Disk Space
- Whisper model: ~3GB (large-v3)
- Translation model: ~8GB (qwen2.5:14b)
- Temporary audio files: varies by video length

---

## License

MIT License
