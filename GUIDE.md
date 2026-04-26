# subtitle-forge 使用指南

本地化视频字幕生成和翻译工具的完整使用指南。

---

## 目录

1. [安装与配置](#一安装与配置)
   - [系统要求](#系统要求)
   - [各平台安装](#各平台安装)
   - [GPU 加速设置](#gpu-加速设置)
   - [HuggingFace 镜像](#huggingface-镜像中国用户)
   - [硬件与模型推荐](#硬件与模型推荐)
   - [更新方法](#更新方法)
2. [基本使用](#二基本使用)
   - [快速开始](#快速开始)
   - [路径格式说明](#路径格式说明)
   - [常见用法示例](#常见用法示例)
   - [配置文件](#配置文件)
3. [命令索引](#三命令索引)
   - [process - 完整处理](#process---完整处理)
   - [batch - 批量处理](#batch---批量处理)
   - [transcribe - 仅转录](#transcribe---仅转录)
   - [translate - 仅翻译](#translate---仅翻译)
   - [config - 配置管理](#config---配置管理)
   - [quickstart - 初始化向导](#quickstart---初始化向导)
4. [故障排查](#四故障排查)
5. [附录](#五附录)
   - [支持的语言](#支持的语言)
   - [时间戳后处理](#时间戳后处理)
   - [翻译提示词配置](#翻译提示词配置)

---

# 一、安装与配置

## 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Windows 10 / macOS 10.15 / Ubuntu 20.04 | Windows 11 / macOS 14 / Ubuntu 22.04 |
| **Python** | 3.9+ | 3.11+ |
| **内存** | 8 GB | 16 GB+ |
| **显卡** | 无（CPU 可运行，较慢） | NVIDIA GPU 6GB+ VRAM |
| **存储** | 10 GB（模型缓存） | SSD 推荐 |

**必需依赖**：
- Python 3.9+
- ffmpeg
- Ollama

---

## 各平台安装

### Windows

```powershell
# 1. 安装 Python（从 https://python.org/downloads 下载）
#    重要：勾选 "Add Python to PATH"

# 2. 安装 ffmpeg（任选一种）
choco install ffmpeg          # Chocolatey
winget install ffmpeg         # winget
# 或手动下载：https://ffmpeg.org/download.html

# 3. 安装 Ollama（从 https://ollama.ai/download 下载）

# 4. 启动 Ollama 服务（保持终端运行）
ollama serve

# 5. 安装 subtitle-forge（新终端）
git clone https://github.com/your-repo/subtitle-forge.git
cd subtitle-forge
pip install -e .

# 6. 运行设置向导
subtitle-forge quickstart
```

### macOS

```bash
# 使用 Homebrew
brew install python@3.11 ffmpeg ollama

# 启动 Ollama
brew services start ollama  # 后台服务
# 或 ollama serve           # 前台运行

# 安装 subtitle-forge
git clone https://github.com/your-repo/subtitle-forge.git
cd subtitle-forge
pip install -e .
subtitle-forge quickstart
```

### Linux (Ubuntu/Debian)

```bash
# 安装依赖
sudo apt update
sudo apt install python3 python3-pip python3-venv ffmpeg

# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve  # 保持运行

# 安装 subtitle-forge（推荐使用虚拟环境）
git clone https://github.com/your-repo/subtitle-forge.git
cd subtitle-forge
python3 -m venv venv
source venv/bin/activate
pip install -e .
subtitle-forge quickstart
```

**其他发行版**：
- **Fedora**: `sudo dnf install python3 python3-pip ffmpeg`
- **Arch**: `sudo pacman -S python python-pip ffmpeg`

### 可选安装项（extras）

`pip install -e .` 是基础安装，根据使用场景可以追加：

| 命令 | 提供的功能 |
|------|----------|
| `pip install -e '.[whisperx]'` | **强烈推荐**。WhisperX 用 wav2vec2 强制对齐，词级时间戳精度比基础 faster-whisper 高很多，直接影响字幕同步感 |
| `pip install -e '.[serve]'` | 启用 HTTP 服务模式（`subtitle-forge serve`），加 fastapi + uvicorn |
| `pip install -e '.[dev]'` | 开发依赖：pytest / ruff / mypy |

可以叠加：`pip install -e '.[whisperx,serve]'`。

---

## GPU 加速设置

GPU 加速可大幅提升处理速度（5-10 倍）。

### 检查 GPU 状态

```bash
# 检查 PyTorch 是否识别 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 查看 GPU 信息
nvidia-smi
```

### 按显卡型号安装 PyTorch

| GPU 系列 | 架构 | 安装命令 |
|----------|------|----------|
| RTX 50xx (5090, 5080...) | Blackwell | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| RTX 40xx (4090, 4080...) | Ada | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| RTX 30xx (3090, 3080...) | Ampere | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| RTX 20xx / GTX 16xx | Turing | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| 无 GPU | - | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

> **注意**：RTX 50 系列必须使用 CUDA 12.8 (`cu128`)，否则会报 "sm_120 is not compatible" 错误。

### 完整安装流程

```bash
# 1. 卸载现有 PyTorch
pip uninstall torch torchvision torchaudio -y

# 2. 安装对应版本（以 RTX 40 系列为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 验证
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

---

## HuggingFace 镜像（中国用户）

Whisper 模型从 HuggingFace 下载，国内用户可配置镜像加速。

### 临时设置

```bash
# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com
subtitle-forge process video.mp4 -t zh

# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"

# Windows CMD
set HF_ENDPOINT=https://hf-mirror.com
```

### 永久设置

```bash
# macOS (zsh)
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.zshrc && source ~/.zshrc

# Linux (bash)
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc

# Windows (需要管理员权限)
[Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "User")
```

### 命令行指定

```bash
subtitle-forge process video.mp4 -t zh --hf-mirror https://hf-mirror.com
```

**可用镜像**：`https://hf-mirror.com`、`https://huggingface.sukaka.top`

---

## 硬件与模型推荐

### Whisper 模型（语音识别）

| 显存 | 推荐模型 | 速度 | 准确度 |
|------|---------|------|--------|
| < 4 GB | `small` | 快 | 较好 |
| 4-6 GB | `medium` | 中等 | 高 |
| 6 GB+ | `large-v3` | 较慢 | 最高 |

```bash
# 设置默认模型
subtitle-forge config set whisper.model large-v3

# 自动选择
subtitle-forge transcribe video.mp4 --auto-model
```

### Ollama 模型（翻译）

| 显存 | 推荐模型 | 翻译速度 | 翻译质量 |
|------|---------|---------|---------|
| < 6 GB | `qwen2.5:7b` | 快 | 一般 |
| 6-12 GB | `qwen2.5:14b` | 中等 | 较好 |
| 16 GB+ | `qwen2.5:32b` | 较慢 | 最佳 |

```bash
# 设置默认模型
subtitle-forge config set ollama.model qwen2.5:14b
```

---

## 更新方法

```bash
# 标准更新
cd subtitle-forge
git pull origin main
pip install -e .

# 一键更新（含缓存清理）
cd subtitle-forge && git pull && find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null; pip install -e .

# 查看版本
subtitle-forge version
```

---

# 二、基本使用

## 快速开始

```bash
# 1. 确保 Ollama 正在运行
ollama serve

# 2. 运行设置向导（首次使用）
subtitle-forge quickstart

# 3. 处理视频
subtitle-forge process video.mp4 -t zh
```

---

## 路径格式说明

不同操作系统的路径格式有所不同：

### Windows

```powershell
# 使用反斜杠或正斜杠
subtitle-forge process "D:\Videos\movie.mp4" -t zh
subtitle-forge process "D:/Videos/movie.mp4" -t zh

# 路径包含空格时必须加引号
subtitle-forge process "C:\My Videos\movie.mp4" -t zh

# 当前目录
subtitle-forge process .\video.mp4 -t zh

# 批量处理
subtitle-forge batch "D:\Videos\" -t zh
```

### macOS / Linux

```bash
# 使用正斜杠
subtitle-forge process /Users/name/Videos/movie.mp4 -t zh

# 路径包含空格时加引号
subtitle-forge process "/Users/name/My Videos/movie.mp4" -t zh

# 使用 ~ 表示主目录
subtitle-forge process ~/Videos/movie.mp4 -t zh

# 当前目录
subtitle-forge process ./video.mp4 -t zh
```

### 通用建议

- **路径包含空格**：始终使用引号包裹
- **特殊字符**：避免使用 `& | < > ! $ ; ` 等特殊字符
- **中文路径**：支持，但建议使用英文路径避免潜在问题

---

## 常见用法示例

### 生成字幕

```bash
# 基本用法：视频 → 中文字幕
subtitle-forge process video.mp4 -t zh

# 多语言字幕
subtitle-forge process video.mp4 -t zh -t ja -t ko

# 双语字幕（原文 + 译文）
subtitle-forge process video.mp4 -t zh --bilingual

# 指定输出目录
subtitle-forge process video.mp4 -t zh -o ./output/

# 指定源语言（跳过自动检测）
subtitle-forge process video.mp4 -s en -t zh
```

### 批量处理

```bash
# 处理文件夹中所有视频
subtitle-forge batch ./videos/ -t zh

# 递归处理子文件夹
subtitle-forge batch ./videos/ -t zh --recursive

# 并发处理
subtitle-forge batch ./videos/ -t zh --workers 4
```

### 仅转录（不翻译）

```bash
# 生成原语言字幕
subtitle-forge transcribe video.mp4

# 指定语言
subtitle-forge transcribe video.mp4 --language en
```

### 仅翻译（已有字幕）

```bash
# 翻译现有字幕文件
subtitle-forge translate video.en.srt -t zh

# 生成双语字幕
subtitle-forge translate video.en.srt -t zh --bilingual
```

---

## 配置文件

### 位置

| 系统 | 路径 |
|------|------|
| Linux/macOS | `~/.config/subtitle-forge/config.yaml` |
| Windows | `%APPDATA%\subtitle-forge\config.yaml` |

### 常用命令

```bash
# 查看当前配置
subtitle-forge config show

# 修改配置
subtitle-forge config set whisper.model large-v3
subtitle-forge config set ollama.model qwen2.5:32b
subtitle-forge config set max_workers 4

# 重置配置
subtitle-forge config reset

# 系统检查
subtitle-forge config check --verbose
```

### 配置示例

```yaml
whisper:
  model: large-v3                # Whisper 模型
  device: cuda                   # cuda 或 cpu
  compute_type: float16          # float16, int8_float16, int8

  # VAD（语音活动检测）调参 - 影响转录段落的边界
  speech_pad_ms: 250             # VAD 检测到语音前后的填充（毫秒）
  min_silence_duration_ms: 700   # 切分相邻语句的最小静音时长（毫秒）

  # WhisperX（强烈推荐，需要 [whisperx] 安装 extra）
  use_whisperx: true             # 用 wav2vec2 做强制对齐
  whisperx_align: true

ollama:
  model: qwen2.5:14b             # 翻译模型
  host: http://localhost:11434
  temperature: 0.0               # 翻译用 0；之前是 0.3 容易丢段
  max_batch_size: 10             # 上限值；实际会按模型自适应：14B→6, ≤8B→4
  request_timeout: 180.0         # 单次请求超时（秒）

output:
  encoding: utf-8
  keep_original: true            # 保留原语言字幕
  bilingual: false               # 默认双语模式

max_workers: 2                   # 并发数
log_level: INFO
```

完整字段说明详见仓库内的 `config/default.yaml`。

---

# 三、命令索引

## process - 完整处理

**功能**：视频 → 音频提取 → 语音识别 → 翻译 → 字幕文件

```bash
subtitle-forge process <video> -t <target_lang> [options]
```

| 参数 | 简写 | 说明 |
|------|------|------|
| `<video>` | - | 视频文件路径（必需） |
| `--target-lang` | `-t` | 目标语言，可多次指定（必需） |
| `--source-lang` | `-s` | 源语言（默认自动检测） |
| `--output-dir` | `-o` | 输出目录（默认视频所在目录） |
| `--whisper-model` | - | 指定 Whisper 模型 |
| `--ollama-model` | - | 指定翻译模型 |
| `--keep-original / --no-keep-original` | - | 是否同时保存原语言字幕（默认保留）|
| `--bilingual` | - | 生成双语字幕（原文上译文下） |
| `--whisperx / --no-whisperx` | - | 使用/不使用 WhisperX |
| `--post-process / --no-post-process` | - | 启用/禁用时间戳后处理 |
| `--timestamp-mode` | - | 后处理模式：off, minimal, full |
| `--split-sentences / --no-split-sentences` | - | 按句子拆分多句字幕 |
| `--vad-mode` | - | VAD 预设：default, aggressive, relaxed, precise |
| `--speech-pad` | - | 语音填充时间（毫秒），覆盖配置 |
| `--min-silence` | - | 最小静音时长（毫秒），覆盖配置 |
| `--prompt-template` | `-p` | 使用指定提示词模板 |
| `--hf-mirror` | - | HuggingFace 镜像 URL |
| `--save-failed-log` | - | 仅保存翻译失败的 JSON 详情 |
| `--save-debug-log` | - | 保存完整调试日志 + 失败 JSON（推荐排查时用） |

**示例**：

```bash
# 基本用法
subtitle-forge process movie.mp4 -t zh

# 高质量处理
subtitle-forge process movie.mp4 -t zh --whisper-model large-v3 --ollama-model qwen2.5:32b

# 使用动漫提示词模板
subtitle-forge process anime.mp4 -t zh -p anime

# 保存调试日志
subtitle-forge process movie.mp4 -t zh --save-debug-log
```

---

## batch - 批量处理

**功能**：批量处理多个视频文件

```bash
subtitle-forge batch <directory> -t <target_lang> [options]
```

| 参数 | 简写 | 说明 |
|------|------|------|
| `<directory>` | - | 视频目录路径（必需） |
| `--target-lang` | `-t` | 目标语言（必需） |
| `--recursive` | `-r` | 递归处理子目录 |
| `--workers` | `-w` | 并发数（默认 2） |
| `--file-list` | - | 从文件列表读取 |
| `--output-dir` | `-o` | 输出目录 |

**示例**：

```bash
# 处理目录
subtitle-forge batch ./videos/ -t zh

# 递归处理 + 4 线程
subtitle-forge batch ./videos/ -t zh -r --workers 4

# 从列表文件处理
subtitle-forge batch --file-list files.txt -t zh
```

---

## transcribe - 仅转录

**功能**：语音识别，生成原语言字幕（不翻译）

```bash
subtitle-forge transcribe <video> [options]
```

| 参数 | 简写 | 说明 |
|------|------|------|
| `<video>` | - | 视频文件路径（必需） |
| `--output` | `-o` | 输出文件路径 |
| `--language` | `-l` | 源语言（默认自动检测） |
| `--model` | `-m` | Whisper 模型 |
| `--auto-model` | - | 根据 GPU 自动选择模型 |
| `--whisperx / --no-whisperx` | - | 使用/不使用 WhisperX |
| `--timestamp-mode` | - | 后处理模式 |
| `--split-sentences / --no-split-sentences` | - | 按句子拆分多句字幕 |
| `--save-debug-log` | - | 保存调试日志 |

**示例**：

```bash
# 基本转录
subtitle-forge transcribe video.mp4

# 指定语言和模型
subtitle-forge transcribe video.mp4 -l en -m large-v3

# 自动选择最佳模型
subtitle-forge transcribe video.mp4 --auto-model
```

---

## translate - 仅翻译

**功能**：翻译现有字幕文件

```bash
subtitle-forge translate <subtitle> -t <target_lang> [options]
```

| 参数 | 简写 | 说明 |
|------|------|------|
| `<subtitle>` | - | 字幕文件路径（必需） |
| `--target-lang` | `-t` | 目标语言（必需） |
| `--source-lang` | `-s` | 源语言 |
| `--output` | `-o` | 输出文件路径 |
| `--bilingual` | - | 生成双语字幕 |

**示例**：

```bash
# 翻译字幕
subtitle-forge translate video.en.srt -t zh

# 双语字幕
subtitle-forge translate video.en.srt -t zh --bilingual
```

---

## config - 配置管理

**功能**：查看和修改配置

| 子命令 | 说明 |
|--------|------|
| `show` | 显示当前配置 |
| `set <key> <value>` | 设置配置项 |
| `reset` | 重置为默认配置 |
| `check` | 检查系统环境 |
| `pull-model` | 下载翻译模型 |
| `show-prompt` | 显示当前翻译提示词 |
| `list-prompts` | 列出可用提示词模板 |
| `use-prompt <id>` | 使用指定模板 |
| `export-prompt` | 导出提示词 |
| `set-prompt` | 设置自定义提示词 |
| `reset-prompt` | 重置为默认提示词 |

**示例**：

```bash
# 查看配置
subtitle-forge config show

# 修改模型
subtitle-forge config set whisper.model large-v3
subtitle-forge config set ollama.model qwen2.5:32b

# 系统检查
subtitle-forge config check --verbose

# 下载模型
subtitle-forge config pull-model

# 使用动漫模板
subtitle-forge config use-prompt anime
```

---

## quickstart - 初始化向导

**功能**：首次使用的设置向导

```bash
subtitle-forge quickstart
```

向导会自动：
- 检测系统环境
- 检查依赖是否安装
- 下载必要的模型
- 创建默认配置

---

# 四、故障排查

## 无法连接 Ollama

**症状**：`Cannot connect to Ollama` / `Connection refused`

**解决方案**：

```bash
# 1. 确认 Ollama 正在运行
ollama serve

# 2. 检查服务状态
curl http://localhost:11434

# 3. 检查端口占用
lsof -i :11434          # Linux/macOS
netstat -ano | findstr 11434  # Windows
```

---

## GPU 未识别

**症状**：`CUDA not available` / `Using CPU`

**解决方案**：

```bash
# 1. 系统诊断
subtitle-forge config check --verbose

# 2. 检查 PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. 检查驱动
nvidia-smi

# 4. 重新安装正确版本 PyTorch（见 GPU 加速设置）
```

---

## RTX 50 系列 sm_120 错误

**症状**：`sm_120 is not compatible with the current PyTorch installation`

**解决方案**：

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 下载速度慢

**症状**：Whisper 模型下载缓慢

**解决方案**：配置 HuggingFace 镜像（见 [HuggingFace 镜像](#huggingface-镜像中国用户)）

---

## 模型下载中断

**症状**：下载过程中断

**解决方案**：直接重新运行命令，支持断点续传

```bash
subtitle-forge process video.mp4 -t zh  # Whisper 模型会自动续传
subtitle-forge config pull-model         # Ollama 模型
```

---

## 处理速度慢

### 转录速度慢

**参考速度（GPU）**：1 小时视频约 2-5 分钟

**解决方案**：

```bash
# 确认 GPU 使用
nvidia-smi -l 1  # 处理时监控

# 使用较小模型
subtitle-forge config set whisper.model medium
```

### 翻译速度慢

**参考速度**：GPU 约 10-20 条/秒，CPU 约 1-3 条/秒

**解决方案**：

```bash
# 使用较小模型
subtitle-forge config set ollama.model qwen2.5:7b
```

---

## 字幕时间轴感觉不对

字幕和声音的同步问题往往是 `lead_in_ms` / `linger_ms` 没调好。
**先用 `--save-debug-log` 跑一次**，对照实际效果调下面的参数。

### 字幕比声音晚出现（"还没看到字幕，话已经讲了一半"）

```bash
subtitle-forge config set timestamp.lead_in_ms 150   # 默认 80，调大
```

### 字幕在声音停的瞬间消失（"字还没看完就没了"）

```bash
subtitle-forge config set timestamp.linger_ms 500    # 默认 300，调大
```

### 字幕一直显示，对话结束后还在屏幕上

通常是某些段在转录时 `segment.end` 被错误延伸到下一段开始或文件末尾。
首先确认你的 `mode` 是 `minimal`（默认），它会自动修这种情况：

```bash
subtitle-forge config set timestamp.mode minimal
```

如果仍然超长，可能是没装 `[whisperx]`——基础 faster-whisper 的对齐精度会更弱。
推荐安装：

```bash
pip install -e '.[whisperx]'
```

### 短字幕快速闪过

通常发生在密集对话场景。如果字幕之间间隔过近，`lead_in_ms + linger_ms + min_gap`
的总和会超过实际词间距，linger 会被自动截短以避免重叠。这是设计取舍——宁可短一些，
不让两段字幕同屏。可以略微减小 lead_in/linger 试试：

```bash
subtitle-forge config set timestamp.lead_in_ms 50
subtitle-forge config set timestamp.linger_ms 200
```

---

## 翻译有些段没翻译

字幕里出现少数几句仍然是原文（未翻译）。

### 第一步：看失败原因

跑一次带 `--save-debug-log`，打开 `<video>_debug/translation_failures.json`：

```bash
subtitle-forge process video.mp4 -t zh --save-debug-log
```

里面的 `failures[*].reason` 字段会分类：

| reason | 含义 | 处理 |
|--------|------|------|
| `模型拒绝翻译` | LLM 内置安全限制（如成人内容） | 换更松的模型，或用 `-p adult` 模板 |
| `内容被过滤` | 同上 | 同上 |
| `LLM返回空响应` | 模型输出空 | 换模型或减小 batch_size |
| `LLM响应过短` | 输出太短被认为失败 | 通常无害，可以忽略 |
| `响应中未找到段落索引` | 解析失败 | 一般会被自动重试修复 |
| `可能是语气词被跳过` | 模型跳过短叹词 | 可忽略 |

### 第二步：手动减小 batch_size（极少需要）

正常情况 batch_size 已按模型大小自动调整（14B → 6, ≤8B → 4）。
如果某个特殊模型还是丢段：

```bash
subtitle-forge config set ollama.max_batch_size 3
```

### 第三步：换更大模型

```bash
subtitle-forge config set ollama.model qwen2.5:32b   # 需要 16GB+ VRAM
```

---

## 保存调试日志

遇到问题时，保存完整日志便于排查：

```bash
subtitle-forge process video.mp4 -t zh --save-debug-log
```

生成文件：

```
output/
├── video.zh.srt
└── video_debug/
    ├── run.log                    # 完整运行日志
    └── translation_failures.json  # 翻译失败详情
```

---

## 完整系统诊断

```bash
subtitle-forge config check --verbose
```

显示：Python 版本、PyTorch/CUDA 版本、GPU 信息、Ollama 状态、已安装模型

---

# 五、附录

## 支持的语言

subtitle-forge 支持 Whisper 识别的 99+ 种语言。

### 常用语言代码

| 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|
| `en` | English | `zh` | 简体中文 |
| `ja` | 日本語 | `ko` | 한국어 |
| `es` | Español | `fr` | Français |
| `de` | Deutsch | `ru` | Русский |
| `pt` | Português | `it` | Italiano |
| `ar` | العربية | `vi` | Tiếng Việt |
| `th` | ไทย | `zh-TW` | 繁體中文 |

---

## 时间戳后处理

### 处理模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `off` | 完全禁用 | WhisperX 对齐效果好 |
| `minimal` | 仅修复重叠和最小时长（**默认**） | 大多数情况 |
| `full` | 完整处理：分割、延长等 | 需要分割多句话 |

### 命令行用法

```bash
# 默认 (minimal)
subtitle-forge process video.mp4 -t zh

# 指定模式
subtitle-forge process video.mp4 -t zh --timestamp-mode off
subtitle-forge process video.mp4 -t zh --timestamp-mode full

# 完全禁用后处理
subtitle-forge process video.mp4 -t zh --no-post-process
```

### 句子级分割

当视频中多句对话被合并到一个字幕段时，可以使用 `--split-sentences` 选项按句子拆分：

```bash
# 启用句子分割（推荐用于电影、电视剧、访谈等对话密集的视频）
subtitle-forge process video.mp4 -t zh --split-sentences

# 也可在 transcribe 命令中使用
subtitle-forge transcribe video.mp4 --split-sentences
```

**工作原理**：
- 利用 WhisperX 提供的词级时间戳
- 在句末标点（。！？等）处拆分
- 根据词时间戳精确计算每个句子的开始和结束时间
- 独立于 `--timestamp-mode` 选项，可与任何模式组合使用

**适用场景**：
- 对话密集的电影、电视剧
- 访谈节目
- 需要逐句显示字幕的场景

### 配置选项

```yaml
timestamp:
  enabled: true
  mode: "minimal"              # off, minimal, full
  min_duration: 1.0            # 最小显示时长（秒）；< 1s 太短读不过来
  max_duration: 8.0            # 最大显示时长（秒）
  min_gap: 0.05                # 相邻字幕的最小间隙（秒）
  max_gap_warning: 10.0        # 间隙超过该值会在日志警告"可能漏录"（秒）
  chars_per_second: 15.0       # 西文阅读速度
  cjk_chars_per_second: 10.0   # 中日韩阅读速度
  split_threshold: 30          # full 模式下的最小拆分字符数
  split_sentences: true        # 句子级分割（默认启用）

  # 字幕显示时间补偿（影响"字幕和声音对齐感"，毫秒）
  # 这两个参数对所有 mode 都生效（包括 mode=off）
  lead_in_ms: 80               # 字幕比第一个字发音提前多少出现
  linger_ms: 300               # 字幕比最后一个字发音延后多少消失
```

#### lead-in / linger 调优指南

声学模型给的"词开始/结束时间"只是发音边界，**不等于字幕该出现/消失的时间**。
为了让观众看得舒服，行业惯例（BBC / Netflix）是字幕略早出现、声音停后再保留一会儿。

- **声音出来了字幕还没出现** → 增大 `lead_in_ms`（试 120 / 150 / 180）
- **字幕在声音停的瞬间就消失了** → 增大 `linger_ms`（试 400 / 500 / 600）
- **字幕拖得太长侵占下一段** → 减小 `linger_ms`（如 150-200）；
  注意：相邻段重叠时，前段的 linger 会被自动截到下一段开始前，不会真的重叠

调整命令：

```bash
subtitle-forge config set timestamp.lead_in_ms 120
subtitle-forge config set timestamp.linger_ms 500
```

### CJK 语言优化

系统自动检测中文、日文、韩文，使用针对性优化：
- 更慢的阅读速度（10 字/秒 vs 15 字/秒）
- 更低的分割阈值（15 字符 vs 30 字符）

---

## 翻译提示词配置

### 内置模板

| 模板 ID | 名称 | 适用场景 |
|---------|------|----------|
| `movie-general` | 通用电影 | 大多数电影/电视剧 |
| `movie-scifi` | 科幻电影 | 科幻、太空题材 |
| `movie-fantasy` | 奇幻电影 | 魔法、奇幻题材 |
| `documentary` | 纪录片 | 纪录片、科教片 |
| `anime` | 动漫 | 日本动漫 |
| `adult` | 成人内容 | 成人影视 |
| `technical` | 技术教程 | 编程、软件教程 |

### 使用模板

```bash
# 查看可用模板
subtitle-forge config list-prompts

# 设置默认模板
subtitle-forge config use-prompt anime

# 临时使用模板
subtitle-forge process anime.mp4 -t zh -p anime
```

### 自定义模板

```bash
# 1. 导出现有模板
subtitle-forge config export-prompt -o my_prompt.txt

# 2. 编辑文件

# 3. 应用自定义提示词
subtitle-forge config set-prompt -f my_prompt.txt

# 4. 重置为默认
subtitle-forge config reset-prompt
```

### 可用占位符

| 占位符 | 说明 |
|--------|------|
| `{source_lang}` | 源语言名称（必需） |
| `{target_lang}` | 目标语言名称（必需） |
| `{segments}` | 要翻译的字幕（必需） |
| `{context_before}` | 前文上下文 |
| `{context_after}` | 后文上下文 |
