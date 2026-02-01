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
  model: large-v3            # Whisper 模型
  device: cuda               # cuda 或 cpu
  compute_type: float16      # float16, int8_float16, int8

ollama:
  model: qwen2.5:14b         # 翻译模型
  host: http://localhost:11434
  temperature: 0.3
  max_batch_size: 10

output:
  encoding: utf-8
  keep_original: true        # 保留原语言字幕
  bilingual: false           # 默认双语模式

max_workers: 2               # 并发数
log_level: INFO
```

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
| `--bilingual` | - | 生成双语字幕 |
| `--whisperx / --no-whisperx` | - | 使用/不使用 WhisperX |
| `--post-process / --no-post-process` | - | 启用/禁用时间戳后处理 |
| `--timestamp-mode` | - | 后处理模式：off, minimal, full |
| `--vad-mode` | - | VAD 预设：default, aggressive, relaxed, precise |
| `--speech-pad` | - | 语音填充时间（毫秒） |
| `--min-silence` | - | 最小静音时长（毫秒） |
| `--prompt-template` | `-p` | 使用指定提示词模板 |
| `--hf-mirror` | - | HuggingFace 镜像 URL |
| `--save-debug-log` | - | 保存调试日志 |

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

### 配置选项

```yaml
timestamp:
  enabled: true
  mode: "minimal"              # off, minimal, full
  min_duration: 0.5            # 最小时长（秒）
  max_duration: 8.0            # 最大时长（秒）
  min_gap: 0.05                # 最小间隙（秒）
  chars_per_second: 15.0       # 西文阅读速度
  cjk_chars_per_second: 10.0   # 中日韩阅读速度
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
