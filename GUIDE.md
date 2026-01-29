# subtitle-forge 使用指南

本文档提供 subtitle-forge 的完整安装指南、环境配置、详细使用说明和故障排查。

---

## 目录

1. [安装指南](#安装指南)
   - [Windows](#windows-安装)
   - [macOS](#macos-安装)
   - [Linux](#linux-安装)
2. [环境配置](#环境配置)
   - [GPU 加速设置](#gpu-加速设置)
   - [HuggingFace 镜像配置](#huggingface-镜像配置中国用户)
3. [详细使用说明](#详细使用说明)
   - [单文件处理](#单文件处理)
   - [批量处理](#批量处理)
   - [仅转写](#仅转写不翻译)
   - [仅翻译](#仅翻译已有字幕)
   - [双语字幕](#双语字幕)
4. [配置文件](#配置文件)
   - [翻译提示词配置](#翻译提示词配置)
5. [故障排查](#故障排查)
6. [支持的语言](#支持的语言)
7. [Whisper 模型选择](#whisper-模型选择)
8. [更新方法](#更新方法)

---

## 安装指南

### Windows 安装

#### 1. 安装 Python

1. 下载 Python 3.11+: https://python.org/downloads
2. 运行安装程序
3. **重要**: 勾选 "Add Python to PATH"
4. 点击 "Install Now"

验证安装:
```powershell
python --version
```

#### 2. 安装 ffmpeg

**方式一：使用 Chocolatey（推荐）**

如果已安装 [Chocolatey](https://chocolatey.org/install):
```powershell
choco install ffmpeg
```

**方式二：使用 winget**
```powershell
winget install ffmpeg
```

**方式三：手动安装**

1. 下载: https://ffmpeg.org/download.html (选择 Windows builds)
2. 解压到 `C:\ffmpeg`
3. 添加 `C:\ffmpeg\bin` 到系统 PATH:
   - 右键"此电脑" → 属性 → 高级系统设置
   - 环境变量 → 系统变量 → Path → 编辑
   - 新建 → 输入 `C:\ffmpeg\bin`
   - 确定保存

验证安装:
```powershell
ffmpeg -version
```

#### 3. 安装 Ollama

1. 下载: https://ollama.ai/download
2. 运行安装程序
3. 打开终端，启动 Ollama 服务:
```powershell
ollama serve
```

> **提示**: 保持此终端窗口运行，或将 Ollama 设置为后台服务。

#### 4. 安装 subtitle-forge

```powershell
# 克隆仓库
git clone https://github.com/your-repo/subtitle-forge.git
cd subtitle-forge

# 安装
pip install -e .
```

#### 5. 运行设置向导

```powershell
subtitle-forge quickstart
```

向导会自动检测环境并下载所需模型。

---

### macOS 安装

#### 使用 Homebrew（推荐）

```bash
# 安装 Homebrew（如未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装依赖
brew install python@3.11 ffmpeg

# 安装 Ollama
brew install ollama

# 启动 Ollama 服务
ollama serve
```

> **提示**: 在另一个终端窗口运行 `ollama serve`，或使用 `brew services start ollama` 设置后台服务。

#### 安装 subtitle-forge

```bash
# 克隆仓库
git clone https://github.com/your-repo/subtitle-forge.git
cd subtitle-forge

# 安装
pip install -e .

# 运行设置向导
subtitle-forge quickstart
```

---

### Linux 安装

#### Ubuntu / Debian

```bash
# 更新包管理器
sudo apt update

# 安装 Python 和 ffmpeg
sudo apt install python3 python3-pip python3-venv ffmpeg

# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 启动 Ollama 服务
ollama serve
```

#### Fedora / RHEL

```bash
# 安装依赖
sudo dnf install python3 python3-pip ffmpeg

# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 启动 Ollama
ollama serve
```

#### Arch Linux

```bash
# 安装依赖
sudo pacman -S python python-pip ffmpeg

# 安装 Ollama (AUR)
yay -S ollama

# 启动 Ollama
ollama serve
```

#### 安装 subtitle-forge

```bash
# 克隆仓库
git clone https://github.com/your-repo/subtitle-forge.git
cd subtitle-forge

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 安装
pip install -e .

# 运行设置向导
subtitle-forge quickstart
```

---

## 环境配置

### GPU 加速设置

GPU 加速可显著提升转写和翻译速度。subtitle-forge 支持 NVIDIA GPU（通过 CUDA）。

#### 检查 GPU 支持

```bash
# 检查 PyTorch 是否识别 GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 查看 GPU 信息
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

#### 按 GPU 型号安装 PyTorch

不同 GPU 需要不同版本的 CUDA。根据你的显卡选择正确的安装命令:

| GPU 系列 | 架构 | 安装命令 |
|----------|------|----------|
| **RTX 50xx** (5090, 5080, 5070...) | Blackwell (sm_120) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` |
| **RTX 40xx** (4090, 4080, 4070...) | Ada Lovelace (sm_89) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| **RTX 30xx** (3090, 3080, 3070...) | Ampere (sm_86) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| **RTX 20xx / GTX 16xx** | Turing (sm_75) | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| **无 GPU / 仅 CPU** | - | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` |

> **重要**: RTX 50 系列（Blackwell 架构）**必须**使用 CUDA 12.8 (`cu128`)。使用其他版本会导致 "sm_120 is not compatible" 错误。

#### 完整安装流程

```bash
# 1. 卸载现有 PyTorch
pip uninstall torch torchvision torchaudio -y

# 2. 安装正确版本（以 RTX 40 系列为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 验证安装
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

---

### HuggingFace 镜像配置（中国用户）

Whisper 模型（约 3GB）从 HuggingFace 下载。中国大陆用户可能遇到下载缓慢问题，可配置镜像加速。

#### 临时设置（当前会话）

**Linux / macOS:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
subtitle-forge process video.mp4 -t zh
```

**Windows PowerShell:**
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
subtitle-forge process video.mp4 -t zh
```

**Windows CMD:**
```cmd
set HF_ENDPOINT=https://hf-mirror.com
subtitle-forge process video.mp4 -t zh
```

#### 永久设置

**macOS (zsh):**
```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.zshrc
source ~/.zshrc
```

**Linux (bash):**
```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

**Windows (永久环境变量):**
```powershell
[Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "User")
# 重启 PowerShell 后生效
```

#### 可用镜像

- `https://hf-mirror.com` (推荐)
- `https://huggingface.sukaka.top`

---

## 详细使用说明

### 单文件处理

基本用法 - 生成指定语言的字幕:

```bash
# 生成中文字幕
subtitle-forge process video.mp4 -t zh

# 生成日文字幕
subtitle-forge process video.mp4 -t ja

# 生成多语言字幕
subtitle-forge process video.mp4 -t zh -t ja -t ko
```

指定输出目录:
```bash
subtitle-forge process video.mp4 -t zh -o ./subtitles/
```

指定源语言（跳过自动检测）:
```bash
subtitle-forge process video.mp4 -s en -t zh
```

---

### 批量处理

处理文件夹中的所有视频:

```bash
# 处理当前文件夹
subtitle-forge batch ./ -t zh

# 处理指定文件夹
subtitle-forge batch /path/to/videos/ -t zh

# 递归处理子文件夹
subtitle-forge batch /path/to/videos/ -t zh --recursive
```

使用文件列表:
```bash
# 创建文件列表
echo "video1.mp4" > files.txt
echo "video2.mkv" >> files.txt

# 处理列表中的文件
subtitle-forge batch --file-list files.txt -t zh
```

设置并发数:
```bash
# 同时处理 4 个视频
subtitle-forge batch ./videos/ -t zh --workers 4
```

---

### 仅转写（不翻译）

只生成原语言字幕，不进行翻译:

```bash
# 自动检测语言
subtitle-forge transcribe video.mp4

# 指定源语言
subtitle-forge transcribe video.mp4 --language en

# 自动选择最佳 Whisper 模型
subtitle-forge transcribe video.mp4 --auto-model

# 指定 Whisper 模型
subtitle-forge transcribe video.mp4 --model large-v3
```

---

### 仅翻译（已有字幕）

翻译现有的 SRT 字幕文件:

```bash
# 翻译字幕文件
subtitle-forge translate video.en.srt -t zh

# 指定源语言
subtitle-forge translate video.srt -s en -t zh

# 生成双语字幕
subtitle-forge translate video.srt -s en -t zh --bilingual
```

---

### 双语字幕

生成包含原文和译文的双语字幕:

```bash
# 处理视频，生成双语字幕
subtitle-forge process video.mp4 -t zh --bilingual

# 翻译现有字幕为双语
subtitle-forge translate video.en.srt -t zh --bilingual
```

双语字幕格式示例:
```
1
00:00:01,000 --> 00:00:03,000
Hello, how are you?
你好，你好吗？

2
00:00:04,000 --> 00:00:06,000
I'm fine, thank you.
我很好，谢谢。
```

---

## 配置文件

### 位置

配置文件位于:
- **Linux/macOS**: `~/.config/subtitle-forge/config.yaml`
- **Windows**: `%APPDATA%\subtitle-forge\config.yaml`

### 查看配置

```bash
subtitle-forge config show
```

### 修改配置

```bash
# 设置 Whisper 模型
subtitle-forge config set whisper.model large-v3

# 设置翻译模型
subtitle-forge config set ollama.model qwen2.5:32b

# 设置并发数
subtitle-forge config set max_workers 4
```

### 重置配置

```bash
subtitle-forge config reset
```

### 配置选项说明

```yaml
# Whisper 语音识别设置
whisper:
  model: large-v3          # 模型名称: tiny, base, small, medium, large-v3
  device: cuda             # 设备: cuda 或 cpu
  compute_type: float16    # 计算精度: float16, int8_float16, int8
  beam_size: 5             # Beam search 大小
  vad_filter: true         # VAD 过滤

# Ollama 翻译设置
ollama:
  model: qwen2.5:14b       # 翻译模型
  host: http://localhost:11434  # Ollama 服务地址
  temperature: 0.3         # 生成温度
  max_batch_size: 10       # 批次大小

# 输出设置
output:
  encoding: utf-8          # 字幕编码
  keep_original: true      # 保留原始字幕
  bilingual: false         # 默认双语模式

# 全局设置
max_workers: 2             # 并发处理数
log_level: INFO            # 日志级别
```

### 翻译提示词配置

翻译质量很大程度上取决于发送给 AI 模型的提示词（Prompt）。subtitle-forge 允许你查看和自定义翻译提示词。

#### 查看当前提示词

```bash
subtitle-forge config show-prompt
```

这会显示当前使用的翻译提示词模板，并标注是默认模板还是自定义模板。

#### 导出提示词

将当前提示词导出到文件以便编辑:

```bash
subtitle-forge config export-prompt -o my_prompt.txt
```

#### 自定义提示词

1. 首先导出默认提示词:
```bash
subtitle-forge config export-prompt -o my_prompt.txt
```

2. 用文本编辑器修改 `my_prompt.txt`

3. 应用自定义提示词:
```bash
subtitle-forge config set-prompt -f my_prompt.txt
```

#### 重置为默认提示词

```bash
subtitle-forge config reset-prompt
```

#### 可用占位符

自定义提示词时，必须包含以下占位符（系统会自动替换为实际值）:

| 占位符 | 说明 | 示例值 |
|--------|------|--------|
| `{source_lang}` | 源语言名称（必需） | English |
| `{target_lang}` | 目标语言名称（必需） | Simplified Chinese |
| `{segments}` | 要翻译的字幕内容（必需） | [1] Hello... |
| `{context_before}` | 前文上下文（可选） | 前 3 条字幕 |
| `{context_after}` | 后文上下文（可选） | 后 2 条字幕 |

#### 自定义提示词示例

```
你是专业的影视字幕翻译专家。

翻译要求:
1. 保持对话的自然流畅
2. 捕捉角色的情感和语气
3. 保持字幕简洁，适合阅读
4. 每行保留 [数字] 前缀
5. 只输出翻译结果，不要解释
{context_before}
将以下字幕从 {source_lang} 翻译成 {target_lang}:
{segments}
{context_after}
翻译结果:
```

> **提示**: 好的提示词应该明确翻译风格（正式/口语）、目标受众、以及特定领域术语的处理方式。针对不同类型的视频（电影、纪录片、教程等）可以使用不同的提示词。

---

## 故障排查

### 问题：无法连接 Ollama

**症状:**
```
Cannot connect to Ollama
Connection refused
```

**解决方案:**

1. 确认 Ollama 正在运行:
```bash
ollama serve
```

2. 检查 Ollama 服务状态:
```bash
curl http://localhost:11434
```

3. 检查端口是否被占用:
```bash
# Linux/macOS
lsof -i :11434

# Windows
netstat -ano | findstr 11434
```

4. 检查防火墙设置，确保 11434 端口未被阻止。

---

### 问题：GPU 未检测

**症状:**
```
CUDA not available
Using CPU (slower)
```

**解决方案:**

1. 运行系统诊断:
```bash
subtitle-forge config check --verbose
```

2. 检查 PyTorch CUDA 支持:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

3. 确认安装了正确版本的 PyTorch（见 [GPU 加速设置](#gpu-加速设置)）

4. 检查 NVIDIA 驱动:
```bash
nvidia-smi
```

---

### 问题：sm_120 不兼容（RTX 50 系列）

**症状:**
```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation
```

**原因:** RTX 50 系列使用 Blackwell 架构 (sm_120)，需要 CUDA 12.8+。

**解决方案:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

### 问题：下载速度慢

**症状:** Whisper 模型下载非常缓慢

**解决方案:** 配置 HuggingFace 镜像（见 [HuggingFace 镜像配置](#huggingface-镜像配置中国用户)）

---

### 问题：模型下载中断

**症状:** 下载过程中网络中断

**解决方案:**

模型下载支持断点续传，直接重新运行命令即可:

```bash
# Ollama 模型
subtitle-forge config pull-model

# Whisper 模型会在下次处理时自动继续下载
subtitle-forge process video.mp4 -t zh
```

---

### 问题：翻译速度慢

**症状:** 翻译过程非常缓慢

**参考速度:**
- **有 GPU**: 约 10-20 条字幕/秒
- **无 GPU**: 约 1-3 条字幕/秒

**解决方案:**

1. 检查 GPU 状态:
```bash
subtitle-forge config check
```

2. 使用更小的翻译模型:
```bash
subtitle-forge config set ollama.model qwen2.5:7b
```

---

### 问题：转写速度慢

**症状:** 转写 1 小时视频需要 30+ 分钟

**参考速度 (有 GPU):**
- 1 小时视频: 约 2-5 分钟
- 3 小时视频: 约 5-15 分钟

**解决方案:**

1. 确认 GPU 正在使用:
```bash
# 处理时另开终端监控
nvidia-smi -l 1
```

2. 使用更小的 Whisper 模型:
```bash
subtitle-forge config set whisper.model medium
```

---

### 完整系统诊断

```bash
subtitle-forge config check --verbose
```

这会显示:
- Python 版本
- PyTorch 和 CUDA 版本
- GPU 信息和 VRAM
- Ollama 连接状态
- 已安装的模型

---

## 支持的语言

subtitle-forge 支持 Whisper 识别的 99+ 种语言，并可翻译为 Ollama 支持的任意语言。

### 常用语言代码

| 代码 | 语言 |
|------|------|
| `en` | English |
| `zh` | 简体中文 |
| `zh-TW` | 繁體中文 |
| `ja` | 日本語 |
| `ko` | 한국어 |
| `es` | Español |
| `fr` | Français |
| `de` | Deutsch |
| `ru` | Русский |
| `pt` | Português |
| `it` | Italiano |
| `ar` | العربية |
| `vi` | Tiếng Việt |
| `th` | ไทย |

---

## Whisper 模型选择

| 模型 | VRAM 需求 | 速度 | 准确度 |
|------|-----------|------|--------|
| tiny | ~1 GB | 最快 | 较低 |
| base | ~1.5 GB | 快 | 一般 |
| small | ~2.5 GB | 中等 | 较好 |
| medium | ~5 GB | 较慢 | 高 |
| large-v3 | ~6 GB | 最慢 | 最高 |

### 自动选择模型

```bash
# 根据 GPU VRAM 自动选择最佳模型
subtitle-forge transcribe video.mp4 --auto-model
```

### 手动指定模型

```bash
# 使用 large-v3 获得最高准确度
subtitle-forge process video.mp4 -t zh --model large-v3

# 使用 small 获得更快速度
subtitle-forge process video.mp4 -t zh --model small
```

---

## 更新方法

### 从 GitHub 更新

```bash
cd subtitle-forge
git pull origin main

# 清理缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 重新安装（如果依赖有变化）
pip install -e .
```

### 一键更新

```bash
cd subtitle-forge && git pull && find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null; pip install -e .
```

### 查看版本

```bash
subtitle-forge version
```
