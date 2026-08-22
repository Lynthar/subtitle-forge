# subtitle-forge

[English](README.md) | **简体中文**

在本地用 AI 生成和翻译视频字幕，不依赖任何云服务。

## 特性

- **本地推理** —— 转写和翻译都跑在自己机器上，音频与字幕文本不发往任何云服务（[网络边界](docs/user-guide.md#五隐私与网络边界)）
- **语音识别** —— faster-whisper，支持 99 种以上语言，可选 WhisperX 强制对齐
- **AI 翻译** —— 走本地 LLM（Ollama），带上下文
- **GPU 加速** —— 可选，支持 CUDA
- **批量处理** —— 多个视频一起跑，并发数可配
- **HTTP 服务模式** —— 可选的 REST 接口，给媒体服务器和自动化用

## 快速开始

### 1. 装依赖

```bash
# macOS
brew install ffmpeg ollama

# Ubuntu/Debian
sudo apt install ffmpeg
curl -fsSL https://ollama.ai/install.sh | sh

# Windows：分别从 https://ffmpeg.org 和 https://ollama.ai 装
```

### 2. 起 Ollama

```bash
ollama serve
```

### 3. 装 subtitle-forge

```bash
# 基础安装
pip install -e .

# 强烈建议加上：WhisperX 提供 wav2vec2 强制对齐，词级时间戳才准。
# 不装的话时间轴会退回 faster-whisper 自己的词级时间戳，精度低一档。
pip install -e '.[whisperx]'
```

### 4. 跑一遍配置向导

```bash
subtitle-forge quickstart
```

向导会检查 ffmpeg / Ollama / GPU 是否就位，并把默认的 Whisper 与 Ollama 模型先下下来——省得第一次真跑的时候卡在几个 GB 的下载上。

### 5. 生成字幕

```bash
subtitle-forge process video.mp4 -t zh
```

## 常见用法

```bash
# 生成单语字幕
subtitle-forge process video.mp4 -t zh        # 中文
subtitle-forge process video.mp4 -t ja        # 日文

# 一次生成多种语言
subtitle-forge process video.mp4 -t zh -t ja -t ko

# 双语字幕
subtitle-forge process video.mp4 -t zh --bilingual

# 批量处理
subtitle-forge batch ./videos/ -t zh
subtitle-forge batch ./videos/ -t zh --recursive

# 只转写，不翻译
subtitle-forge transcribe video.mp4

# 翻译现成的字幕文件
subtitle-forge translate video.en.srt -t zh

# 感觉哪里不对时用——会留下 run.log 和 translation_failures.json
subtitle-forge process video.mp4 -t zh --save-debug-log
```

## 配置

```bash
subtitle-forge config show                          # 看全部设置
subtitle-forge config set whisper.model large-v3    # 改一项
subtitle-forge config check --verbose               # 系统诊断
```

配置文件在 `~/.config/subtitle-forge/config.yaml`（Windows 是 `%APPDATA%\subtitle-forge\config.yaml`）。每个字段的默认值和这么定的理由，见 [`config/default.yaml`](config/default.yaml)。

## 隐私与网络

推理是在本地，但「本地优先」不等于「从不联网」：

- **首次下模型要联网** —— Whisper 权重从 HuggingFace 拿，翻译模型从 Ollama 拿。之后就全程离线了。
- **`ollama.host` 可以指向另一台机器。** 默认是 `localhost`；改了它，字幕文本就发去那台主机了。
- **`--save-debug-log` / `--save-failed-log` 会把对白写进磁盘** —— 失败报告里含原始字幕文本，发给别人之前先读一遍。
- **输出默认写在源视频旁边**，文件权限跟着系统走。

细节、遥测关闭方式、以及完全离线部署的检查清单：[隐私与网络边界](docs/user-guide.md#五隐私与网络边界)。

## 文档

**[使用指南](docs/user-guide.md)** 是完整参考——分平台安装、按 RTX 系列的 GPU/CUDA 配置、命令参考、HTTP 服务模式、隐私边界、排障。

## 环境要求

- Python 3.9+
- ffmpeg
- Ollama
- NVIDIA 显卡（可选，用于加速）

## 许可证

MIT License
