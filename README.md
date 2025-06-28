# 🎙️ Edge Gemma Speak
[![PyPI Status](https://badge.fury.io/py/edge-gemma-speak.svg)](https://badge.fury.io/py/edge-gemma-speak)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/MIMICLab/EdgeGemmaSpeak/blob/main/LICENSE)
[![Downloads](https://pepy.tech/badge/edge-gemma-speak)](https://pepy.tech/project/edge-gemma-speak)

Edge-based voice assistant using Gemma LLM with Speech-to-Text and Text-to-Speech capabilities

## Key Features

- **Speech Recognition (STT)**: High-speed speech recognition using Faster Whisper
- **Conversational AI (LLM)**: Local LLM based on Llama.cpp (Gemma 3 12B)
- **Speech Synthesis (TTS)**: Fast response with Edge-TTS streaming
- **Complete Offline Operation**: All processing is done locally, ensuring privacy

## Installation

### 1. Install via pip

```bash
pip install edge-gemma-speak
```

Or install from source:

```bash
git clone https://github.com/yourusername/edge_gemma_speak.git
cd edge_gemma_speak
pip install -e .
```

#### For NVIDIA CUDA Users

If you have an NVIDIA GPU and want to use CUDA acceleration, you need to rebuild llama-cpp-python with CUDA support:

```bash
# Rebuild llama-cpp-python with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

This will significantly improve LLM inference performance on NVIDIA GPUs.

### 2. Download Model

```bash
# Automatically download Gemma model (~7GB)
edge-gemma-speak --download-model
```

The model will be saved in `~/.edge_gemma_speak/models/` directory.

## Usage

### Basic Usage

```bash
# Start voice conversation
edge-gemma-speak
```

Speak into your microphone and the AI will respond with voice.

### Voice Selection

```bash
# List all available voices
edge-gemma-speak --list-voices

# Use preset voices
edge-gemma-speak --voice male       # Korean male voice
edge-gemma-speak --voice female     # Korean female voice
edge-gemma-speak --voice multilingual  # Korean multilingual male (default)

# Use any Edge-TTS voice directly
edge-gemma-speak --voice en-US-JennyNeural
edge-gemma-speak --voice ja-JP-NanamiNeural
edge-gemma-speak --voice zh-CN-XiaoxiaoNeural
```

### Advanced Configuration

#### STT (Speech Recognition) Parameters

```bash
# Recognize speech in different languages
edge-gemma-speak --stt-language en

# Increase beam size for more accurate recognition (default: 5)
edge-gemma-speak --stt-beam-size 10

# Adjust VAD sensitivity (default: 0.5)
edge-gemma-speak --stt-vad-threshold 0.3

# Adjust minimum speech duration in ms (default: 250)
edge-gemma-speak --stt-vad-min-speech-duration 200

# Adjust minimum silence duration in ms (default: 1000)
edge-gemma-speak --stt-vad-min-silence-duration 800

# Change Whisper model size (tiny, base, small, medium, large)
edge-gemma-speak --stt-model small
```

#### LLM (Language Model) Parameters

```bash
# Generate longer responses (default: 512)
edge-gemma-speak --llm-max-tokens 1024

# More creative responses (higher temperature, default: 0.7)
edge-gemma-speak --llm-temperature 0.9

# More conservative responses (lower temperature)
edge-gemma-speak --llm-temperature 0.3

# Adjust context size (default: 4096)
edge-gemma-speak --llm-context-size 8192

# Adjust top-p sampling (default: 0.95)
edge-gemma-speak --llm-top-p 0.9
```

#### Device Configuration

```bash
# Auto-detect best available device (default)
edge-gemma-speak

# Explicitly use CPU
edge-gemma-speak --device cpu

# Explicitly use CUDA GPU
edge-gemma-speak --device cuda

# Explicitly use Apple Silicon MPS
edge-gemma-speak --device mps
```

The system automatically detects the best available device:
- NVIDIA GPU with CUDA → `cuda`
- Apple Silicon → `mps`
- Otherwise → `cpu`

### Combined Examples

```bash
# English female voice + English recognition + longer responses
edge-gemma-speak --voice en-US-JennyNeural --stt-language en --llm-max-tokens 1024

# Japanese voice + high accuracy STT + creative responses
edge-gemma-speak --voice ja-JP-NanamiNeural --stt-beam-size 10 --llm-temperature 0.9

# Use custom model path
edge-gemma-speak --model /path/to/your/model.gguf
```

## Python API Usage

```python
from edge_gemma_speak import VoiceAssistant, ModelConfig, AudioConfig

# Configuration
model_config = ModelConfig(
    stt_model="base",
    llm_temperature=0.7,
    tts_voice="en-US-JennyNeural"  # English female voice
)

audio_config = AudioConfig()

# Initialize voice assistant
assistant = VoiceAssistant(model_config, audio_config)

# Start conversation
assistant.run_conversation_loop()
```

### Using Individual Modules

```python
from edge_gemma_speak import STTModule, LLMModule, TTSModule, ModelConfig

config = ModelConfig()

# STT (Speech to Text)
stt = STTModule(config)
text = stt.transcribe("audio.wav")

# LLM (Generate text response)
llm = LLMModule(config)
response = llm.generate_response(text)

# TTS (Text to Speech)
tts = TTSModule(config)
tts.speak(response)
```

## Available Commands During Conversation

- **"exit"** or **"종료"**: Exit the program
- **"reset"** or **"초기화"**: Reset conversation history
- **"history"** or **"대화 내역"**: View conversation history

## System Requirements

- Python 3.8 or higher
- macOS (with MPS support), Linux, Windows
- Minimum 8GB RAM (16GB recommended)
- Approximately 7GB disk space (for model storage)

### Required Packages

- torch >= 2.0.0
- faster-whisper
- llama-cpp-python
- edge-tts
- numpy
- speech_recognition
- pygame
- sounddevice
- soundfile
- pyaudio

## Project Structure

```
edge_gemma_speak/
├── edge_gemma_speak/              # Package directory
│   ├── __init__.py               # Package initialization
│   ├── voice_assistant.py        # Main module
│   └── cli.py                    # CLI interface
├── setup.py                      # Package setup
├── pyproject.toml               # Build configuration
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
└── .gitignore                   # Git ignore file
```

## Troubleshooting

### PyAudio Installation Error

macOS:
```bash
brew install portaudio
pip install pyaudio
```

Linux:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

Windows:
```bash
# Visual Studio Build Tools required
pip install pipwin
pipwin install pyaudio
```

### Out of Memory

For large LLM models:
- Use smaller quantized models
- Reduce context size: `--llm-context-size 2048`
- Use CPU mode: `--device cpu`

### Microphone Recognition Issues

- Check microphone permissions in system settings
- Close other audio applications
- Adjust VAD threshold: `--stt-vad-threshold 0.3`
- Reduce silence duration for faster response: `--stt-vad-min-silence-duration 500`

### Model File Not Found

```bash
# Download model
edge-gemma-speak --download-model

# Or download directly
wget https://huggingface.co/tgisaturday/Docsray/resolve/main/gemma-3-12b-it-GGUF/gemma-3-12b-it-Q4_K_M.gguf \
  -O ~/.edge_gemma_speak/models/gemma-3-12b-it-Q4_K_M.gguf
```

## Performance Optimization

### Improve Response Speed

1. **Use smaller STT model**: `--stt-model tiny` or `base`
2. **Limit LLM response length**: `--llm-max-tokens 256`
3. **Reduce beam size**: `--stt-beam-size 3`

### GPU Acceleration

- **macOS**: Automatic MPS support (`--device mps`)
- **NVIDIA GPU**: CUDA support (`--device cuda`)
- **AMD GPU**: Requires PyTorch with ROCm support

## Developer Information

Developed by MimicLab at Sogang University

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

This project uses several third-party libraries:
- **edge-tts**: LGPL-3.0 License (for TTS functionality)
- **faster-whisper**: MIT License (for STT functionality)
- **llama-cpp-python**: MIT License (for LLM inference)
- **Gemma Model**: Check the model provider's license terms

For complete third-party license information, see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

**Note on edge-tts**: The edge-tts library is licensed under LGPL-3.0. This project uses it as a library dependency without modifications. Users are free to replace edge-tts with their own version if desired. The LGPL-3.0 license of edge-tts does not affect the MIT licensing of this project's source code.

## Contributing

Issues and Pull Requests are always welcome!

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/edge_gemma_speak.git
cd edge_gemma_speak

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## Multilingual Support

Edge Gemma Speak supports multiple languages through Edge-TTS. You can use voices in various languages:

- **English**: en-US, en-GB, en-AU, en-CA, en-IN
- **Japanese**: ja-JP
- **Chinese**: zh-CN, zh-TW, zh-HK
- **Spanish**: es-ES, es-MX
- **French**: fr-FR, fr-CA
- **German**: de-DE
- **Korean**: ko-KR
- And many more...

Use `--list-voices` to see all available voices and their language codes.