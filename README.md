# 🎙️ AgentVox
[![PyPI Status](https://badge.fury.io/py/agentvox.svg)](https://badge.fury.io/py/agentvox)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/MIMICLab/AgentVox/blob/main/LICENSE)
[![Downloads](https://pepy.tech/badge/agentvox)](https://pepy.tech/project/agentvox)

Edge-based voice assistant using Gemma LLM with real-time Speech-to-Text and Text-to-Speech capabilities

## 🆕 What's New (v0.2.0)

- **Real-time Streaming STT/TTS**: Replaced traditional STT/TTS with RealtimeSTT and RealtimeTTS for lower latency
- **Live Transcription**: See what you're saying in real-time as you speak
- **Streaming TTS**: Faster response times with streaming audio synthesis
- **Improved Performance**: Better voice activity detection and faster processing
- **Configurable TTS Speed**: Adjust speech speed with `--tts-speed` parameter

## Key Features

- **Real-time Speech Recognition (STT)**: Live transcription using RealtimeSTT with Whisper
- **Conversational AI (LLM)**: Local LLM based on Llama.cpp (Gemma 3 12B)
- **Streaming Speech Synthesis (TTS)**: Real-time voice synthesis with RealtimeTTS using Coqui XTTS v2
- **Complete Offline Operation**: All processing is done locally, ensuring privacy
- **Voice Cloning**: Clone any voice with a short audio sample

## Installation

### 1. Install via pip

```bash
pip install agentvox
```

Or install from source:

```bash
git clone https://github.com/yourusername/agentvox.git
cd agentvox
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
agentvox --download-model
```

The model will be saved in `~/.agentvox/models/` directory.

## Usage

### Basic Usage

```bash
# Start voice conversation
agentvox
```

Speak into your microphone and the AI will respond with voice.

### Voice Configuration

```bash
# Voice cloning with Coqui-TTS (default engine)
agentvox --speaker-wav speaker_sample.wav

# Record your own voice sample for cloning
agentvox --record-speaker
# Then use the recorded sample
agentvox --speaker-wav speaker_ko.wav

# Adjust TTS speed (1.0 is normal, higher is faster)
agentvox --tts-speed 1.5  # 50% faster
agentvox --tts-speed 1.3  # 30% faster (default)
agentvox --tts-speed 1.0  # Normal speed
agentvox --tts-speed 0.8  # 20% slower
```

### Advanced Configuration

#### STT (Speech Recognition) Parameters

```bash
# Recognize speech in different languages
agentvox --stt-language en

# Increase beam size for more accurate recognition (default: 5)
agentvox --stt-beam-size 10

# Adjust VAD sensitivity (default: 0.5)
agentvox --stt-vad-threshold 0.3

# Adjust minimum speech duration in ms (default: 250)
agentvox --stt-vad-min-speech-duration 200

# Adjust minimum silence duration in ms (default: 1000)
agentvox --stt-vad-min-silence-duration 800

# Change Whisper model size (tiny, base, small, medium, large)
agentvox --stt-model small
```

#### LLM (Language Model) Parameters

```bash
# Generate longer responses (default: 512)
agentvox --llm-max-tokens 1024

# More creative responses (higher temperature, default: 0.7)
agentvox --llm-temperature 0.9

# More conservative responses (lower temperature)
agentvox --llm-temperature 0.3

# Adjust context size (default: 4096)
agentvox --llm-context-size 8192

# Adjust top-p sampling (default: 0.95)
agentvox --llm-top-p 0.9
```

#### Device Configuration

```bash
# Auto-detect best available device (default)
agentvox

# Explicitly use CPU
agentvox --device cpu

# Explicitly use CUDA GPU
agentvox --device cuda

# Explicitly use Apple Silicon MPS
agentvox --device mps
```

The system automatically detects the best available device:
- NVIDIA GPU with CUDA → `cuda`
- Apple Silicon → `mps`
- Otherwise → `cpu`

### Combined Examples

```bash
# English recognition + fast speech + longer responses
agentvox --stt-language en --tts-speed 1.5 --llm-max-tokens 1024

# High accuracy STT + creative responses + voice cloning
agentvox --stt-beam-size 10 --llm-temperature 0.9 --speaker-wav voice_sample.wav

# Use custom model path with fast TTS
agentvox --model /path/to/your/model.gguf --tts-speed 1.4
```

## Python API Usage

```python
from agentvox import VoiceAssistant, ModelConfig, AudioConfig

# Configuration
model_config = ModelConfig(
    stt_model="base",
    llm_temperature=0.7,
    tts_speed=1.0,  # Adjust TTS speed
    speaker_wav="voice_sample.wav"  # Optional: voice cloning
)

audio_config = AudioConfig()

# Initialize voice assistant
assistant = VoiceAssistant(model_config, audio_config)

# Start conversation
assistant.run_conversation_loop()
```

### Using Individual Modules

```python
from agentvox import STTModule, LLMModule, TTSModule, ModelConfig

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
- realtimestt (Real-time speech-to-text)
- realtimetts[coqui] (Real-time text-to-speech with Coqui engine)
- llama-cpp-python
- numpy
- pygame
- sounddevice
- soundfile
- pyaudio
- hangul-romanize (for Korean language support)

## Project Structure

```
agentvox/
├── agentvox/              # Package directory
│   ├── __init__.py               # Package initialization
│   ├── voice_assistant.py        # Main module
│   ├── cli.py                    # CLI interface
│   └── record_speaker_wav.py     # Voice recording module
├── setup.py                      # Package setup
├── pyproject.toml               # Build configuration
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
└── .gitignore                   # Git ignore file
```

## Troubleshooting

### First Run Model Download

On first run, the Coqui TTS model (XTTS v2, ~1.86GB) will be automatically downloaded. This only happens once.

### Multiprocessing Errors on macOS/Windows

If you encounter multiprocessing errors, ensure your script uses:
```python
if __name__ == "__main__":
    # Your code here
```

### ctranslate2 Version Issues

If you get ctranslate2 compatibility errors:
```bash
pip install ctranslate2==4.4.0
```

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

### Speaker Echo/Feedback Issues

If the TTS output is being picked up by the microphone:
- Use headphones instead of speakers
- Reduce speaker volume
- Increase the distance between microphone and speakers
- The system automatically pauses STT during TTS playback to minimize echo

### Model File Not Found

```bash
# Download model
agentvox --download-model

# Or download directly
wget https://huggingface.co/tgisaturday/Docsray/resolve/main/gemma-3-12b-it-GGUF/gemma-3-12b-it-Q4_K_M.gguf \
  -O ~/.agentvox/models/gemma-3-12b-it-Q4_K_M.gguf
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
- **RealtimeSTT**: MIT License (for real-time speech-to-text)
- **RealtimeTTS**: MIT License (for real-time text-to-speech)
- **coqui-tts**: Mozilla Public License 2.0 (used by RealtimeTTS for voice synthesis)
- **faster-whisper**: MIT License (used by RealtimeSTT for speech recognition)
- **llama-cpp-python**: MIT License (for LLM inference)
- **Gemma Model**: Check the model provider's license terms

For complete third-party license information, see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

**Note on copyleft licenses**: 
- The coqui-tts library (MPL-2.0) is used as a dependency through RealtimeTTS. The MPL-2.0 license only requires that modifications to coqui-tts itself be shared, not your application code.
- The MPL-2.0 license of coqui-tts does not affect the MIT licensing of this project's source code.

## Contributing

Issues and Pull Requests are always welcome!

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/agentvox.git
cd agentvox

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## Multilingual Support

AgentVox supports multiple languages through both STT and TTS:

### Speech Recognition (STT)
Set the language with `--stt-language`:
- **Korean**: ko (default)
- **English**: en
- **Japanese**: ja
- **Chinese**: zh
- **Spanish**: es
- **French**: fr
- **German**: de
- And many more...

### Text-to-Speech (TTS)
The Coqui XTTS v2 model supports multiple languages automatically. For best results:
- Use voice cloning with a native speaker's voice sample
- The model will automatically detect and use the appropriate language

Example:
```bash
# English conversation with cloned voice
agentvox --stt-language en --speaker-wav english_voice.wav

# Japanese conversation
agentvox --stt-language ja --speaker-wav japanese_voice.wav
```