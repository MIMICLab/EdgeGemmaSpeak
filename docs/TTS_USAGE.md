# TTS (Text-to-Speech) Usage Guide

AgentVox supports two TTS engines: Edge-TTS (default) and Coqui TTS.

## Edge-TTS (Default)

Edge-TTS is the default engine, providing fast and high-quality speech synthesis.

### Basic Usage
```bash
# Use default Korean voice
agentvox

# Use specific voice
agentvox --voice ko-KR-InJoonNeural

# List available voices
agentvox --list-voices
```

### Available Korean Voices
- `ko-KR-HyunsuMultilingualNeural` (male, multilingual) - Default
- `ko-KR-InJoonNeural` (male)
- `ko-KR-SunHiNeural` (female)

## Coqui TTS

Coqui TTS provides advanced features including voice cloning and more model options.

### Installation
```bash
pip install TTS
```

### Basic Usage
```bash
# Use Coqui TTS with default model
agentvox --tts-engine coqui

# Use specific model
agentvox --tts-engine coqui --coqui-model tts_models/de/thorsten/tacotron2-DDC

# List available models
agentvox --list-tts-models
```

### Voice Cloning
Voice cloning allows you to synthesize speech in a specific person's voice.

```bash
# Use voice cloning with a sample file
agentvox --tts-engine coqui --speaker-wav path/to/voice_sample.wav
```

#### Requirements for Voice Sample:
- Format: WAV or MP3
- Duration: 3-10 seconds of clear speech
- Quality: Good audio quality without background noise
- Content: Natural speech in the target language

### Recommended Models

#### Multilingual (with voice cloning support):
- `tts_models/multilingual/multi-dataset/xtts_v2` (Default, best quality)

#### Korean:
- Check available models with `agentvox --list-tts-models`

#### Other Languages:
- English: `tts_models/en/ljspeech/tacotron2-DDC`
- Japanese: `tts_models/ja/kokoro/tacotron2`
- Chinese: Check model list for available options

### Device Selection
Both Edge-TTS and Coqui TTS use the same device setting:

```bash
# Auto-detect (default)
agentvox --device auto

# Force CPU
agentvox --tts-engine coqui --device cpu

# Use CUDA GPU
agentvox --tts-engine coqui --device cuda

# Use Apple Silicon (MPS)
agentvox --tts-engine coqui --device mps
```

### Language Settings
The TTS language is automatically set based on the STT language:

```bash
# Korean TTS (default)
agentvox --tts-engine coqui --stt-language ko

# English TTS
agentvox --tts-engine coqui --stt-language en

# Japanese TTS
agentvox --tts-engine coqui --stt-language ja
```

## Comparison: Edge-TTS vs Coqui TTS

| Feature | Edge-TTS | Coqui TTS |
|---------|----------|-----------|
| Speed | Fast | Slower (depends on model) |
| Quality | High | Very High |
| Voice Cloning | ❌ | ✅ |
| Offline | ❌ | ✅ |
| Model Options | Limited | Many |
| Resource Usage | Low | High |

## Examples

### Example 1: Basic Conversation with Coqui TTS
```bash
agentvox --tts-engine coqui
```

### Example 2: Voice Cloning
```bash
# Record your voice sample first
# Then use it for synthesis
agentvox --tts-engine coqui --speaker-wav my_voice.wav
```

### Example 3: Different Language Models
```bash
# German TTS
agentvox --tts-engine coqui --coqui-model tts_models/de/thorsten/tacotron2-DDC --stt-language de

# Japanese TTS  
agentvox --tts-engine coqui --coqui-model tts_models/ja/kokoro/tacotron2 --stt-language ja

# English TTS with specific device
agentvox --tts-engine coqui --stt-language en --device cuda
```

## Troubleshooting

### Coqui TTS Installation Issues
If you encounter issues installing TTS:
```bash
# Update pip
pip install --upgrade pip

# Install with specific dependencies
pip install TTS torch torchvision torchaudio
```

### Memory Issues
Some Coqui models require significant memory:
- Use smaller models if running out of memory
- Use CPU instead of GPU for lower memory usage
- Close other applications

### Voice Quality Issues
- Ensure your speaker_wav file is high quality
- Use longer samples (5-10 seconds) for better cloning
- Try different models for different languages