# AgentVox v0.2.0 Release Notes

## 🎉 Major Updates

### Real-time Streaming Architecture
- **Migrated to RealtimeSTT/RealtimeTTS**: Replaced traditional STT/TTS engines with streaming alternatives for significantly lower latency
- **Live Transcription**: See what you're saying in real-time as you speak
- **Streaming TTS**: Audio synthesis starts playing while still generating, resulting in faster response times

### Performance Improvements
- **Better Voice Activity Detection**: More accurate detection of when users start and stop speaking
- **Reduced Latency**: Faster overall response times due to streaming architecture
- **Configurable TTS Speed**: New `--tts-speed` parameter to adjust speech synthesis speed

### Technical Changes
- Updated dependencies to use `realtimestt` and `realtimetts[coqui]`
- Improved error handling and stability
- Better resource management for audio streams

### Bug Fixes
- Fixed various bugs from v0.1.1
- Improved stability during long conversations
- Better handling of edge cases in voice detection

## 🚀 Quick Start

```bash
# Install or upgrade
pip install --upgrade agentvox

# Run with default settings
agentvox

# Run with custom TTS speed
agentvox --tts-speed 1.5
```

## 📋 Dependencies Updated
- Added `realtimestt` for streaming speech-to-text
- Added `realtimetts[coqui]` for streaming text-to-speech
- Updated `torch` to >=2.6.0
- Added Korean language support dependencies (`hangul-romanize`, `mecab-python3`, `unidic-lite`)

## 🔧 Breaking Changes
None - This release maintains backward compatibility with v0.1.1

## 📝 Notes
- All processing remains completely offline
- Voice cloning functionality is preserved and enhanced
- Coqui XTTS v2 model is used for high-quality voice synthesis

## 🙏 Acknowledgments
Thanks to all contributors and users who provided feedback for this release!

---
For more information, visit our [GitHub repository](https://github.com/yourusername/agentvox)