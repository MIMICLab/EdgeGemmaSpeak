# Third Party Licenses

This project uses the following third-party libraries:

## RealtimeTTS
- **License**: MIT License
- **Source**: https://github.com/KoljaB/RealtimeTTS
- **Usage**: Real-time Text-to-Speech functionality with multiple engine support

RealtimeTTS provides a unified interface for multiple TTS engines including Azure Cognitive Services, Elevenlabs API, OpenAI TTS API, and Coqui XTTS.

## RealtimeSTT
- **License**: MIT License
- **Source**: https://github.com/KoljaB/RealtimeSTT
- **Usage**: Real-time Speech-to-Text functionality

RealtimeSTT provides low-latency, real-time speech recognition capabilities with support for multiple backends including faster-whisper.

## llama-cpp-python
- **License**: MIT License
- **Source**: https://github.com/abetlen/llama-cpp-python
- **Usage**: Local LLM inference

## coqui-tts (Coqui AI TTS) - via RealtimeTTS
- **License**: Mozilla Public License 2.0 (MPL-2.0)
- **Source**: https://github.com/coqui-ai/TTS
- **Usage**: Advanced Text-to-Speech functionality with voice cloning support (used through RealtimeTTS)

The Coqui-TTS library is licensed under MPL-2.0, which is a permissive copyleft license. This project uses it as an optional dependency through RealtimeTTS for advanced TTS features. The MPL-2.0 license requires that modifications to the Coqui-TTS library itself be made available under the same license, but does not affect the licensing of this project's code.

## Other Dependencies

Most other dependencies (numpy, torch, pygame, etc.) are licensed under permissive licenses (MIT, BSD, Apache 2.0) that are compatible with this project's MIT license.

For complete license information of all dependencies, please refer to their respective repositories and PyPI pages.