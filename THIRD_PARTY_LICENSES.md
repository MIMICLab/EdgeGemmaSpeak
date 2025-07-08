# Third Party Licenses

This project uses the following third-party libraries:

## edge-tts
- **License**: LGPL-3.0 (GNU Lesser General Public License v3.0)
- **Source**: https://github.com/rany2/edge-tts
- **Usage**: Text-to-Speech functionality

The edge-tts library is licensed under LGPL-3.0, which allows this project to use it as a library dependency while maintaining a different license (MIT) for the main project code. Users are free to replace the edge-tts library with their own modified version if desired.

## faster-whisper
- **License**: MIT License
- **Source**: https://github.com/guillaumekln/faster-whisper
- **Usage**: Speech-to-Text functionality

## llama-cpp-python
- **License**: MIT License
- **Source**: https://github.com/abetlen/llama-cpp-python
- **Usage**: Local LLM inference

## coqui-tts (Coqui AI TTS)
- **License**: Mozilla Public License 2.0 (MPL-2.0)
- **Source**: https://github.com/coqui-ai/TTS
- **Usage**: Advanced Text-to-Speech functionality with voice cloning support

The Coqui-TTS library is licensed under MPL-2.0, which is a permissive copyleft license. This project uses it as an optional dependency for advanced TTS features. The MPL-2.0 license requires that modifications to the Coqui-TTS library itself be made available under the same license, but does not affect the licensing of this project's code.

## Other Dependencies

Most other dependencies (numpy, torch, pygame, etc.) are licensed under permissive licenses (MIT, BSD, Apache 2.0) that are compatible with this project's MIT license.

For complete license information of all dependencies, please refer to their respective repositories and PyPI pages.