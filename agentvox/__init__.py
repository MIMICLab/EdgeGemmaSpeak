"""
AgentVox - Edge-based voice assistant using Gemma LLM with STT and TTS capabilities
"""

from .voice_assistant import (
    VoiceAssistant,
    STTModule,
    LLMModule,
    TTSModule,
    AudioConfig,
    ModelConfig,
    main
)
from .record_speaker_wav import SpeakerRecorder

__version__ = "0.1.0"
__author__ = "MIMIC Lab"

__all__ = [
    "VoiceAssistant",
    "STTModule", 
    "LLMModule",
    "TTSModule",
    "AudioConfig",
    "ModelConfig",
    "main",
    "SpeakerRecorder"
]