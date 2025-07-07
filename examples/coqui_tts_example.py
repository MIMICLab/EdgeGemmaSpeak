#!/usr/bin/env python3
"""
Example of using Coqui TTS with AgentVox

This script demonstrates:
1. How to list available TTS models
2. How to use Coqui TTS engine
3. How to use voice cloning with speaker_wav
"""

import sys
import os

# Add parent directory to path to import agentvox
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentvox.voice_assistant import ModelConfig, AudioConfig, VoiceAssistant, CoquiTTSModule


def list_tts_models():
    """List all available TTS models"""
    print("Available Coqui TTS models:")
    print("=" * 60)
    
    models = CoquiTTSModule.list_models()
    
    # Filter multilingual models
    multilingual = [m for m in models if "multilingual" in m or "multi-dataset" in m]
    print("\nMultilingual models (support voice cloning):")
    for model in multilingual:
        print(f"  - {model}")
    
    # Filter Korean models
    korean = [m for m in models if "/ko/" in m or "korean" in m.lower()]
    if korean:
        print("\nKorean models:")
        for model in korean:
            print(f"  - {model}")
    
    print(f"\nTotal models: {len(models)}")


def demo_basic_tts():
    """Demo basic TTS without voice cloning"""
    print("\n" + "=" * 60)
    print("Demo: Basic Coqui TTS")
    print("=" * 60)
    
    # Configure for Coqui TTS
    model_config = ModelConfig(
        tts_engine="coqui",
        coqui_model="tts_models/multilingual/multi-dataset/xtts_v2",
        device="auto"  # Will auto-detect cuda/mps/cpu
    )
    
    audio_config = AudioConfig()
    
    # Create assistant
    assistant = VoiceAssistant(model_config, audio_config)
    
    # Test TTS
    test_text_ko = "안녕하세요. 저는 Coqui TTS를 사용하는 AI 어시스턴트입니다."
    test_text_en = "Hello. I am an AI assistant using Coqui TTS."
    
    print(f"\nSynthesizing Korean: {test_text_ko}")
    assistant.tts.speak(test_text_ko)
    
    print(f"\nSynthesizing English: {test_text_en}")
    assistant.tts.speak(test_text_en)


def demo_voice_cloning():
    """Demo voice cloning with speaker_wav"""
    print("\n" + "=" * 60)
    print("Demo: Voice Cloning with Coqui TTS")
    print("=" * 60)
    
    # Check if sample voice file exists
    speaker_wav_path = "sample_voice.wav"
    
    if not os.path.exists(speaker_wav_path):
        print(f"Error: Speaker voice sample '{speaker_wav_path}' not found.")
        print("Please provide a voice sample file for cloning.")
        print("\nTo record a sample:")
        print("  - Record 3-10 seconds of clear speech")
        print("  - Save as WAV or MP3 format")
        print("  - Place in current directory as 'sample_voice.wav'")
        return
    
    # Configure for voice cloning
    model_config = ModelConfig(
        tts_engine="coqui",
        coqui_model="tts_models/multilingual/multi-dataset/xtts_v2",
        speaker_wav=speaker_wav_path,
        device="auto"
    )
    
    audio_config = AudioConfig()
    
    # Create assistant
    assistant = VoiceAssistant(model_config, audio_config)
    
    # Test voice cloning
    test_text = "이것은 음성 복제 테스트입니다. 제공된 음성 샘플을 기반으로 합성되었습니다."
    
    print(f"\nCloning voice from: {speaker_wav_path}")
    print(f"Synthesizing: {test_text}")
    assistant.tts.speak(test_text)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Coqui TTS Example")
    parser.add_argument("--list-models", action="store_true", help="List available TTS models")
    parser.add_argument("--basic-demo", action="store_true", help="Run basic TTS demo")
    parser.add_argument("--voice-clone-demo", action="store_true", help="Run voice cloning demo")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_tts_models()
    elif args.basic_demo:
        demo_basic_tts()
    elif args.voice_clone_demo:
        demo_voice_cloning()
    else:
        # Run all demos
        list_tts_models()
        demo_basic_tts()
        # demo_voice_cloning()  # Commented out as it requires a voice sample


if __name__ == "__main__":
    main()