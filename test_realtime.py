#!/usr/bin/env python3
"""Test script for RealtimeSTT and RealtimeTTS integration"""

import sys
from pathlib import Path

# Add the project directory to the Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from agentvox.voice_assistant import ModelConfig, AudioConfig, STTModule, TTSModule

def test_stt():
    """Test RealtimeSTT functionality"""
    print("Testing RealtimeSTT...")
    config = ModelConfig(stt_model="tiny", device="cpu")
    stt = STTModule(config)
    
    print("Please say something...")
    text = stt.transcribe_once()
    if text:
        print(f"You said: {text}")
        return text
    else:
        print("No speech detected")
        return None

def test_tts(text="안녕하세요. 리얼타임 티티에스 테스트입니다."):
    """Test RealtimeTTS functionality"""
    print("\nTesting RealtimeTTS with CoquiEngine...")
    config = ModelConfig(device="cpu")
    tts = TTSModule(config)
    
    print(f"Speaking: {text}")
    tts.speak_streaming(text)
    print("TTS test completed")

def test_full_integration():
    """Test full STT -> TTS pipeline"""
    print("\nTesting full integration...")
    print("Say something and I'll repeat it back to you:")
    
    # Initialize modules
    config = ModelConfig(stt_model="tiny", device="cpu")
    stt = STTModule(config)
    tts = TTSModule(config)
    
    # Get speech input
    text = stt.transcribe_once()
    if text:
        print(f"You said: {text}")
        print("Repeating back...")
        tts.speak_streaming(text)
    else:
        print("No speech detected")

if __name__ == "__main__":
    print("RealtimeSTT/TTS Test Script")
    print("=" * 50)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["stt", "tts", "full"], default="full",
                       help="Test mode: stt, tts, or full integration")
    parser.add_argument("--text", type=str, default=None,
                       help="Text to speak for TTS test")
    args = parser.parse_args()
    
    try:
        if args.test == "stt":
            test_stt()
        elif args.test == "tts":
            if args.text:
                test_tts(args.text)
            else:
                test_tts()
        else:
            test_full_integration()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()