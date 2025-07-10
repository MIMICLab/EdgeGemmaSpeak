#!/usr/bin/env python3
"""Simple test script for RealtimeSTT and RealtimeTTS"""

def test_stt():
    """Test RealtimeSTT functionality"""
    from RealtimeSTT import AudioToTextRecorder
    
    print("Testing RealtimeSTT...")
    print("Initializing recorder...")
    
    recorder = AudioToTextRecorder(
        model="tiny",
        language="ko",
        compute_type="int8",
        device="cpu"
    )
    
    print("Please say something...")
    
    text = recorder.text()
    print(f"You said: {text}")
    
    recorder.shutdown()
    return text

def test_tts(text="안녕하세요. 리얼타임 티티에스 테스트입니다."):
    """Test RealtimeTTS functionality"""
    from RealtimeTTS import TextToAudioStream, CoquiEngine
    
    print("\nTesting RealtimeTTS with CoquiEngine...")
    print("Initializing TTS engine...")
    
    # Use a simpler model that doesn't require downloading large files
    engine = CoquiEngine(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        device="cpu",
        language="ko"
    )
    
    stream = TextToAudioStream(engine)
    
    print(f"Speaking: {text}")
    stream.feed(text)
    stream.play_async()
    
    import time
    while stream.is_playing():
        time.sleep(0.1)
    
    print("TTS test completed")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "stt":
        test_stt()
    elif len(sys.argv) > 1 and sys.argv[1] == "tts":
        test_tts()
    else:
        # Test STT first
        text = test_stt()
        
        # Then test TTS with the recognized text
        if text:
            test_tts(text)