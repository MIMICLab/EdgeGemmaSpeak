#!/usr/bin/env python3
"""Minimal example of RealtimeSTT + RealtimeTTS integration"""

def main():
    # Required for multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    from RealtimeSTT import AudioToTextRecorder
    from RealtimeTTS import TextToAudioStream, CoquiEngine
    import time
    
    print("Minimal RealtimeSTT + RealtimeTTS Example")
    print("=" * 40)
    
    # Initialize STT
    print("Initializing STT...")
    stt_recorder = AudioToTextRecorder(
        model="tiny",
        language="ko",
        device="cpu"
    )
    
    # Initialize TTS
    print("Initializing TTS (first run downloads model)...")
    tts_engine = CoquiEngine(
        device="cpu",
        language="ko"
    )
    tts_stream = TextToAudioStream(tts_engine)
    
    print("\nReady! Say something and I'll repeat it back.")
    print("Press Ctrl+C to exit.\n")
    
    try:
        while True:
            # Get speech input
            print("Listening...")
            text = stt_recorder.text()
            print(f"You said: {text}")
            
            # Speak it back
            print("Speaking...")
            tts_stream.feed(text)
            tts_stream.play_async()
            
            # Wait for speech to complete
            while tts_stream.is_playing():
                time.sleep(0.1)
                
            print("-" * 40)
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        stt_recorder.shutdown()

if __name__ == "__main__":
    main()