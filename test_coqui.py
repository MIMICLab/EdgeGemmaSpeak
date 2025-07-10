#!/usr/bin/env python3
"""Test CoquiEngine directly"""

from RealtimeTTS import TextToAudioStream, CoquiEngine
import time

def main():
    # Simple test text
    text = "안녕하세요. 코키 엔진 테스트입니다."

    print("Initializing CoquiEngine...")
    print("Note: First run will download the model (~1.86GB)")

    try:
        # Initialize with minimal parameters
        engine = CoquiEngine(
            device="cpu",
            language="ko"
        )
        
        print("Creating audio stream...")
        stream = TextToAudioStream(engine)
        
        print(f"Synthesizing: {text}")
        stream.feed(text)
        stream.play()
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()