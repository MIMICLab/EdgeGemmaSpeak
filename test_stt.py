#!/usr/bin/env python3
"""Test STT with specific device"""

from RealtimeSTT import AudioToTextRecorder
import logging

# Set logging level
logging.basicConfig(level=logging.DEBUG)

try:
    print("Initializing STT with specific device...")
    
    # Try with specific device index
    recorder = AudioToTextRecorder(
        model="base",
        language="ko",
        device="cpu",
        spinner=False,
        use_microphone=True,
        level=logging.DEBUG,
        input_device_index=0  # Studio Display Microphone
    )
    
    print("STT initialized successfully!")
    print("Listening... (speak something)")
    
    text = recorder.text()
    print(f"Transcribed: {text}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()