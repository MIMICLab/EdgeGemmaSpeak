#!/usr/bin/env python3
"""Test microphone access"""

import pyaudio
import numpy as np
import time

def test_microphone():
    p = pyaudio.PyAudio()
    
    # List devices
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"{i}: {info['name']} - channels: {info['maxInputChannels']}")
    
    # Try to open default input device
    try:
        print("\nTrying to open default microphone...")
        stream = p.open(format=pyaudio.paInt16,
                       channels=1,
                       rate=16000,
                       input=True,
                       frames_per_buffer=1024)
        
        print("Microphone opened successfully!")
        print("Recording for 3 seconds...")
        
        for i in range(0, int(16000 / 1024 * 3)):
            data = stream.read(1024)
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.sqrt(np.mean(audio_data**2))
            print(f"Volume: {'#' * int(volume/100)}")
            
        stream.stop_stream()
        stream.close()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
    
    p.terminate()

if __name__ == "__main__":
    test_microphone()