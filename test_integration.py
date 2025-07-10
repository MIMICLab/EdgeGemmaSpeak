#!/usr/bin/env python3
"""Integration test for RealtimeSTT and RealtimeTTS with AgentVox"""

import sys
from pathlib import Path

# Add the project directory to the Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def main():
    # Set up multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # Import modules after multiprocessing setup
    from agentvox.voice_assistant import ModelConfig, AudioConfig, VoiceAssistant
    
    print("AgentVox Integration Test")
    print("=" * 50)
    print("This will test the full voice assistant with RealtimeSTT and RealtimeTTS")
    print("Commands: 'exit/종료' to quit, 'reset/초기화' to reset conversation")
    print("=" * 50)
    
    # Initialize with minimal configuration
    audio_config = AudioConfig()
    model_config = ModelConfig(
        stt_model="tiny",  # Use tiny model for faster testing
        device="cpu",
        tts_engine="coqui"
    )
    
    try:
        # Initialize voice assistant
        assistant = VoiceAssistant(model_config, audio_config)
        
        # Run conversation loop
        assistant.run_conversation_loop()
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()