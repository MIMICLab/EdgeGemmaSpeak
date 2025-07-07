"""
Command-line interface for agentvox
"""

import argparse
import sys
from pathlib import Path
from .voice_assistant import main as voice_assistant_main, ModelConfig, AudioConfig


def download_model():
    """Download the default Gemma model"""
    import subprocess
    import sys
    import os
    
    model_dir = Path.home() / ".agentvox" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_filename = "gemma-3-12b-it-Q4_K_M.gguf"
    model_path = model_dir / model_filename
    
    if model_path.exists():
        print(f"✓ Model already exists at {model_path}")
        return
    
    model_url = "https://huggingface.co/tgisaturday/Docsray/resolve/main/gemma-3-12b-it-GGUF/gemma-3-12b-it-Q4_K_M.gguf"
    
    print(f"Downloading Gemma model to {model_path}")
    print("This may take a while depending on your internet connection...")
    print()
    
    # Try wget first, then curl
    try:
        # Check if wget is available
        result = subprocess.run(["which", "wget"], capture_output=True, text=True)
        if result.returncode == 0:
            # Use wget
            cmd = ["wget", "-c", model_url, "-O", str(model_path)]
            print(f"Using wget: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            # Check if curl is available
            result = subprocess.run(["which", "curl"], capture_output=True, text=True)
            if result.returncode == 0:
                # Use curl
                cmd = ["curl", "-L", "-C", "-", model_url, "-o", str(model_path)]
                print(f"Using curl: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            else:
                # Fallback to Python urllib
                print("Neither wget nor curl found. Using Python to download...")
                import urllib.request
                from tqdm import tqdm
                
                def download_with_progress(url, path):
                    with urllib.request.urlopen(url) as response:
                        total_size = int(response.headers.get('Content-Length', 0))
                        
                        with open(path, 'wb') as f:
                            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                                while True:
                                    chunk = response.read(8192)
                                    if not chunk:
                                        break
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                
                download_with_progress(model_url, model_path)
        
        print(f"\n✓ Model downloaded successfully to {model_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Download failed with error: {e}")
        if model_path.exists():
            os.remove(model_path)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Download failed with error: {e}")
        if model_path.exists():
            os.remove(model_path)
        sys.exit(1)
    

def main():
    parser = argparse.ArgumentParser(description="AgentVox - Voice Assistant")
    parser.add_argument("--model", type=str, help="Path to GGUF model file")
    parser.add_argument("--stt-model", type=str, default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size for STT")
    parser.add_argument("--device", type=str, default=None,
                       choices=["cpu", "cuda", "mps", "auto"],
                       help="Device to use for inference (default: auto-detect)")
    parser.add_argument("--voice", type=str, default="multilingual",
                       help="TTS voice: use preset (male/female/multilingual) or any Edge-TTS voice name")
    parser.add_argument("--download-model", action="store_true",
                       help="Download the default Gemma model")
    parser.add_argument("--list-voices", action="store_true",
                       help="List all available Korean TTS voices")
    
    # STT 파라미터
    parser.add_argument("--stt-language", type=str, default="ko",
                       help="STT language (default: ko)")
    parser.add_argument("--stt-beam-size", type=int, default=5,
                       help="STT beam size for decoding (default: 5)")
    parser.add_argument("--stt-temperature", type=float, default=0.0,
                       help="STT temperature for sampling (default: 0.0)")
    parser.add_argument("--stt-vad-threshold", type=float, default=0.5,
                       help="STT VAD threshold (default: 0.5)")
    parser.add_argument("--stt-vad-min-speech-duration", type=int, default=250,
                       help="Minimum speech duration in ms (default: 250)")
    parser.add_argument("--stt-vad-min-silence-duration", type=int, default=1000,
                       help="Minimum silence duration in ms before cutting off (default: 1000)")
    
    # LLM 파라미터
    parser.add_argument("--llm-max-tokens", type=int, default=512,
                       help="Maximum tokens for LLM response (default: 512)")
    parser.add_argument("--llm-temperature", type=float, default=0.7,
                       help="LLM temperature for sampling (default: 0.7)")
    parser.add_argument("--llm-top-p", type=float, default=0.95,
                       help="LLM top-p for nucleus sampling (default: 0.95)")
    parser.add_argument("--llm-context-size", type=int, default=4096,
                       help="LLM context window size (default: 4096)")
    
    args = parser.parse_args()
    
    if args.download_model:
        download_model()
        sys.exit(0)
    
    if args.list_voices:
        import subprocess
        import json
        
        print("\nFetching available Edge-TTS voices...")
        print("=" * 70)
        
        try:
            # Run edge-tts --list-voices command
            result = subprocess.run(
                ["edge-tts", "--list-voices"], 
                capture_output=True, 
                text=True,
                check=True
            )
            
            # Parse the table output
            voices = []
            lines = result.stdout.strip().split('\n')
            
            # Skip header lines (first two lines are header and separator)
            if len(lines) > 2:
                for line in lines[2:]:
                    # Parse table columns
                    parts = line.split()
                    if len(parts) >= 2:
                        name = parts[0]
                        gender = parts[1]
                        
                        # Extract locale from name (format: lang-COUNTRY-NameNeural)
                        locale_parts = name.split('-')
                        if len(locale_parts) >= 2:
                            locale = f"{locale_parts[0]}-{locale_parts[1]}"
                        else:
                            locale = "Unknown"
                        
                        voices.append({
                            'name': name,
                            'gender': gender,
                            'locale': locale
                        })
            
            # Group by language
            voices_by_lang = {}
            for voice in voices:
                if 'locale' in voice:
                    lang = voice['locale'].split('-')[0]
                    if lang not in voices_by_lang:
                        voices_by_lang[lang] = []
                    voices_by_lang[lang].append(voice)
            
            # Display voices grouped by language
            print("\nAvailable Edge-TTS voices by language:")
            print("=" * 70)
            
            # Show Korean voices first
            if 'ko' in voices_by_lang:
                print("\n[Korean Voices]")
                for voice in voices_by_lang['ko']:
                    gender = voice.get('gender', 'Unknown')
                    print(f"  {voice['name']:<35} ({gender}, {voice.get('locale', 'Unknown')})")
            
            # Show other popular languages
            for lang in ['en', 'ja', 'zh', 'es', 'fr', 'de']:
                if lang in voices_by_lang:
                    lang_name = {
                        'en': 'English', 'ja': 'Japanese', 'zh': 'Chinese',
                        'es': 'Spanish', 'fr': 'French', 'de': 'German'
                    }.get(lang, lang.upper())
                    print(f"\n[{lang_name} Voices]")
                    for voice in voices_by_lang[lang][:5]:  # Show first 5 voices
                        gender = voice.get('gender', 'Unknown')
                        print(f"  {voice['name']:<35} ({gender}, {voice.get('locale', 'Unknown')})")
                    if len(voices_by_lang[lang]) > 5:
                        print(f"  ... and {len(voices_by_lang[lang]) - 5} more")
            
            # Show total count
            print(f"\n\nTotal voices available: {len(voices)}")
            print("\nUsage examples:")
            print("  agentvox --voice ko-KR-InJoonNeural")
            print("  agentvox --voice en-US-JennyNeural")
            print("  agentvox --voice ja-JP-NanamiNeural")
            
            print("\nQuick presets:")
            print("  agentvox --voice male     (Korean male)")
            print("  agentvox --voice female   (Korean female)")
            
        except subprocess.CalledProcessError:
            print("Error: Could not fetch Edge-TTS voices.")
            print("Make sure edge-tts is installed: pip install edge-tts")
        except Exception as e:
            print(f"Error: {e}")
            
        print("=" * 70)
        sys.exit(0)
    
    # Create configurations
    
    # Map voice choice to actual voice name
    voice_map = {
        "male": "ko-KR-InJoonNeural",
        "female": "ko-KR-SunHiNeural",
        "multilingual": "ko-KR-HyunsuMultilingualNeural"
    }
    
    # Use preset or direct voice name
    tts_voice = voice_map.get(args.voice, args.voice)
    
    # Extract language from TTS voice if STT language not explicitly set
    if args.stt_language == "ko" and tts_voice not in voice_map.values():
        # Extract language code from voice name (e.g., "en-US-JennyNeural" -> "en")
        voice_parts = tts_voice.split('-')
        if len(voice_parts) >= 2:
            voice_lang = voice_parts[0]
            # Only override if it's a known language code
            if voice_lang in ['en', 'ja', 'zh', 'es', 'fr', 'de', 'ko']:
                stt_language = voice_lang
                print(f"Notice: STT language automatically set to '{stt_language}' to match TTS voice '{tts_voice}'")
            else:
                stt_language = args.stt_language
        else:
            stt_language = args.stt_language
    else:
        stt_language = args.stt_language
    
    # Warn if languages don't match
    if tts_voice not in voice_map:
        voice_lang = tts_voice.split('-')[0] if '-' in tts_voice else None
        if voice_lang and voice_lang != stt_language:
            print(f"Warning: TTS voice language '{voice_lang}' doesn't match STT language '{stt_language}'")
            print(f"         Consider using --stt-language {voice_lang} for better recognition")
    
    # Set device (auto-detection will happen in ModelConfig.__post_init__)
    device = args.device if args.device else "auto"
    
    model_config = ModelConfig(
        stt_model=args.stt_model,
        llm_model=args.model,
        device=device,
        # STT parameters
        stt_language=stt_language,
        stt_beam_size=args.stt_beam_size,
        stt_temperature=args.stt_temperature,
        stt_vad_threshold=args.stt_vad_threshold,
        stt_vad_min_speech_duration_ms=args.stt_vad_min_speech_duration,
        stt_vad_min_silence_duration_ms=args.stt_vad_min_silence_duration,
        # TTS parameters
        tts_voice=tts_voice,
        # LLM parameters
        llm_max_tokens=args.llm_max_tokens,
        llm_temperature=args.llm_temperature,
        llm_top_p=args.llm_top_p,
        llm_context_size=args.llm_context_size
    )
    
    audio_config = AudioConfig()
    
    # Run the voice assistant
    try:
        # Import and create voice assistant with configurations
        from .voice_assistant import VoiceAssistant
        
        assistant = VoiceAssistant(model_config, audio_config)
        
        # Run conversation loop
        assistant.run_conversation_loop()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo download the model, run:")
        print("  agentvox --download-model")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()