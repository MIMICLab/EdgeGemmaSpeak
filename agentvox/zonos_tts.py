"""Zonos TTS module for AgentVox"""

import os
import sys
import asyncio
import tempfile
import subprocess
import platform
import pygame
import pyaudio
import threading
import queue
from pathlib import Path
from typing import Optional
import warnings
import io
import logging

# Configure logging to suppress INFO and DEBUG messages
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger('zonos').setLevel(logging.WARNING)
logging.getLogger('matcha').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

# Add Zonos path
ZONOS_PATH = Path(__file__).parent.parent / "third_party" / "Zonos"
if ZONOS_PATH.exists():
    sys.path.append(str(ZONOS_PATH))

try:
    import torch
    import torchaudio
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    ZONOS_AVAILABLE = True
    from zonos.utils import DEFAULT_DEVICE as device
except ImportError:
    ZONOS_AVAILABLE = False
    print("Warning: Zonos not installed. Run 'agentvox --install-zonos' to install.")

# Import EdgeTTS for fallback speaker generation
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class ZonosTTS:
    """Zonos TTS module with speaker embedding support"""
    
    def __init__(self, config):
        self.config = config
        
        if not ZONOS_AVAILABLE:
            raise ImportError("Zonos is not installed. Run 'agentvox --install-zonos' to install.")
        
        # Get device from config
        self.device = device
        
        # Initialize Zonos model
        print("Loading Zonos model...")

        self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=self.device)
        print(f"✓ Loaded Zonos transformer model on {self.device}")

        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Speaker embedding cache
        self.speaker_embedding = None
        self.speaker_audio_path = None
        
        # Load or create speaker embedding
        self._initialize_speaker_embedding()
    
    def _initialize_speaker_embedding(self):
        """Initialize speaker embedding from provided audio or generate default"""
        if hasattr(self.config, 'tts_speaker_audio') and self.config.tts_speaker_audio:
            # Use provided speaker audio
            speaker_path = Path(self.config.tts_speaker_audio)
            if speaker_path.exists():
                print(f"Loading speaker audio from {speaker_path}")
                self._load_speaker_embedding(str(speaker_path))
            else:
                print(f"Warning: Speaker audio file not found: {speaker_path}")
                print("Generating default speaker audio...")
                self._generate_default_speaker()
        else:
            # Generate default speaker audio using EdgeTTS
            print("No speaker audio provided. Generating default speaker...")
            self._generate_default_speaker()
    
    def _load_speaker_embedding(self, audio_path: str):
        """Load speaker embedding from audio file"""
        try:
            wav, sampling_rate = torchaudio.load(audio_path)
            # Move wav tensor to the correct device
            if ZONOS_AVAILABLE and hasattr(wav, 'to'):
                wav = wav.to(self.device)
            self.speaker_embedding = self.model.make_speaker_embedding(wav, sampling_rate)
            self.speaker_audio_path = audio_path
            print(f"✓ Speaker embedding loaded from {audio_path} on device: {self.device}")
        except Exception as e:
            print(f"Error loading speaker audio: {e}")
            print("Falling back to default speaker generation...")
            self._generate_default_speaker()
    
    def _generate_default_speaker(self):
        """Generate default speaker audio using EdgeTTS"""
        if not EDGE_TTS_AVAILABLE:
            print("Error: EdgeTTS not available for default speaker generation")
            print("Please provide a speaker audio file with --speaker-audio option")
            return
        
        # Create temporary directory for speaker audio
        speaker_dir = Path.home() / ".agentvox" / "speakers"
        speaker_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate speaker audio based on configured voice
        voice = self.config.tts_voice
        speaker_file = speaker_dir / f"{voice}_speaker.mp3"
        
        if not speaker_file.exists():
            print(f"Generating speaker audio with voice: {voice}")
            
            # Sample text for speaker generation
            sample_text = "안녕하세요. 저는 음성 합성 시스템입니다. 오늘도 좋은 하루 되세요."
            if not voice.startswith("ko-"):
                sample_text = "Hello. I am a text-to-speech system. Have a great day."
            
            # Generate speaker audio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                communicate = edge_tts.Communicate(sample_text, voice)
                loop.run_until_complete(communicate.save(str(speaker_file)))
                print(f"✓ Generated speaker audio: {speaker_file}")
            except Exception as e:
                print(f"Error generating speaker audio: {e}")
                return
            finally:
                loop.close()
        
        # Load the generated speaker audio
        self._load_speaker_embedding(str(speaker_file))
    
    def set_speaker_audio(self, audio_path: str):
        """Update speaker embedding with new audio file"""
        if Path(audio_path).exists():
            self._load_speaker_embedding(audio_path)
        else:
            print(f"Warning: Speaker audio file not found: {audio_path}")
    
    async def _synthesize_async(self, text: str, output_path: str) -> str:
        """Asynchronously convert text to speech file"""
        try:
            if self.speaker_embedding is None:
                raise Exception("No speaker embedding available")
            
            # Determine language based on text or config
            language = self._detect_language(text)
            
            # Ensure speaker embedding is on the correct device
            speaker_embedding = self.speaker_embedding
            if hasattr(speaker_embedding, 'to'):
                speaker_embedding = speaker_embedding.to(self.device)
            
            # Create conditioning
            cond_dict = make_cond_dict(
                text=text,
                speaker=speaker_embedding,
                language=language
            )
            conditioning = self.model.prepare_conditioning(cond_dict)
            
            # Generate audio codes
            codes = self.model.generate(conditioning)
            
            # Decode to audio
            wavs = self.model.autoencoder.decode(codes).cpu()
            
            # Save audio file
            torchaudio.save(output_path, wavs[0], self.model.autoencoder.sampling_rate)
            
            return output_path
            
        except Exception as e:
            print(f"Zonos synthesis error: {e}")
            raise
    
    def _detect_language(self, text: str) -> str:
        """Detect language from text or use configured language"""
        # Simple language detection based on Unicode ranges
        import unicodedata
        
        # Count characters by script
        korean_count = sum(1 for c in text if '\uAC00' <= c <= '\uD7AF')
        chinese_count = sum(1 for c in text if '\u4E00' <= c <= '\u9FFF')
        japanese_count = sum(1 for c in text if ('\u3040' <= c <= '\u309F') or ('\u30A0' <= c <= '\u30FF'))
        
        # Determine language
        if korean_count > len(text) * 0.3:
            return "ko"
        elif chinese_count > len(text) * 0.3:
            return "zh"
        elif japanese_count > len(text) * 0.3:
            return "ja"
        elif any(c in 'äöüßÄÖÜ' for c in text):
            return "de"
        elif any(c in 'àâçèéêëîïôùûüÿÀÂÇÈÉÊËÎÏÔÙÛÜŸ' for c in text):
            return "fr"
        else:
            return "en-us"  # Default to English
    
    def synthesize(self, text: str, output_path: str = "output.wav", speaker_wav: str = None) -> str:
        """Convert text to speech file"""
        if not text or not text.strip():
            text = "텍스트가 제공되지 않았습니다"  # "No text provided" in Korean
        
        # Override speaker if speaker_wav is provided
        if speaker_wav and Path(speaker_wav).exists():
            old_embedding = self.speaker_embedding
            self._load_speaker_embedding(speaker_wav)
        
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._synthesize_async(text, output_path))
                return result
            finally:
                loop.close()
        finally:
            # Restore original speaker if it was overridden
            if speaker_wav and 'old_embedding' in locals():
                self.speaker_embedding = old_embedding
    
    async def _stream_and_play_async(self, text: str) -> None:
        """Asynchronous streaming playback with immediate audio output"""
        try:
            if self.speaker_embedding is None:
                raise Exception("No speaker embedding available")
            
            # Determine language
            language = self._detect_language(text)
            
            # Ensure speaker embedding is on correct device
            speaker_embedding = self.speaker_embedding
            if hasattr(speaker_embedding, 'to'):
                speaker_embedding = speaker_embedding.to(self.device)
            
            # Create conditioning
            cond_dict = make_cond_dict(
                text=text,
                speaker=speaker_embedding,
                language=language
            )
            conditioning = self.model.prepare_conditioning(cond_dict)
            
            # Generate audio codes
            codes = self.model.generate(conditioning)
            
            # Decode to audio
            wavs = self.model.autoencoder.decode(codes).cpu()
            audio_data = wavs[0]
            
            # Convert to numpy array for playback
            audio_np = audio_data.numpy()
            if len(audio_np.shape) > 1:
                audio_np = audio_np[0]  # Take first channel if stereo
            
            # Initialize PyAudio for streaming
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.model.autoencoder.sampling_rate,
                output=True,
                frames_per_buffer=1024
            )
            
            # Stream audio in chunks for immediate playback
            chunk_size = 1024
            for i in range(0, len(audio_np), chunk_size):
                chunk = audio_np[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    # Pad last chunk if needed
                    chunk = torch.nn.functional.pad(
                        torch.tensor(chunk), 
                        (0, chunk_size - len(chunk))
                    ).numpy()
                stream.write(chunk.tobytes())
                await asyncio.sleep(0)  # Allow other async operations
            
            # Clean up
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            print(f"Zonos streaming error: {e}")
            # Fallback to file-based playback
            await self._fallback_playback(text)
    
    def speak_streaming(self, text: str):
        """Stream text to speech with immediate playback"""
        if not text or not text.strip():
            print("Warning: Empty text - skipping TTS")
            return
        
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._stream_and_play_async(text))
            finally:
                loop.close()
                
        except Exception as e:
            print(f"Zonos streaming error: {e}")
    
    async def _fallback_playback(self, text: str):
        """Fallback to file-based playback if streaming fails"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            await self._synthesize_async(text, tmp_path)
            
            if platform.system() == "Darwin":
                subprocess.call(["afplay", tmp_path])
            else:
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def speak(self, text: str):
        """Convert text to speech and play with streaming"""
        if not text or not text.strip():
            print("Warning: Empty text - skipping TTS")
            return
        
        # Use streaming playback
        self.speak_streaming(text)
    
    def set_voice(self, voice_type: str = "default"):
        """Set voice type by regenerating speaker embedding"""
        # Update config voice
        voice_map = {
            "male": "ko-KR-InJoonNeural",
            "female": "ko-KR-SunHiNeural",
            "multilingual": "ko-KR-HyunsuMultilingualNeural"
        }
        
        new_voice = voice_map.get(voice_type, voice_type)
        if new_voice != self.config.tts_voice:
            self.config.tts_voice = new_voice
            # Regenerate speaker embedding with new voice
            self._generate_default_speaker()