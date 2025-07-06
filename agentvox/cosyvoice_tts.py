"""CosyVoice TTS module for AgentVox"""

import os
import sys
import asyncio
import tempfile
import subprocess
import platform
import pygame
import re
from pathlib import Path
from typing import Optional, Generator
import warnings

# Add CosyVoice path
COSYVOICE_PATH = Path(__file__).parent.parent / "third_party" / "CosyVoice"
if COSYVOICE_PATH.exists():
    sys.path.append(str(COSYVOICE_PATH))
    sys.path.append(str(COSYVOICE_PATH / "third_party" / "Matcha-TTS"))

try:
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    COSYVOICE_AVAILABLE = True
except ImportError:
    COSYVOICE_AVAILABLE = False
    print("Warning: CosyVoice not installed. Run 'agentvox --install-cosyvoice' to install.")


class CosyVoiceTTS:
    """CosyVoice TTS module for streaming synthesis"""
    
    def __init__(self, config):
        self.config = config
        
        if not COSYVOICE_AVAILABLE:
            raise ImportError("CosyVoice is not installed. Run 'agentvox --install-cosyvoice' to install.")
        
        # Initialize CosyVoice model
        model_path = COSYVOICE_PATH / "pretrained_models" / "CosyVoice2-0.5B"
        if not model_path.exists():
            raise FileNotFoundError(f"CosyVoice model not found at {model_path}. Run installation script.")
        
        print("Loading CosyVoice model...")
        self.cosyvoice = CosyVoice2(
            str(model_path),
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False
        )
        print("✓ CosyVoice model loaded")
        
        # Load prompt audio
        self.prompt_speech = None
        self.prompt_text = "希望你以后能够做的比我还好呦。"  # Default prompt text
        self._load_prompt_audio()
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
    def _load_prompt_audio(self):
        """Load prompt audio from file or default"""
        if hasattr(self.config, 'tts_cosyvoice_prompt') and self.config.tts_cosyvoice_prompt:
            prompt_path = Path(self.config.tts_cosyvoice_prompt)
            if prompt_path.exists():
                print(f"Loading prompt audio from {prompt_path}")
                self.prompt_speech = load_wav(str(prompt_path), 16000)
                return
            else:
                print(f"Warning: Prompt audio file not found: {prompt_path}")
        
        # Try default prompt
        default_prompt = COSYVOICE_PATH / "asset" / "zero_shot_prompt.wav"
        if default_prompt.exists():
            print(f"Loading default prompt audio from {default_prompt}")
            self.prompt_speech = load_wav(str(default_prompt), 16000)
        else:
            print("Warning: No prompt audio found. Using instruct mode instead.")
    
    def _text_generator(self, text: str) -> Generator[str, None, None]:
        """Generator for streaming text input"""
        # Split text into sentences for better streaming
        sentences = re.split(r'[.!?。！？]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Add appropriate punctuation based on language
                if any('\u4e00' <= c <= '\u9fff' for c in sentence):  # Chinese
                    yield sentence + '。'
                elif any('\uac00' <= c <= '\ud7af' for c in sentence):  # Korean
                    yield sentence + '.'
                else:
                    yield sentence + '.'
    
    async def _synthesize_async(self, text: str, output_path: str) -> str:
        """Asynchronously convert text to speech file"""
        try:
            results = []
            
            # Use zero-shot synthesis if prompt is available
            if self.prompt_speech is not None:
                # Use text generator for better streaming support
                for result in self.cosyvoice.inference_zero_shot(
                    self._text_generator(text),
                    self.prompt_text,
                    self.prompt_speech,
                    stream=False
                ):
                    results.append(result)
            else:
                # Use instruct mode as fallback
                # Detect language and use appropriate instruction
                if any('\uac00' <= c <= '\ud7af' for c in text):  # Korean
                    instruction = '용温柔的语气说韩语'
                elif any('\u4e00' <= c <= '\u9fff' for c in text):  # Chinese
                    instruction = '用温柔的语气说这句话'
                else:  # English or other
                    instruction = 'Speak in a gentle tone'
                
                for result in self.cosyvoice.inference_instruct2(
                    text,
                    instruction,
                    None,
                    stream=False
                ):
                    results.append(result)
            
            # Combine all audio chunks
            if results:
                combined_audio = torch.cat([r['tts_speech'] for r in results], dim=1)
                torchaudio.save(output_path, combined_audio, self.cosyvoice.sample_rate)
                return output_path
            else:
                raise Exception("No audio generated")
                
        except Exception as e:
            print(f"CosyVoice synthesis error: {e}")
            raise
    
    def synthesize(self, text: str, output_path: str = "output.wav", speaker_wav: str = None) -> str:
        """Convert text to speech file"""
        if not text or not text.strip():
            text = "텍스트가 제공되지 않았습니다"  # Default Korean text
        
        # Override prompt if speaker_wav is provided
        old_prompt = None
        if speaker_wav and Path(speaker_wav).exists():
            old_prompt = self.prompt_speech
            self.prompt_speech = load_wav(speaker_wav, 16000)
            print(f"Using custom prompt audio: {speaker_wav}")
        
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
            # Restore original prompt if it was overridden
            if old_prompt is not None:
                self.prompt_speech = old_prompt
    
    async def _stream_and_play_async(self, text: str) -> None:
        """Asynchronous streaming playback"""
        # CosyVoice supports streaming, but for simplicity we'll generate full audio first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            await self._synthesize_async(text, tmp_path)
            
            # Platform-specific playback
            if platform.system() == "Darwin":
                # macOS: use afplay
                subprocess.call(["afplay", tmp_path])
            else:
                # Other OS: use pygame
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
                    
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def speak_streaming(self, text: str):
        """Stream text to speech with playback"""
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
            print(f"CosyVoice streaming error: {e}")
    
    def speak(self, text: str):
        """Convert text to speech and play"""
        if not text or not text.strip():
            print("Warning: Empty text - skipping TTS")
            return
        
        try:
            output_path = "temp_speech.wav"
            self.synthesize(text, output_path)
            
            # Platform-specific playback
            if platform.system() == "Darwin":
                subprocess.call(["afplay", output_path])
            else:
                pygame.mixer.music.load(output_path)
                pygame.mixer.music.play()
                
                # Wait for playback
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)
                
        except Exception as e:
            print(f"CosyVoice playback error: {e}")
    
    def set_voice(self, voice_type: str = "default"):
        """Set voice type (for compatibility)"""
        # CosyVoice uses prompt audio instead of voice presets
        print(f"CosyVoice uses prompt audio for voice cloning. Voice type '{voice_type}' ignored.")
        print("Use --cosyvoice-prompt option to specify a custom voice prompt audio file.")