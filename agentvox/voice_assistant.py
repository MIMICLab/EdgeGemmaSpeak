import os
import torch
import numpy as np
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import time
import threading
import logging
import base64
import io
import math
import ctypes
from PIL import Image
import tempfile
import wave
from collections import deque
from datetime import datetime, timedelta
import cv2
import json

# Google Gemini imports
import google.generativeai as genai
# New Gemini SDK for TTS
from google import genai as google_genai
from google.genai import types
GEMINI_TTS_AVAILABLE = True

# PyTorch 2.6 security settings
import warnings
warnings.filterwarnings("ignore", message="torch.load warnings")

# Ignore numpy RuntimeWarning (divide by zero, overflow, invalid value)
np.seterr(divide='ignore', invalid='ignore', over='ignore')

# Libraries for speech recognition
from RealtimeSTT import AudioToTextRecorder

# Libraries for audio
import pygame
import soundfile as sf
import subprocess
import platform

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it with: export GEMINI_API_KEY='your_api_key'")
genai.configure(api_key=api_key)

@dataclass
class AudioConfig:
    """Class for managing audio configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 2048
    audio_format: str = "wav"
    
@dataclass
class ModelConfig:
    """Class for managing model configuration"""
    stt_model: str = "base"  # Whisper model size
    device: str = "auto"  # Device: auto, cpu, cuda, mps
    
    # STT detailed settings
    stt_language: str = "ko"
    stt_beam_size: int = 5
    stt_temperature: float = 0.0
    stt_vad_threshold: float = 0.5
    stt_vad_min_speech_duration_ms: int = 250
    stt_vad_min_silence_duration_ms: int = 1000  # Reduced from 2000ms for faster response
    
    # Gemini settings
    gemini_model: str = "models/gemini-2.5-flash-lite"
    gemini_temperature: float = 0.7
    gemini_top_p: float = 0.95
    gemini_max_tokens: int = 8192
    
    # TTS settings
    tts_model: str = "gemini-2.5-flash-preview-tts"
    tts_voice: str = "Kore"  # Korean voice
    tts_speed: float = 1.0
    speaker_wav: Optional[str] = None  # Voice cloning source file for Coqui TTS
    
    # Video buffer settings
    video_buffer_seconds: int = 30  # Keep last 30 seconds of video
    video_fps: int = 10  # Frames per second to capture
    
    def __post_init__(self):
        """Auto-detect device after initialization"""
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"Auto-detected device: CUDA (GPU: {torch.cuda.get_device_name(0)})")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                print("Auto-detected device: Apple Silicon (MPS)")
            else:
                self.device = "cpu"
                print("Auto-detected device: CPU")

class VideoBuffer:
    """Buffer to store last N seconds of video frames with timestamps"""
    
    def __init__(self, buffer_seconds: int = 30, fps: int = 10):
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.max_frames = buffer_seconds * fps
        self.frames = deque(maxlen=self.max_frames)
        self.timestamps = deque(maxlen=self.max_frames)
        
    def add_frame(self, frame: Image.Image, timestamp: Optional[datetime] = None):
        """Add a frame to the buffer with timestamp"""
        if timestamp is None:
            timestamp = datetime.now()
        self.frames.append(frame)
        self.timestamps.append(timestamp)
    
    def get_video_frames(self, duration_seconds: int = 30) -> List[Image.Image]:
        """Get frames from the last N seconds"""
        if not self.frames:
            return []
        
        cutoff_time = datetime.now() - timedelta(seconds=duration_seconds)
        result_frames = []
        
        for frame, timestamp in zip(self.frames, self.timestamps):
            if timestamp >= cutoff_time:
                result_frames.append(frame)
        
        return result_frames
    
    def create_video_file(self, duration_seconds: int = 30, output_fps: int = 10) -> Optional[str]:
        """Create a video file from buffered frames"""
        frames = self.get_video_frames(duration_seconds)
        if not frames:
            return None
        
        try:
            # Create temporary video file
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Get frame dimensions from first frame
            first_frame = np.array(frames[0])
            height, width = first_frame.shape[:2]
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, output_fps, (width, height))
            
            # Write frames to video
            for frame in frames:
                # Convert PIL Image to numpy array
                frame_array = np.array(frame)
                # Convert RGB to BGR for OpenCV
                if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_array
                out.write(frame_bgr)
            
            out.release()
            return temp_path
            
        except Exception as e:
            print(f"Error creating video file: {e}")
            return None
    
    def clear(self):
        """Clear the buffer"""
        self.frames.clear()
        self.timestamps.clear()

class STTModule:
    """Module for converting speech to text using RealtimeSTT"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_listening = False
        self.speech_start_time = None
        self.speech_end_time = None
        
        # Initialize RealtimeSTT recorder
        self.recorder = AudioToTextRecorder(
            model=config.stt_model,
            language=config.stt_language,
            device=config.device,
            spinner=False,
            use_microphone=True,
            level=logging.WARNING
        )
        
    def transcribe_once(self) -> Optional[str]:
        """Listen and transcribe once, tracking speech timing"""
        is_korean = self.config.stt_language.startswith('ko')
        
        if is_korean:
            print("\n말씀해주세요...")
        else:
            print("\nPlease speak...")
        
        # Mark when we start listening
        self.is_listening = True
        self.speech_start_time = datetime.now()
        
        # Get text
        text = self.recorder.text()
        
        # Mark when speech ends
        self.speech_end_time = datetime.now()
        self.is_listening = False
        
        if text:
            print(f"\n사용자: {text}" if is_korean else f"\nUser: {text}")
            return text
        return None

class GeminiLLMModule:
    """Gemini-based LLM module for multimodal response generation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = genai.GenerativeModel(config.gemini_model)
        
    def generate_response(self, 
                         text: str, 
                         video_path: Optional[str] = None,
                         final_image: Optional[Image.Image] = None,
                         audio_data: Optional[bytes] = None) -> str:
        """
        Generate response using Gemini with multimodal input
        
        Args:
            text: User's transcribed speech
            video_path: Path to video file from last 30 seconds
            final_image: Final frame with gaze indicator
            audio_data: Raw audio data (for future use)
        """
        is_korean = self.config.stt_language.startswith('ko')
        
        # Build the prompt
        system_prompt = self._build_system_prompt()
        
        # Prepare contents for Gemini
        contents = []
        
        # Add system prompt as text
        contents.append(system_prompt)
        
        # Upload and add video if available
        if video_path and os.path.exists(video_path):
            try:
                # Upload video file to Gemini
                video_file = genai.upload_file(path=video_path)
                contents.append("\n이전 30초간의 비디오:\n" if is_korean else "\nVideo from the last 30 seconds:\n")
                contents.append(video_file)
            except Exception as e:
                print(f"Failed to upload video: {e}")
        
        # Add final image with gaze indicator
        if final_image:
            contents.append("\n사용자가 보고 있는 화면:\n" if is_korean 
                          else "\nUser's view:\n")
            contents.append(final_image)
        
        # Add user's request
        contents.append(f"\n사용자 요청: {text}" if is_korean else f"\nUser request: {text}")
        
        try:
            # Generate response
            response = self.model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(
                    temperature=self.config.gemini_temperature,
                    top_p=self.config.gemini_top_p,
                    max_output_tokens=self.config.gemini_max_tokens,
                )
            )
            
            response_text = response.text if hasattr(response, 'text') else str(response)
            return self._clean_response(response_text)
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            if is_korean:
                return "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."
            else:
                return "I'm sorry. An error occurred while generating the response."
    
    def _build_system_prompt(self) -> str:
        """Build system prompt"""
        is_korean = self.config.stt_language.startswith('ko')
        
        if is_korean:
            return """
당신은 서강대학교 미믹랩에서 개발한 AI 어시스턴트입니다.

응답 규칙:
1. 별표(*), 하이픈(-), 콜론(:) 등의 특수문자를 답변에 사용하지 않습니다.
2. 리스트나 강조 없이 평서문만 사용합니다.
3. 추측이나 배경 설명 없이, 눈에 보이는 사실만 묘사합니다.
4. 불필요한 설명이나 추가 질문을 하지 않습니다.
5. 영어 단어는 한글 음차 표기로 적습니다. 예) AI → 에이아이.

시각 정보 처리:
• 제공된 비디오는 사용자가 질문하기 전 30초간의 화면입니다.
• 마지막 이미지의 초록색 원은 사용자가 말을 마친 시점의 시선 위치를 나타내는 시스템 표시입니다.
• 초록색 원 자체를 언급하지 마세요. 이것은 시선 추적 시스템의 표시일 뿐입니다.
• 사용자가 보고 있는 대상(초록색 원이 가리키는 곳)에 대해서만 답변하세요.

간결하고 자연스럽게 답변하세요.
"""
        else:
            return """
You are an AI assistant developed by MimicLab at Sogang University.

Response rules:
1. Do not use symbols such as asterisks, hyphens, or colons in your replies.
2. Write plain sentences only; no lists or formatting.
3. State only what is visually apparent without speculation.
4. Do not add unnecessary explanations or follow-up questions.

Visual information processing:
• The provided video shows the last 30 seconds before the user's question.
• The green circle in the final image is a system indicator showing where the user was looking.
• DO NOT mention the green circle itself. It is just a gaze tracking system indicator.
• Only describe what the user is looking at (the location pointed by the green circle).

Respond concisely and naturally.
"""
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        if not response:
            return response
        
        # Remove common prefixes
        response = response.strip()
        if response.startswith("Assistant:"):
            response = response[10:].strip()
        elif response.startswith("어시스턴트:"):
            response = response[6:].strip()
        
        # Remove special characters
        response = response.replace("*", "").replace("--", "").strip()
        
        # Check for empty or invalid response
        if not response or not re.search(r'[\uac00-\ud7a3a-zA-Z0-9]', response):
            is_korean = self.config.stt_language.startswith('ko')
            if is_korean:
                response = "죄송합니다. 다시 한 번 말씀해 주시겠어요?"
            else:
                response = "I'm sorry. Could you please say that again?"
        
        return response

class GeminiTTSModule:
    """TTS module using Gemini's text-to-speech API"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.use_gemini_tts = GEMINI_TTS_AVAILABLE
        
        if self.use_gemini_tts:
            try:
                self.client = google_genai.Client()
            except Exception as e:
                print(f"Failed to initialize Gemini TTS client: {e}")
                self.use_gemini_tts = False
        
        # Fallback to Coqui TTS if Gemini TTS is not available
        if not self.use_gemini_tts:
            print("Using fallback TTS engine (Coqui)")
            from RealtimeTTS import TextToAudioStream, CoquiEngine
            self.fallback_engine = CoquiEngine(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                device=config.device,
                language=config.stt_language,
                speed=config.tts_speed
            )
            self.fallback_stream = TextToAudioStream(self.fallback_engine)
        
    def speak(self, text: str) -> Optional[str]:
        """
        Convert text to speech using Gemini TTS and play it
        Returns the path to the saved audio file
        """
        if not text or not text.strip():
            return None
        
        if self.use_gemini_tts:
            try:
                # Generate speech using Gemini TTS
                response = self.client.models.generate_content(
                    model=self.config.tts_model,
                    contents=text,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=self.config.tts_voice,
                                )
                            )
                        ),
                    )
                )
                
                # Extract audio data
                audio_data = response.candidates[0].content.parts[0].inline_data.data
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                self._save_wave_file(temp_file.name, audio_data)
                
                # Play the audio
                self._play_audio(temp_file.name)
                
                # Clean up
                os.unlink(temp_file.name)
                
                return temp_file.name
                
            except Exception as e:
                print(f"Gemini TTS error: {e}, falling back to Coqui")
                if hasattr(self, 'fallback_stream'):
                    self.fallback_stream.feed(text)
                    self.fallback_stream.play()
        else:
            # Use fallback TTS
            try:
                self.fallback_stream.feed(text)
                self.fallback_stream.play()
            except Exception as e:
                print(f"Fallback TTS error: {e}")
                
        return None
    
    def _save_wave_file(self, filename: str, pcm_data: bytes, 
                       channels: int = 1, rate: int = 24000, sample_width: int = 2):
        """Save PCM data as WAV file"""
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)
    
    def _play_audio(self, audio_file: str):
        """Play audio file using system audio player"""
        system = platform.system()
        
        try:
            if system == "Darwin":  # macOS
                subprocess.run(["afplay", audio_file], check=True)
            elif system == "Linux":
                # Try different audio players
                for player in ["aplay", "paplay", "play"]:
                    try:
                        subprocess.run([player, audio_file], check=True)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
            elif system == "Windows":
                # Use pygame for Windows
                pygame.mixer.init()
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"Error playing audio: {e}")

class VoiceAssistant:
    """Voice conversation system: Record -> 30s Video + Gaze Image -> Gemini -> TTS"""
    
    def __init__(self, model_config: ModelConfig, audio_config: AudioConfig):
        self.model_config = model_config
        self.audio_config = audio_config
        
        # Video buffer for last 30 seconds
        self.video_buffer = VideoBuffer(
            buffer_seconds=model_config.video_buffer_seconds,
            fps=model_config.video_fps
        )
        
        # Image at speech end time with gaze indicator
        self.current_gaze_image = None
        
        is_korean = model_config.stt_language.startswith('ko')
        
        if is_korean:
            print("모델을 초기화하는 중입니다...")
            print("Gemini 2.5 Flash 모델을 사용합니다.")
        else:
            print("Initializing models...")
            print("Using Gemini 2.5 Flash model.")
            
        self.stt = STTModule(model_config)
        self.llm = GeminiLLMModule(model_config)
        self.tts = GeminiTTSModule(model_config)
        
    
    def add_frame(self, frame: Image.Image, timestamp: Optional[datetime] = None):
        """Add a video frame to the buffer"""
        self.video_buffer.add_frame(frame, timestamp)
    
    def set_gaze_image(self, image: Image.Image):
        """Set the current image with gaze indicator"""
        self.current_gaze_image = image
    
    # Aria compatibility methods
    def add_image(self, image: Image.Image):
        """Aria compatibility: Set gaze image"""
        self.current_gaze_image = image
    
    def clear_images(self):
        """Aria compatibility: Clear gaze image"""
        self.current_gaze_image = None
    
    def set_external_audio_source(self, audio_source):
        """Aria compatibility: External audio not used in simplified version"""
        pass
    
    
    
    def run_conversation_loop(self):
        """Core loop: Record audio -> 30s video + gaze image -> Gemini -> TTS"""
        is_korean = self.model_config.stt_language.startswith('ko')
        
        print("=" * 50)
        print("음성 대화 시스템 시작" if is_korean else "Voice System Started")
        print("종료: '종료' 또는 'exit'" if is_korean else "Exit: say 'exit'")
        print("=" * 50)
        
        while True:
            # 1. Record and transcribe user speech
            user_input = self.stt.transcribe_once()
            
            if not user_input:
                continue
                
            # Check exit command
            if "exit" in user_input.lower() or "종료" in user_input:
                print("\n👋 " + ("종료합니다." if is_korean else "Goodbye."))
                break
            
            # 2. Create 30-second video from buffer
            video_path = None
            if self.video_buffer.frames:
                video_path = self.video_buffer.create_video_file(duration_seconds=30)
            
            # 3. Get gaze image at speech end time
            final_gaze_image = self.current_gaze_image
            
            # 4. Send to Gemini with prompt
            response = self.llm.generate_response(
                text=user_input,
                video_path=video_path,
                final_image=final_gaze_image
            )
            
            # 5. Show text output
            print(f"\n💬 {response}")
            
            # 6. Convert to speech and play
            self.tts.speak(response)
            
            # Clean up
            if video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except:
                    pass
            self.current_gaze_image = None

# Main execution function
def main():
    """Main execution function"""
    # Initialize configuration
    audio_config = AudioConfig()
    model_config = ModelConfig()
    
    # Initialize voice assistant
    assistant = VoiceAssistant(model_config, audio_config)
    
    # Run conversation loop
    assistant.run_conversation_loop()

if __name__ == "__main__":
    # Required for multiprocessing on macOS/Windows
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()