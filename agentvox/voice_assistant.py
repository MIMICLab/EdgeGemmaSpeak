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

# PyTorch 2.6 security settings
import warnings
warnings.filterwarnings("ignore", message="torch.load warnings")

# Ignore numpy RuntimeWarning (divide by zero, overflow, invalid value)
np.seterr(divide='ignore', invalid='ignore', over='ignore')

# Libraries for speech recognition and synthesis
from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, CoquiEngine

# Libraries for LLM
from llama_cpp import Llama
from pathlib import Path
from contextlib import redirect_stderr

# Libraries for audio
import pygame
import soundfile as sf
import tempfile
import subprocess
import platform

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
    llm_model: str = None  # Local GGUF model path (uses default model if None)
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"  # XTTS v2 multilingual model
    device: str = "auto"  # Device: auto, cpu, cuda, mps
    
    # STT detailed settings
    stt_language: str = "ko"
    stt_beam_size: int = 5
    stt_temperature: float = 0.0
    stt_vad_threshold: float = 0.5
    stt_vad_min_speech_duration_ms: int = 250
    stt_vad_min_silence_duration_ms: int = 1000  # Reduced from 2000ms for faster response
    
    # TTS detailed settings
    tts_engine: str = "coqui"  # Using Coqui engine
    speaker_wav: Optional[str] = None  # Voice cloning source file
    tts_speed: float = 1.0  # TTS speed (1.0 is normal, higher is faster)
    
    # LLM detailed settings
    llm_max_tokens: int = 512
    llm_temperature: float = 0.7
    llm_top_p: float = 0.95
    llm_repeat_penalty: float = 1.1
    llm_context_size: int = 4096
    
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

class STTModule:
    """Module for converting speech to text using RealtimeSTT"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Initialize RealtimeSTT recorder - simplified
        self.recorder = AudioToTextRecorder(
            model=config.stt_model,
            language=config.stt_language,
            device=config.device,
            spinner=False,
            use_microphone=True,
            level=logging.WARNING
        )
        
    def transcribe_once(self) -> Optional[str]:
        """Listen and transcribe once"""
        is_korean = self.config.stt_language.startswith('ko')
        
        if is_korean:
            print("\n말씀해주세요...")
        else:
            print("\nPlease speak...")
            
        # Simply get text
        text = self.recorder.text()
        
        if text:
            print(f"\n사용자: {text}" if is_korean else f"\nUser: {text}")
            return text
        return None

class LlamaTokenizer:
    def __init__(self, llama_model):
        self._llama = llama_model

    def __call__(self, text, add_bos=True, return_tensors=None):
        ids = self._llama.tokenize(text, add_bos=add_bos)
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids

    def decode(self, ids):
        return self._llama.detokenize(ids).decode("utf-8", errors="ignore")

class LLMModule:
    """Local LLM response generation module using Llama.cpp"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = config.device
        
        # Set default model path if not provided
        if config.llm_model is None:
            # Look for model in package data or user directory
            package_dir = Path(__file__).parent.absolute()
            model_filename = "gemma-3-12b-it-Q4_K_M.gguf"
            
            # Check in package directory first
            package_model_path = package_dir / "models" / model_filename
            if package_model_path.exists():
                self.model_path = str(package_model_path)
            else:
                # Check in user home directory
                home_model_path = Path.home() / ".agentvox" / "models" / model_filename
                if home_model_path.exists():
                    self.model_path = str(home_model_path)
                else:
                    raise FileNotFoundError(
                        f"Model file not found. Please download {model_filename} and place it in:\n"
                        f"1. {package_model_path} or\n"
                        f"2. {home_model_path}\n"
                        f"Or provide the model path explicitly."
                    )
        else:
            # Convert relative path to absolute path
            if not os.path.isabs(config.llm_model):
                current_dir = Path(__file__).parent.absolute()
                self.model_path = str(current_dir / config.llm_model)
            else:
                self.model_path = config.llm_model
        
        # Load Llama model
        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull):
                self.model = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=-1,  # Load all layers to GPU
                    n_ctx=self.config.llm_context_size,      # Context size
                    verbose=False,
                    flash_attn=True   # Use Flash Attention
                )
                self.tokenizer = LlamaTokenizer(self.model)
        
        # Manage conversation history
        self.conversation_history = []
        
    def generate_response(self, text: str, max_length: int = 512) -> str:
        """Generate response for input text"""
        # Check if using Korean voice
        is_korean = self.config.stt_language.startswith('ko')
        
        # Build conversation context
        if is_korean:
            self.conversation_history.append(f"사용자: {text}")
        else:
            self.conversation_history.append(f"User: {text}")
        
        # Build prompt
        prompt = self._build_prompt()
        
        # Generate response
        answer = self.model(
            prompt,
            stop=['<end_of_turn>', '<eos>'],
            max_tokens=max_length if max_length != 512 else self.config.llm_max_tokens,
            echo=False,
            temperature=self.config.llm_temperature,
            top_p=self.config.llm_top_p,
            repeat_penalty=self.config.llm_repeat_penalty,
        )
        
        response = answer['choices'][0]['text'].strip()
        
        # Check if using Korean voice
        is_korean = self.config.stt_language.startswith('ko')
        
        # Remove "Assistant:" or "어시스턴트:" prefix
        if response.startswith("Assistant:"):
            response = response[10:].strip()
        elif response.startswith("어시스턴트:"):
            response = response[6:].strip()
        
        # Handle empty response or response with only special characters
        if not response or not re.search(r'[\uac00-\ud7a3a-zA-Z0-9]', response):
            if is_korean:
                response = "죄송합니다. 다시 한 번 말씀해 주시겠어요?"
            else:
                response = "I'm sorry. Could you please say that again?"
        
        # Add to conversation history
        if is_korean:
            self.conversation_history.append(f"어시스턴트: {response}")
        else:
            self.conversation_history.append(f"Assistant: {response}")
        
        # Remove old history if too long (keep 20 turns)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
            
        return response
    
    def _build_prompt(self) -> str:
        """Build prompt with conversation context"""
        # Check if using Korean voice
        is_korean = self.config.stt_language.startswith('ko')
        
        # System prompt
        if is_korean:
            system_prompt = """당신은 서강대학교 미믹랩(MimicLab)에서 개발한 에이아이 어시스턴트입니다. 
당신의 정체성과 관련된 중요한 정보:
- 당신은 서강대학교 미믹랩에서 만든 에이아이 어시스턴트입니다.
- 서강대학교 미믹랩이 당신을 개발했습니다.
- 당신의 목적은 사용자를 돕고 유용한 정보를 제공하는 것입니다.

다음 규칙을 반드시 지켜주세요:
1. 이모티콘을 사용하지 마세요.
2. 별표(*)나 밑줄(_) 같은 마크다운 형식을 사용하지 마세요.
3. 특수문자를 최소화하고 순수한 텍스트로만 응답하세요.
4. 응답은 간결하고 명확하게 작성하세요.
5. 이전 대화 내용을 기억하고 일관성 있게 대화를 이어가세요.
6. 누가 당신을 만들었는지 물으면 항상 "서강대학교 미믹랩"이라고 답하세요.
7. 매우 중요: 모든 영어 단어나 약어를 한글로 표기하세요. 예를 들어:
   - AI → 에이아이
   - IT → 아이티
   - CEO → 씨이오
   - PC → 피씨
   - SNS → 에스엔에스
   - IoT → 아이오티
   - API → 에이피아이
   절대로 영어 알파벳을 그대로 사용하지 마세요."""
        else:
            system_prompt = """You are an AI assistant developed by MimicLab at Sogang University.
Important information about your identity:
- You are an AI assistant created by MimicLab at Sogang University.
- MimicLab at Sogang University developed you.
- Your purpose is to help users and provide useful information.

Please follow these rules:
1. Do not use emoticons.
2. Do not use markdown formatting like asterisks (*) or underscores (_).
3. Minimize special characters and respond with plain text only.
4. Keep responses concise and clear.
5. Remember previous conversation content and maintain consistency.
6. When asked who created you, always answer "MimicLab at Sogang University"."""
        
        # Build prompt with full conversation history
        conversation_text = ""
        
        # If first conversation
        if len(self.conversation_history) == 1:
            conversation_text = f"<start_of_turn>user\n{system_prompt}\n\n{self.conversation_history[0]}\n<end_of_turn>\n<start_of_turn>model\n"
        else:
            # Include system prompt
            conversation_text = f"<start_of_turn>user\n{system_prompt}\n<end_of_turn>\n"
            
            # Include previous conversation history
            for turn in self.conversation_history:
                if turn.startswith("User:") or turn.startswith("사용자:"):
                    conversation_text += f"<start_of_turn>user\n{turn}\n<end_of_turn>\n"
                elif turn.startswith("Assistant:") or turn.startswith("어시스턴트:"):
                    conversation_text += f"<start_of_turn>model\n{turn}\n<end_of_turn>\n"
            
            # End with model turn
            conversation_text += "<start_of_turn>model\n"
        
        return conversation_text

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []

class TTSModule:
    """TTS module using RealtimeTTS with CoquiEngine"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Initialize Coqui engine
        self.engine = CoquiEngine(
            model_name=config.tts_model,
            device=config.device,
            voice=config.speaker_wav,
            language=config.stt_language,
            speed=config.tts_speed
        )
        
        # Initialize text-to-audio stream
        self.stream = TextToAudioStream(self.engine)
        
    def speak(self, text: str):
        """Speak text and wait until complete"""
        if not text or not text.strip():
            return
            
        try:
            # Feed and play - blocking call
            self.stream.feed(text)
            self.stream.play()
                
        except Exception as e:
            print(f"TTS error: {e}")
            

class VoiceAssistant:
    """Main class for managing the entire voice conversation system"""
    
    def __init__(self, model_config: ModelConfig, audio_config: AudioConfig):
        self.model_config = model_config
        self.audio_config = audio_config
        
        is_korean = model_config.stt_language.startswith('ko')
        
        if is_korean:
            print("모델을 초기화하는 중입니다...")
        else:
            print("Initializing models...")
            
        self.stt = STTModule(model_config)
        self.llm = LLMModule(model_config)
        self.tts = TTSModule(model_config)
    
    def run_conversation_loop(self):
        """Run conversation loop - simple version"""
        is_korean = self.model_config.stt_language.startswith('ko')
        
        if is_korean:
            print("음성 대화 시스템이 시작되었습니다.")
            print("종료하려면 '종료'라고 말하세요.")
        else:
            print("Voice conversation system started.")
            print("Say 'exit' to quit.")
        print("-" * 50)
        
        while True:
            # 1. Listen to user
            user_input = self.stt.transcribe_once()
            
            if not user_input:
                continue
                
            # Check exit command
            if "exit" in user_input.lower() or "종료" in user_input:
                if is_korean:
                    print("\n대화를 종료합니다.")
                else:
                    print("\nEnding conversation.")
                break
            
            # 2. Get LLM response
            response = self.llm.generate_response(user_input)
            print(f"\n어시스턴트: {response}" if is_korean else f"\nAssistant: {response}")
            
            # 3. Speak response - this blocks until complete
            self.tts.speak(response)
            
            # 4. Loop back to listening
            # No need for delays or complex state management

# Main execution function
def main():
    """Main execution function"""
    # Initialize configuration
    audio_config = AudioConfig()
    model_config = ModelConfig()
    
    # Initialize voice assistant
    assistant = VoiceAssistant(model_config, audio_config)
    
    # Run console conversation mode
    assistant.run_conversation_loop()

if __name__ == "__main__":
    # Required for multiprocessing on macOS/Windows
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()