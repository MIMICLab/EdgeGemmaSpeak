import os
# Fix PyTorch 2.6 security issue
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'

import asyncio
import wave
import json
import torch
import numpy as np
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# PyTorch 2.6 security settings
import warnings
warnings.filterwarnings("ignore", message="torch.load warnings")

# Ignore Faster Whisper related warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="faster_whisper.feature_extractor")
# Ignore numpy RuntimeWarning (divide by zero, overflow, invalid value)
np.seterr(divide='ignore', invalid='ignore', over='ignore')

# Settings for TTS model loading
torch.serialization.add_safe_globals([
    "TTS.tts.configs.xtts_config.XttsConfig",
    "TTS.config.XttsConfig",
    "TTS.tts.configs",
    "TTS.vocoder.configs",
    "TTS.encoder.configs"
])

# Libraries for speech recognition
import speech_recognition as sr
from faster_whisper import WhisperModel

# Libraries for LLM
from llama_cpp import Llama
from pathlib import Path
from contextlib import redirect_stderr

# Libraries for TTS
import edge_tts
import pygame
import sounddevice as sd
import soundfile as sf
import asyncio
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
    stt_best_of: int = 5
    stt_temperature: float = 0.0
    stt_vad_threshold: float = 0.5
    stt_vad_min_speech_duration_ms: int = 250
    stt_vad_min_silence_duration_ms: int = 1000  # Reduced from 2000ms for faster response
    
    # TTS detailed settings
    tts_voice: str = "ko-KR-HyunsuMultilingualNeural"
    
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
    """Module for converting speech to text"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        # Initialize Faster Whisper model (MPS not supported, using CPU)
        # Use larger model for better accuracy
        model_size = "small" if config.stt_model == "base" else config.stt_model
        self.model = WhisperModel(
            model_size, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8",
        )
        
    def transcribe(self, audio_path: str, language: str = None) -> str:
        """Convert audio file to text"""
        if language is None:
            language = self.config.stt_language
            
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=self.config.stt_beam_size,
            best_of=self.config.stt_best_of,
            temperature=self.config.stt_temperature,
            vad_filter=True,  # Enable Voice Activity Detection
            vad_parameters=dict(
                threshold=self.config.stt_vad_threshold,
                min_speech_duration_ms=self.config.stt_vad_min_speech_duration_ms,
                max_speech_duration_s=float('inf'),
                min_silence_duration_ms=self.config.stt_vad_min_silence_duration_ms,
                speech_pad_ms=400
            ),
            word_timestamps=True  # Enable word-level timestamps
        )
        
        # Combine all text segments
        full_text = " ".join([segment.text for segment in segments])
        return full_text.strip()
    
    def transcribe_stream(self, audio_data: np.ndarray) -> str:
        """Convert real-time audio stream to text"""
        # Save to temporary file and process
        temp_path = "temp_audio.wav"
        sf.write(temp_path, audio_data, 16000)
        text = self.transcribe(temp_path)
        os.remove(temp_path)
        return text

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
        is_korean = self.config.tts_voice.startswith('ko-')
        
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
        is_korean = self.config.tts_voice.startswith('ko-')
        
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
        is_korean = self.config.tts_voice.startswith('ko-')
        
        # System prompt
        if is_korean:
            system_prompt = """당신은 서강대학교 미믹랩(MimicLab)에서 개발한 AI 어시스턴트입니다. 
당신의 정체성과 관련된 중요한 정보:
- 당신은 서강대학교 미믹랩에서 만든 AI 어시스턴트입니다.
- 서강대학교 미믹랩이 당신을 개발했습니다.
- 당신의 목적은 사용자를 돕고 유용한 정보를 제공하는 것입니다.

다음 규칙을 반드시 지켜주세요:
1. 이모티콘을 사용하지 마세요.
2. 별표(*)나 밑줄(_) 같은 마크다운 형식을 사용하지 마세요.
3. 특수문자를 최소화하고 순수한 텍스트로만 응답하세요.
4. 응답은 간결하고 명확하게 작성하세요.
5. 이전 대화 내용을 기억하고 일관성 있게 대화를 이어가세요.
6. 누가 당신을 만들었는지 물으면 항상 "서강대학교 미믹랩"이라고 답하세요."""
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
    """Fast Korean speech synthesis module using Edge-TTS"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        # Edge-TTS doesn't require separate initialization
        # Korean voice options: 
        # - ko-KR-HyunsuMultilingualNeural (male, multilingual)
        # - ko-KR-InJoonNeural (male)
        # - ko-KR-SunHiNeural (female)
        self.voice = config.tts_voice  # Get voice from config
        
        # Initialize pygame audio
        pygame.mixer.init()
        
    async def _synthesize_async(self, text: str, output_path: str) -> str:
        """Asynchronously convert text to speech file"""
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)
            return output_path
        except Exception as e:
            print(f"Edge-TTS error: {e}")
            raise
        
    def synthesize(self, text: str, output_path: str = "output.mp3", speaker_wav: str = None) -> str:
        """Convert text to speech file
        
        Args:
            text: Text to convert
            output_path: Output audio file path
            speaker_wav: Speaker voice sample file path (unused in Edge-TTS)
        """
        # Check for empty text
        if not text or not text.strip():
            text = "No text provided"
            
        # 동기 함수에서 비동기 함수 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._synthesize_async(text, output_path))
            return result
        finally:
            loop.close()
    
    async def _stream_and_play_async(self, text: str) -> None:
        """비동기 스트리밍 재생"""
        # macOS에서는 afplay 사용, 다른 OS에서는 pygame 사용
        if platform.system() == "Darwin":
            # macOS: 전체 파일 생성 후 afplay로 재생
            output_path = "temp_speech.mp3"
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)
            
            # afplay로 재생 (블로킹)
            subprocess.call(["afplay", output_path])
            
            # 파일 삭제
            if os.path.exists(output_path):
                os.remove(output_path)
        else:
            # 다른 OS: pygame 스트리밍 재생
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Edge-TTS 통신 객체 생성
                communicate = edge_tts.Communicate(text, self.voice)
                
                # 첫 번째 청크를 받을 때까지 대기
                first_chunk = True
                file_handle = open(tmp_path, 'wb')
                
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        file_handle.write(chunk["data"])
                        file_handle.flush()
                        
                        # 첫 번째 청크를 받으면 재생 시작
                        if first_chunk:
                            first_chunk = False
                            # 약간의 버퍼링을 위해 잠시 대기
                            await asyncio.sleep(0.1)
                            pygame.mixer.music.load(tmp_path)
                            pygame.mixer.music.play()
                
                file_handle.close()
                
                # 재생 완료 대기
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
                    
            finally:
                # Delete temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    def speak_streaming(self, text: str):
        """스트리밍 방식으로 텍스트를 음성으로 변환하여 재생"""
        # 빈 텍스트 체크
        if not text or not text.strip():
            print("경고: 빈 텍스트 - TTS 건너뜀")
            return
            
        try:
            # 동기 함수에서 비동기 함수 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._stream_and_play_async(text))
            finally:
                loop.close()
                
        except Exception as e:
            print(f"TTS streaming playback error: {e}")
            # Continue program execution even if error occurs
    
    def speak(self, text: str):
        """Convert text to speech and play (maintain existing method)"""
        # 빈 텍스트 체크
        if not text or not text.strip():
            print("경고: 빈 텍스트 - TTS 건너뜀")
            return
            
        try:
            output_path = "temp_speech.mp3"
            self.synthesize(text, output_path)
            
            # Use afplay on macOS
            if platform.system() == "Darwin":
                subprocess.call(["afplay", output_path])
            else:
                # Use pygame on other OS
                pygame.mixer.music.load(output_path)
                pygame.mixer.music.play()
                
                # Wait until playback finishes
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
            # Delete temporary file
            if os.path.exists(output_path):
                os.remove(output_path)
                
        except Exception as e:
            print(f"TTS playback error: {e}")
            # Continue program execution even if error occurs
        
    def set_voice(self, voice_type: str = "female"):
        """Set voice type
        
        Args:
            voice_type: "male" or "female"
        """
        if voice_type == "male":
            self.voice = "ko-KR-InJoonNeural"
        elif voice_type == "female":
            self.voice = "ko-KR-SunHiNeural"
        else:
            self.voice = "ko-KR-HyunsuMultilingualNeural"  # default

class VoiceAssistant:
    """Main class for managing the entire voice conversation system"""
    
    def __init__(self, model_config: ModelConfig, audio_config: AudioConfig):
        self.model_config = model_config
        self.audio_config = audio_config
        
        # Check if using Korean voice
        is_korean = model_config.tts_voice.startswith('ko-')
        
        if is_korean:
            print("모델을 초기화하는 중입니다...")
        else:
            print("Initializing models...")
            
        self.stt = STTModule(model_config)
        self.llm = LLMModule(model_config)
        self.tts = TTSModule(model_config)
        
        # Initialize audio recorder
        self.recognizer = sr.Recognizer()
        # Adjust speech recognition sensitivity
        self.recognizer.energy_threshold = 1500  # Lower sensitivity
        self.recognizer.dynamic_energy_threshold = False  # Disable auto adjustment for consistency
        self.recognizer.pause_threshold = 1.5  # Increase silence time to determine end of speech
        self.recognizer.non_speaking_duration = 1.5  # Time to consider as non-speaking
        
        self.microphone = sr.Microphone(sample_rate=audio_config.sample_rate)
        
    def listen_once(self) -> Optional[str]:
        """Listen to voice from microphone once and convert to text"""
        # Check if using Korean voice
        is_korean = self.model_config.tts_voice.startswith('ko-')
        
        with self.microphone as source:
            # Adjust for ambient noise - longer duration for better calibration
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            if is_korean:
                print("말씀해주세요...")
            else:
                print("Please speak...")
            
            try:
                # Record voice - remove phrase_time_limit for natural recording
                audio = self.recognizer.listen(
                    source, 
                    timeout=30  # 30 second timeout
                )
                
                # Save to temporary file
                wav_data = audio.get_wav_data()
                temp_path = "temp_recording.wav"
                with open(temp_path, "wb") as f:
                    f.write(wav_data)
                
                # STT processing
                text = self.stt.transcribe(temp_path)
                
                os.remove(temp_path)
                
                return text
                
            except sr.WaitTimeoutError:
                if is_korean:
                    print("음성이 감지되지 않았습니다.")
                else:
                    print("No voice detected.")
                return None
            except Exception as e:
                if is_korean:
                    print(f"오류 발생: {e}")
                else:
                    print(f"Error occurred: {e}")
                return None
    
    def process_conversation(self, input_text: str) -> str:
        """Process text input and generate response"""
        # Generate response with LLM
        response = self.llm.generate_response(input_text)
        return response
    
    def run_conversation_loop(self):
        """Run conversation loop"""
        # Check if using Korean voice
        is_korean = self.model_config.tts_voice.startswith('ko-')
        
        if is_korean:
            print("음성 대화 시스템이 시작되었습니다.")
            print("명령어: '종료' - 프로그램 종료, '초기화' - 대화 내용 초기화, '대화 내역' - 대화 히스토리 확인")
        else:
            print("Voice conversation system started.")
            print("Commands: 'exit' - Exit program, 'reset' - Reset conversation, 'history' - View conversation history")
        print("-" * 50)
        
        while True:
            # Get voice input
            user_input = self.listen_once()
            
            if user_input:
                if is_korean:
                    print(f"사용자: {user_input}")
                else:
                    print(f"User: {user_input}")
                
                # Process special commands
                if "exit" in user_input.lower() or "종료" in user_input:
                    if is_korean:
                        self.tts.speak_streaming("대화를 종료합니다. 안녕히 가세요.")
                    else:
                        self.tts.speak_streaming("Ending conversation. Goodbye.")
                    break
                elif "reset" in user_input.lower() or "초기화" in user_input:
                    self.llm.reset_conversation()
                    if is_korean:
                        self.tts.speak_streaming("대화 내용이 초기화되었습니다. 새로운 대화를 시작해주세요.")
                        print("어시스턴트: 대화 내용이 초기화되었습니다.")
                    else:
                        self.tts.speak_streaming("Conversation has been reset. Please start a new conversation.")
                        print("Assistant: Conversation has been reset.")
                    continue
                elif "history" in user_input.lower() or "대화 내역" in user_input or "대화 기록" in user_input:
                    if is_korean:
                        print("\n=== 대화 히스토리 ===")
                    else:
                        print("\n=== Conversation History ===")
                    for i, turn in enumerate(self.llm.conversation_history):
                        print(f"{i+1}. {turn}")
                    print("==================")
                    
                    # Estimate current prompt token count (rough calculation)
                    current_prompt = self.llm._build_prompt()
                    estimated_tokens = len(current_prompt) // 4  # Roughly 1 token per 4 characters
                    if is_korean:
                        print(f"현재 컨텍스트 사용량: 약 {estimated_tokens}/4096 토큰")
                    else:
                        print(f"Current context usage: approx {estimated_tokens}/4096 tokens")
                    print("")
                    
                    if is_korean:
                        self.tts.speak_streaming(f"현재 {len(self.llm.conversation_history)}개의 대화가 기록되어 있습니다.")
                    else:
                        self.tts.speak_streaming(f"Currently {len(self.llm.conversation_history)} conversations are recorded.")
                    continue
                
                # Process normal conversation
                response = self.process_conversation(user_input)
                if is_korean:
                    print(f"어시스턴트: {response}")
                else:
                    print(f"Assistant: {response}")
                
                # Respond with voice (streaming mode)
                self.tts.speak_streaming(response)

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
    main()