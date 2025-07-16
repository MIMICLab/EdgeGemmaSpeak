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


import ctypes
from typing import (
    List,
    Literal,
    Tuple,
)

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
import llama_cpp.llama as llama
import llama_cpp
from llama_cpp.llama_chat_format import Llava15ChatHandler
from pathlib import Path
from contextlib import redirect_stderr

# Libraries for audio
import pygame
import soundfile as sf
import tempfile
import subprocess
import platform

# Gemma3 Chat Handler for multimodal support
class Gemma3ChatHandler(Llava15ChatHandler):
    # Chat Format:
    # '<bos><start_of_turn>user\n{system_prompt}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n'

    DEFAULT_SYSTEM_MESSAGE = None

    CHAT_FORMAT = (
        "{{ '<bos>' }}"
        "{%- if messages[0]['role'] == 'system' -%}"
        "{%- if messages[0]['content'] is string -%}"
        "{%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}"
        "{%- else -%}"
        "{%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}"
        "{%- endif -%}"
        "{%- set loop_messages = messages[1:] -%}"
        "{%- else -%}"
        "{%- set first_user_prefix = \"\" -%}"
        "{%- set loop_messages = messages -%}"
        "{%- endif -%}"
        "{%- for message in loop_messages -%}"
        "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}"
        "{{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}"
        "{%- endif -%}"
        "{%- if (message['role'] == 'assistant') -%}"
        "{%- set role = \"model\" -%}"
        "{%- else -%}"
        "{%- set role = message['role'] -%}"
        "{%- endif -%}"
        "{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}"
        "{%- if message['content'] is string -%}"
        "{{ message['content'] | trim }}"
        "{%- elif message['content'] is iterable -%}"
        "{%- for item in message['content'] -%}"
        "{%- if item['type'] == 'image_url' -%}"
        "{{ '<start_of_image>' }}"
        "{%- elif item['type'] == 'text' -%}"
        "{{ item['text'] | trim }}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- else -%}"
        "{{ raise_exception(\"Invalid content type\") }}"
        "{%- endif -%}"
        "{{ '<end_of_turn>\n' }}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "{{ '<start_of_turn>model\n' }}"
        "{%- endif -%}"
    )

    @staticmethod
    def split_text_on_image_urls(text: str, image_urls: List[str]):
        split_text: List[Tuple[Literal["text", "image_url"], str]] = []
        copied_urls = image_urls[:]
        remaining = text
        image_placeholder = "<start_of_image>"

        while remaining:
            # Find placeholder
            pos = remaining.find(image_placeholder)
            if pos != -1:
                assert len(copied_urls) > 0
                if pos > 0:
                    split_text.append(("text", remaining[:pos]))
                split_text.append(("text", "\n\n<start_of_image>"))
                split_text.append(("image_url", copied_urls.pop(0)))
                split_text.append(("text", "<end_of_image>\n\n"))
                remaining = remaining[pos + len(image_placeholder):]
            else:
                assert len(copied_urls) == 0
                split_text.append(("text", remaining))
                remaining = ""
        return split_text

    def eval_image(self, llama: llama.Llama, image_url: str):

        n_tokens = 256
        if llama.n_tokens + n_tokens > llama.n_ctx():
            raise ValueError(
                f"Prompt exceeds n_ctx: {llama.n_tokens + n_tokens} > {llama.n_ctx()}"
            )

        img_bytes = self.load_image(image_url)
        img_u8_p = self._llava_cpp.clip_image_u8_init()
        if not self._llava_cpp.clip_image_load_from_bytes(
            ctypes.create_string_buffer(img_bytes, len(img_bytes)),
            ctypes.c_size_t(len(img_bytes)),
            img_u8_p,
        ):
            self._llava_cpp.clip_image_u8_free(img_u8_p)
            raise ValueError("Failed to load image.")

        img_f32 = self._llava_cpp.clip_image_f32_batch()
        img_f32_p = ctypes.byref(img_f32)
        if not self._llava_cpp.clip_image_preprocess(self.clip_ctx, img_u8_p, img_f32_p):
            self._llava_cpp.clip_image_f32_batch_free(img_f32_p)
            self._llava_cpp.clip_image_u8_free(img_u8_p)
            raise ValueError("Failed to preprocess image.")

        n_embd = llama_cpp.llama_model_n_embd(llama._model.model)
        embed = (ctypes.c_float * (n_tokens * n_embd))()
        if not self._llava_cpp.clip_image_batch_encode(self.clip_ctx, llama.n_threads, img_f32_p, embed):
            self._llava_cpp.clip_image_f32_batch_free(img_f32_p)
            self._llava_cpp.clip_image_u8_free(img_u8_p)
            raise ValueError("Failed to encode image.")

        self._llava_cpp.clip_image_f32_batch_free(img_f32_p)
        self._llava_cpp.clip_image_u8_free(img_u8_p)
        llama_cpp.llama_set_causal_attn(llama.ctx, False)

        seq_id_0 = (ctypes.c_int32 * 1)()
        seq_ids = (ctypes.POINTER(ctypes.c_int32) * (n_tokens + 1))()
        for i in range(n_tokens):
            seq_ids[i] = seq_id_0

        batch = llama_cpp.llama_batch()
        batch.n_tokens = n_tokens
        batch.token = None
        batch.embd = embed
        batch.pos = (ctypes.c_int32 * n_tokens)(*[i + llama.n_tokens for i in range(n_tokens)])
        batch.seq_id = seq_ids
        batch.n_seq_id = (ctypes.c_int32 * n_tokens)(*([1] * n_tokens))
        batch.logits = (ctypes.c_int8 * n_tokens)()

        if llama_cpp.llama_decode(llama.ctx, batch):
            raise ValueError("Failed to decode image.")

        llama_cpp.llama_set_causal_attn(llama.ctx, True)
        # Required to avoid issues with hf tokenizer
        llama.input_ids[llama.n_tokens : llama.n_tokens + n_tokens] = -1
        llama.n_tokens += n_tokens

def image_to_base64_data_uri(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Convert PIL Image to base64 data URI."""
    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=quality, optimize=True)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    mime_type = f"image/{format.lower()}"
    return f'data:{mime_type};base64,{img_base64}'

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
    mmproj_model: str = None  # Multimodal projection model path (for vision)
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"  # XTTS v2 multilingual model
    device: str = "auto"  # Device: auto, cpu, cuda, mps
    is_multimodal: bool = False  # Enable multimodal (vision) support
    
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
    """Local LLM response generation module using Llama.cpp with multimodal support"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = config.device
        self.is_multimodal = config.is_multimodal
        
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
        
        # Set up multimodal projection model path
        self.mmproj_path = None
        chat_handler = None
        
        if self.is_multimodal and config.mmproj_model:
            if not os.path.isabs(config.mmproj_model):
                # If relative path, resolve it relative to model directory
                model_dir = Path(self.model_path).parent
                self.mmproj_path = str(model_dir / config.mmproj_model)
            else:
                self.mmproj_path = config.mmproj_model
            
            # Check if mmproj file exists
            if not os.path.exists(self.mmproj_path):
                # Try default mmproj filename
                model_dir = Path(self.model_path).parent
                default_mmproj = model_dir / "mmproj-gemma-3-12b-it-F16.gguf"
                if default_mmproj.exists():
                    self.mmproj_path = str(default_mmproj)
                else:
                    raise FileNotFoundError(f"Multimodal projection model not found: {self.mmproj_path}")
            
            # Initialize chat handler for multimodal
            chat_handler = Gemma3ChatHandler(clip_model_path=self.mmproj_path, verbose=False)
        
        # Load Llama model
        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull):
                self.model = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=-1,  # Load all layers to GPU
                    n_ctx=self.config.llm_context_size,      # Context size
                    verbose=False,
                    flash_attn=True,   # Use Flash Attention
                    chat_handler=chat_handler  # Add chat handler for multimodal
                )
                self.tokenizer = LlamaTokenizer(self.model)
        
        # Manage conversation history
        self.conversation_history = []
        
    def generate_response(self, text: str, images: Optional[List[Image.Image]] = None, max_length: int = 512) -> str:
        """Generate response for input text, optionally with images for multimodal models"""
        # Check if using Korean voice
        is_korean = self.config.stt_language.startswith('ko')
        
        # Build conversation context
        if is_korean:
            self.conversation_history.append(f"사용자: {text}")
        else:
            self.conversation_history.append(f"User: {text}")
        
        # Handle multimodal input
        if images is not None and self.is_multimodal:
            try:
                # Merge images into a grid
                image = images[-1]
                # Convert image to data URI
                image_uri = image_to_base64_data_uri(image, format="JPEG", quality=90)
                
                # Use chat completion API for multimodal input
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {'type': 'text', 'text': text},
                            {'type': 'image_url', 'image_url': {'url': image_uri}}
                        ]
                    }
                ]
                
                response = self.model.create_chat_completion(
                    messages=messages,
                    stop=['<end_of_turn>', '<eos>'],
                    max_tokens=max_length if max_length != 512 else self.config.llm_max_tokens,  # Reduce for multimodal
                    temperature=self.config.llm_temperature,
                    top_p=self.config.llm_top_p,
                    repeat_penalty=self.config.llm_repeat_penalty
                )
                response_text = response['choices'][0]['message']['content'].strip()
            except Exception as e:
                # Fallback to text-only if multimodal fails
                is_korean = self.config.stt_language.startswith('ko')
                if is_korean:
                    print(f"⚠️ 멀티모달 처리 실패, 텍스트만 처리합니다: {e}")
                else:
                    print(f"⚠️ Multimodal processing failed, falling back to text-only: {e}")
                
                # Process as text-only
                prompt = self._build_prompt()
                answer = self.model(
                    prompt,
                    stop=['<end_of_turn>', '<eos>'],
                    max_tokens=max_length if max_length != 512 else self.config.llm_max_tokens,
                    echo=False,
                    temperature=self.config.llm_temperature,
                    top_p=self.config.llm_top_p,
                    repeat_penalty=self.config.llm_repeat_penalty,
                )
                response_text = answer['choices'][0]['text'].strip()
        else:
            # Text-only generation
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
            
            response_text = answer['choices'][0]['text'].strip()
        
        response = response_text
        
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
            system_prompt = """당신은 서강대학교 미믹랩에서 개발한 시각 어시스턴트입니다.

이미지 분석 규칙:
- 제공된 이미지는 사용자의 현재 시야입니다
- 초록색 점은 사용자가 보고 있는 위치를 표시한 것입니다 (시스템이 추가한 마커)
- 초록색 점 자체를 언급하지 말고, 그 위치에 있는 실제 객체나 내용을 설명하세요
- 메타 질문이나 설명 없이 바로 답변하세요

응답 규칙:
- 반드시 한국어로만 답변
- 한두 문장으로 짧고 간결하게 답변
- "사진", "이미지", "화면" 같은 단어 절대 사용하지 않기
- 별표(*), 하이픈(-), 콜론(:) 등 특수문자 사용하지 않기
- 리스트나 강조 표시 없이 일반 문장으로만 답변
- 불필요한 설명이나 추가 질문 하지 않기
- 영어는 한글로 표기 (예: AI→에이아이)"""
        else:
            system_prompt = """You are a visual assistant developed by MimicLab at Sogang University.

Image Analysis Rules:
- The provided image is the user's current view
- The green dot is a system marker showing where the user is looking
- Don't mention the green dot itself, describe the actual object or content at that location
- Answer directly without meta questions or explanations

Response Rules:
- Be concise and clear
- Never use words like "photo", "image", "picture", "screen"
- No special characters like asterisks (*), hyphens (-), colons (:)
- No lists or formatting, only plain sentences
- Plain text only, no markdown or emoticons"""
        
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
        
        # External audio source (for Aria integration)
        self.external_audio_source = None
        self.use_external_audio = False
        
        is_korean = model_config.stt_language.startswith('ko')
        
        if is_korean:
            print("모델을 초기화하는 중입니다...")
            if model_config.is_multimodal:
                print("멀티모달(비전) 기능이 활성화되었습니다.")
        else:
            print("Initializing models...")
            if model_config.is_multimodal:
                print("Multimodal (vision) capabilities enabled.")
            
        self.stt = STTModule(model_config)
        self.llm = LLMModule(model_config)
        self.tts = TTSModule(model_config)
        
        # Image buffer for multimodal input
        self.image_buffer = []
    
    def add_image(self, image: Image.Image):
        """Add an image to the buffer for multimodal input"""
        if not self.model_config.is_multimodal:
            print("Warning: Multimodal support is not enabled. Image will be ignored.")
            return
        
        self.image_buffer.append(image)
        is_korean = self.model_config.stt_language.startswith('ko')
        if is_korean:
            print(f"이미지가 추가되었습니다. 현재 {len(self.image_buffer)}개의 이미지가 있습니다.")
        else:
            print(f"Image added. Currently {len(self.image_buffer)} images in buffer.")
    
    def add_image_from_path(self, image_path: str):
        """Add an image from file path to the buffer"""
        try:
            image = Image.open(image_path)
            self.add_image(image)
        except Exception as e:
            is_korean = self.model_config.stt_language.startswith('ko')
            if is_korean:
                print(f"이미지 로드 실패: {e}")
            else:
                print(f"Failed to load image: {e}")
    
    def clear_images(self):
        """Clear all images from the buffer"""
        self.image_buffer = []
        is_korean = self.model_config.stt_language.startswith('ko')
        if is_korean:
            print("이미지 버퍼가 클리어되었습니다.")
        else:
            print("Image buffer cleared.")
    
    def set_external_audio_source(self, audio_source):
        """Set external audio source (e.g., Aria glasses)"""
        self.external_audio_source = audio_source
        self.use_external_audio = True
        is_korean = self.model_config.stt_language.startswith('ko')
        if is_korean:
            print("외부 오디오 소스(Aria)가 설정되었습니다.")
        else:
            print("External audio source (Aria) configured.")
    
    def listen_from_external_audio(self) -> Optional[str]:
        """Listen to audio from external source (Aria) and transcribe"""
        import time
        
        is_korean = self.model_config.stt_language.startswith('ko')
        
        if is_korean:
            print("\n🎤 Aria 마이크로 말씀해주세요...")
        else:
            print("\n🎤 Please speak into Aria microphone...")
        
        # Collect audio chunks from Aria
        audio_chunks = []
        silence_count = 0
        max_silence = 50  # About 5 seconds of silence
        
        while True:
            # Get audio chunk from Aria
            chunk = self.external_audio_source.get_audio_chunk(1600)  # 1600 samples ≈ 33ms at 48kHz
            
            if chunk is not None:
                audio_chunks.append(chunk)
                # Reset silence counter if we got audio
                silence_count = 0
            else:
                silence_count += 1
                if silence_count > max_silence:
                    break
                time.sleep(0.1)  # Wait 100ms before next check
        
        if audio_chunks:
            # Concatenate all chunks
            import numpy as np
            audio_data = np.concatenate(audio_chunks)
            
            # Feed to RealtimeSTT
            self.stt.recorder.feed_audio(audio_data)
            
            # Get transcription
            return self.stt.recorder.text()
        
        return None
    
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
            if self.use_external_audio and self.external_audio_source:
                # Use external audio source (Aria)
                user_input = self.listen_from_external_audio()
            else:
                # Use computer microphone
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
            
            # 2. Get LLM response (with images if available)
            images_to_use = self.image_buffer if self.image_buffer else None
            if images_to_use:
                if is_korean:
                    print(f"\n{len(images_to_use)}개의 이미지와 함께 응답을 생성하는 중...")
                else:
                    print(f"\nGenerating response with {len(images_to_use)} images...")
            
            response = self.llm.generate_response(user_input, images=images_to_use)
            print(f"\n어시스턴트: {response}" if is_korean else f"\nAssistant: {response}")
            
            # Clear images after use
            if images_to_use:
                self.clear_images()
            
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