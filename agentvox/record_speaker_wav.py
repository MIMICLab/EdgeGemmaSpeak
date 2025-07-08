"""
Speaker voice recording tool for AgentVox
Records high-quality voice samples for TTS voice cloning
"""

import argparse
import sys
import os
import wave
import tempfile
import speech_recognition as sr
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List


# Language-specific prompts for voice recording
RECORDING_PROMPTS = {
    "ko": {
        "title": "음성 샘플 녹음",
        "instructions": "아래 문장을 읽어주세요. 자연스럽고 평소 말하는 톤으로 읽어주시면 됩니다.",
        "prompts": [
            "안녕하세요. 저는 오늘 날씨가 정말 좋다고 생각합니다.",
            "인공지능 기술은 우리의 일상생활을 많이 변화시키고 있습니다.",
            "좋은 하루 되세요. 다음에 또 만나요.",
            "미믹랩은 서강대학교의 연구실입니다.",
            "저는 당신과 대화하는 것을 즐깁니다."
        ],
        "ready": "준비되면 Enter를 누르고 말씀해주세요:",
        "recording": "녹음 중... (말이 끝나면 자동으로 멈춥니다)",
        "recorded": "녹음 완료!",
        "save_prompt": "이 녹음을 저장하시겠습니까? (y/n): ",
        "retry": "다시 녹음하시겠습니까?",
        "next": "다음 문장으로 진행",
        "finished": "모든 녹음이 완료되었습니다!",
        "saved_as": "저장 위치:",
        "quality_check": "녹음 품질 확인 중...",
        "quality_good": "✓ 녹음 품질이 좋습니다.",
        "quality_poor": "✗ 녹음 품질이 낮습니다. 다시 녹음해주세요.",
        "no_speech": "음성이 감지되지 않았습니다.",
        "error": "오류가 발생했습니다:"
    },
    "en": {
        "title": "Voice Sample Recording",
        "instructions": "Please read the following sentences. Speak naturally in your normal tone.",
        "prompts": [
            "Hello. I think the weather is really nice today.",
            "Artificial intelligence technology is changing our daily lives significantly.",
            "Have a great day. See you next time.",
            "MimicLab is a research laboratory at Sogang University.",
            "I enjoy having conversations with you."
        ],
        "ready": "Press Enter when ready to speak:",
        "recording": "Recording... (will stop automatically when you finish speaking)",
        "recorded": "Recording complete!",
        "save_prompt": "Save this recording? (y/n): ",
        "retry": "Record again?",
        "next": "Proceed to next sentence",
        "finished": "All recordings completed!",
        "saved_as": "Saved to:",
        "quality_check": "Checking recording quality...",
        "quality_good": "✓ Recording quality is good.",
        "quality_poor": "✗ Recording quality is poor. Please record again.",
        "no_speech": "No speech detected.",
        "error": "An error occurred:"
    },
    "ja": {
        "title": "音声サンプル録音",
        "instructions": "以下の文章を読んでください。自然な普段の口調で読んでください。",
        "prompts": [
            "こんにちは。今日は本当にいい天気だと思います。",
            "人工知能技術は私たちの日常生活を大きく変えています。",
            "良い一日を。また会いましょう。",
            "ミミックラボは西江大学の研究室です。",
            "私はあなたと会話することを楽しんでいます。"
        ],
        "ready": "準備ができたらEnterを押して話してください：",
        "recording": "録音中...（話し終わると自動的に停止します）",
        "recorded": "録音完了！",
        "save_prompt": "この録音を保存しますか？ (y/n): ",
        "retry": "もう一度録音しますか？",
        "next": "次の文章へ進む",
        "finished": "すべての録音が完了しました！",
        "saved_as": "保存先：",
        "quality_check": "録音品質を確認中...",
        "quality_good": "✓ 録音品質は良好です。",
        "quality_poor": "✗ 録音品質が低いです。もう一度録音してください。",
        "no_speech": "音声が検出されませんでした。",
        "error": "エラーが発生しました："
    },
    "zh": {
        "title": "语音样本录制",
        "instructions": "请朗读以下句子。用您平常的语调自然地说话。",
        "prompts": [
            "你好。我觉得今天天气真的很好。",
            "人工智能技术正在显著改变我们的日常生活。",
            "祝你有美好的一天。下次见。",
            "MimicLab是西江大学的研究实验室。",
            "我喜欢和你交谈。"
        ],
        "ready": "准备好后按Enter键开始说话：",
        "recording": "录音中...（说完后会自动停止）",
        "recorded": "录音完成！",
        "save_prompt": "保存此录音吗？ (y/n): ",
        "retry": "重新录音？",
        "next": "进入下一句",
        "finished": "所有录音已完成！",
        "saved_as": "保存位置：",
        "quality_check": "正在检查录音质量...",
        "quality_good": "✓ 录音质量良好。",
        "quality_poor": "✗ 录音质量较差。请重新录音。",
        "no_speech": "未检测到语音。",
        "error": "发生错误："
    }
}


class SpeakerRecorder:
    def __init__(self, language: str = "ko", sample_rate: int = 22050):
        self.language = language
        self.sample_rate = sample_rate
        self.prompts = RECORDING_PROMPTS.get(language, RECORDING_PROMPTS["en"])
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 1500
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 1.5
        self.microphone = sr.Microphone(sample_rate=sample_rate)
        
        # Adjust for ambient noise
        with self.microphone as source:
            print(f"\n{self.prompts['quality_check']}")
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
    
    def check_audio_quality(self, audio_data: np.ndarray) -> bool:
        """Check if audio quality is sufficient for voice cloning"""
        # Check volume level
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude < 0.01:  # Too quiet
            return False
        
        # Check for clipping
        if np.sum(np.abs(audio_data) > 0.95) > len(audio_data) * 0.01:  # More than 1% clipping
            return False
        
        # Check signal-to-noise ratio (simple version)
        signal_power = np.mean(audio_data ** 2)
        if signal_power < 0.001:  # Too weak signal
            return False
            
        return True
    
    def record_single_prompt(self, prompt: str) -> tuple[np.ndarray, bool]:
        """Record a single prompt and return audio data"""
        print(f"\n{'='*60}")
        print(f"📝 {prompt}")
        print(f"{'='*60}")
        
        input(f"\n{self.prompts['ready']} ")
        
        with self.microphone as source:
            print(f"{self.prompts['recording']}")
            
            try:
                # Record with timeout
                audio = self.recognizer.listen(
                    source,
                    timeout=30,
                    phrase_time_limit=20  # Maximum 20 seconds per recording
                )
                
                # Convert to numpy array
                wav_data = audio.get_wav_data()
                audio_array = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                print(f"{self.prompts['recorded']}")
                
                # Check quality
                print(f"{self.prompts['quality_check']}")
                if self.check_audio_quality(audio_array):
                    print(f"{self.prompts['quality_good']}")
                    return audio_array, True
                else:
                    print(f"{self.prompts['quality_poor']}")
                    return audio_array, False
                    
            except sr.WaitTimeoutError:
                print(f"{self.prompts['no_speech']}")
                return None, False
            except Exception as e:
                print(f"{self.prompts['error']} {e}")
                return None, False
    
    def record_all_prompts(self, output_path: str):
        """Record all prompts and save to a single file"""
        print(f"\n{self.prompts['title']}")
        print(f"{self.prompts['instructions']}\n")
        
        all_audio = []
        
        for i, prompt in enumerate(self.prompts['prompts']):
            while True:
                audio_data, quality_ok = self.record_single_prompt(prompt)
                
                if audio_data is not None:
                    # Play back the recording (optional)
                    response = input(f"\n{self.prompts['save_prompt']}").lower()
                    
                    if response == 'y':
                        all_audio.append(audio_data)
                        if i < len(self.prompts['prompts']) - 1:
                            print(f"✓ {self.prompts['next']}")
                        break
                    else:
                        retry = input(f"{self.prompts['retry']} (y/n): ").lower()
                        if retry != 'y':
                            break
                else:
                    retry = input(f"{self.prompts['retry']} (y/n): ").lower()
                    if retry != 'y':
                        break
        
        if all_audio:
            # Combine all audio with short pauses
            pause_samples = int(0.5 * self.sample_rate)  # 0.5 second pause
            pause = np.zeros(pause_samples)
            
            combined_audio = []
            for i, audio in enumerate(all_audio):
                combined_audio.extend(audio)
                if i < len(all_audio) - 1:
                    combined_audio.extend(pause)
            
            combined_audio = np.array(combined_audio)
            
            # Save as WAV file
            sf.write(output_path, combined_audio, self.sample_rate)
            print(f"\n{self.prompts['finished']}")
            print(f"{self.prompts['saved_as']} {output_path}")
            
            # Print file info
            duration = len(combined_audio) / self.sample_rate
            print(f"Duration: {duration:.1f} seconds")
            print(f"Sample rate: {self.sample_rate} Hz")
            
            return True
        else:
            print(f"\n{self.prompts['error']} No audio recorded.")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Record speaker voice samples for AgentVox TTS voice cloning"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="ko",
        choices=["ko", "en", "ja", "zh"],
        help="Language for recording prompts (default: ko)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: speaker_[language].wav)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate for recording (default: 22050 Hz, recommended for TTS)"
    )
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if args.output is None:
        args.output = f"speaker_{args.language}.wav"
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create recorder and start recording
    recorder = SpeakerRecorder(
        language=args.language,
        sample_rate=args.sample_rate
    )
    
    try:
        success = recorder.record_all_prompts(str(output_path))
        if success:
            print(f"\n✅ Voice sample ready for use with AgentVox!")
            print(f"Usage: agentvox --tts-engine coqui --speaker-wav {output_path}")
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nRecording cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()