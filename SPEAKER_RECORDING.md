# Speaker Voice Recording Guide

AgentVox now includes a built-in tool for recording high-quality voice samples for TTS voice cloning.

## Quick Start

Record your voice sample:
```bash
agentvox --record-speaker
```

This will:
1. Display recording prompts in your selected language
2. Guide you through recording 5 sample sentences
3. Check audio quality automatically
4. Save the combined recording as `speaker_ko.wav` (or appropriate language code)

## Language-Specific Recording

The recording tool automatically adapts to your STT language setting:

```bash
# Korean (default)
agentvox --record-speaker --stt-language ko

# English
agentvox --record-speaker --stt-language en

# Japanese
agentvox --record-speaker --stt-language ja

# Chinese
agentvox --record-speaker --stt-language zh
```

## Custom Output Path

Specify where to save the recording:
```bash
agentvox --record-speaker --speaker-wav ./voices/my_voice.wav
```

## Using Your Voice Sample

After recording, use your voice with Coqui TTS:
```bash
agentvox --tts-engine coqui --speaker-wav speaker_ko.wav
```

## Recording Tips

1. **Environment**: Record in a quiet room with minimal echo
2. **Microphone**: Use a good quality microphone if possible
3. **Speaking Style**: Speak naturally in your normal tone
4. **Consistency**: Keep the same distance from the microphone
5. **Quality**: The tool will automatically check recording quality

## Sample Prompts

The tool provides language-specific prompts designed for optimal voice cloning:

### Korean (ko)
- 안녕하세요. 저는 오늘 날씨가 정말 좋다고 생각합니다.
- 인공지능 기술은 우리의 일상생활을 많이 변화시키고 있습니다.
- 좋은 하루 되세요. 다음에 또 만나요.
- 미믹랩은 서강대학교의 연구실입니다.
- 저는 당신과 대화하는 것을 즐깁니다.

### English (en)
- Hello. I think the weather is really nice today.
- Artificial intelligence technology is changing our daily lives significantly.
- Have a great day. See you next time.
- MimicLab is a research laboratory at Sogang University.
- I enjoy having conversations with you.

### Japanese (ja)
- こんにちは。今日は本当にいい天気だと思います。
- 人工知能技術は私たちの日常生活を大きく変えています。
- 良い一日を。また会いましょう。
- ミミックラボは西江大学の研究室です。
- 私はあなたと会話することを楽しんでいます。

### Chinese (zh)
- 你好。我觉得今天天气真的很好。
- 人工智能技术正在显著改变我们的日常生活。
- 祝你有美好的一天。下次见。
- MimicLab是西江大学的研究实验室。
- 我喜欢和你交谈。

## Audio Quality Requirements

The recording tool automatically checks for:
- Sufficient volume level
- No audio clipping
- Good signal-to-noise ratio

If quality is poor, you'll be prompted to re-record.

## Advanced Usage

### Standalone Recording Script

You can also use the recording module directly:

```python
from agentvox import SpeakerRecorder

recorder = SpeakerRecorder(language="ko", sample_rate=22050)
recorder.record_all_prompts("my_voice.wav")
```

### Custom Sample Rate

For higher quality recordings:
```bash
python -m agentvox.record_speaker_wav --sample-rate 44100 --output high_quality_voice.wav
```

## Troubleshooting

1. **No microphone detected**: Ensure your microphone is connected and permissions are granted
2. **Poor quality warnings**: Try recording in a quieter environment
3. **Recording too quiet**: Move closer to the microphone or increase input gain
4. **Recording clips/distorts**: Move further from the microphone or reduce input gain

## Best Practices for Voice Cloning

1. **Recording Length**: The tool records about 30-60 seconds of speech, which is optimal for XTTS v2
2. **Consistency**: Try to maintain consistent tone and pace throughout recording
3. **Multiple Takes**: Don't hesitate to re-record if you're not satisfied
4. **Natural Speech**: Avoid overly dramatic or unnatural pronunciation

## Integration with AgentVox

Once you have a good recording, AgentVox will use it to clone your voice for all TTS outputs:

```bash
# Start AgentVox with your voice
agentvox --tts-engine coqui --speaker-wav speaker_ko.wav --stt-language ko
```

The assistant will now speak in your cloned voice!