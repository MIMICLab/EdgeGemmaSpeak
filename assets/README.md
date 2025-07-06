# Assets Directory

This directory contains sample audio files for CosyVoice prompt audio.

## CosyVoice Prompt Audio

To use CosyVoice with custom voice cloning, you need a prompt audio file (WAV format, 16kHz).

### Using default prompt
If no prompt is specified, CosyVoice will look for the default prompt in:
`third_party/CosyVoice/asset/zero_shot_prompt.wav`

### Using custom prompt
You can specify a custom prompt audio file with:
```bash
agentvox --tts cosyvoice --cosyvoice-prompt /path/to/your/prompt.wav
```

### Recording a prompt audio
Record a 5-10 second audio sample with clear speech:
```bash
# Using sox (Linux/Mac)
sox -d -r 16000 -c 1 prompt.wav trim 0 10

# Using ffmpeg
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -t 10 prompt.wav
```

The prompt audio should contain:
- Clear speech without background noise
- Natural speaking pace
- The voice characteristics you want to clone