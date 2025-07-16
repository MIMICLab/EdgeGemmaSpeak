# Streaming Eye Gaze Demo

This demo shows real-time eye gaze tracking using Project Aria glasses with live streaming data.

## Prerequisites

1. Project Aria glasses with streaming capabilities
2. Python environment with required packages installed
3. Eye tracking model downloaded (will auto-download on first run)

## Usage

### Basic streaming (WiFi connection):
```bash
python projectaria_eyetracking/projectaria_eyetracking/streaming_eye_gaze_demo.py --device-ip 192.168.0.23
```

### USB streaming:
```bash
python projectaria_eyetracking/projectaria_eyetracking/streaming_eye_gaze_demo.py --interface usb
```

### With OpenCV visualization window:
```bash
python projectaria_eyetracking/projectaria_eyetracking/streaming_eye_gaze_demo.py --device-ip 192.168.0.23 --show-window
```

### Save gaze data to CSV:
```bash
python projectaria_eyetracking/projectaria_eyetracking/streaming_eye_gaze_demo.py --device-ip 192.168.0.23 --output_csv gaze_data.csv
```

### Disable Rerun visualization:
```bash
python projectaria_eyetracking/projectaria_eyetracking/streaming_eye_gaze_demo.py --device-ip 192.168.0.23 --no-rerun
```

### Using CUDA for faster inference:
```bash
python projectaria_eyetracking/projectaria_eyetracking/streaming_eye_gaze_demo.py --device-ip 192.168.0.23 --device cuda
```

### Using MPS (Apple Silicon) for faster inference:
```bash
python projectaria_eyetracking/projectaria_eyetracking/streaming_eye_gaze_demo.py --device-ip 192.168.0.23 --device mps
```

### Using a different model:
```bash
python projectaria_eyetracking/projectaria_eyetracking/streaming_eye_gaze_demo.py --model_checkpoint_path /path/to/custom/model.pth
```

## Command Line Arguments

- `--interface`: Connection interface (usb or wifi, default: wifi)
- `--device-ip`: Device IP address (required for WiFi)
- `--profile-name`: Streaming profile (default: profile18)
- `--model_checkpoint_path`: Path to model checkpoint file (required)
- `--model_config_path`: Path to model config file (optional)
- `--device`: Device for inference (cpu, cuda, or mps, default: cpu)
- `--output_csv`: Path to save gaze data CSV
- `--show-window`: Enable OpenCV visualization window
- `--no-rerun`: Disable Rerun visualization

## Features

1. **Real-time Eye Tracking**: Processes eye camera images as they stream from the device
2. **Gaze Visualization**: Shows gaze direction with uncertainty bounds
3. **Multiple Output Options**:
   - Rerun visualization (3D viewer)
   - OpenCV window (2D overlay)
   - CSV export for analysis
4. **Performance Monitoring**: Shows FPS counter for tracking performance

## Visualization

The OpenCV window shows:
- Eye camera image
- Gaze direction arrow (green)
- Uncertainty bounds (orange lines)
- Yaw/Pitch angles in degrees
- FPS counter

Press 'q' in the OpenCV window or Ctrl+C in terminal to stop.

## Notes

- Ensure your device is properly connected before running
- The model will be downloaded on first run (~50MB)
- WiFi streaming requires the device to be on the same network
- USB streaming typically provides better performance