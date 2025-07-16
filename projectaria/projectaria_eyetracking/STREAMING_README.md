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
- `--show-window`: Enable OpenCV visualization window (with undistorted RGB)
- `--no-rerun`: Disable Rerun visualization

## Features

1. **Real-time Eye Tracking**: Processes eye camera images as they stream from the device
2. **RGB Camera Integration**: Shows gaze point projected onto RGB camera view
3. **Switchable View Modes**: Toggle between fisheye and undistorted RGB views (press 'f')
4. **Color Correction**: Advanced color correction with CLAHE enhancement (press 'c' to toggle)
5. **Accurate Gaze Projection**: Gaze projection adapts to the current view mode
6. **Gaze Visualization**: Shows gaze direction with uncertainty bounds
7. **Multiple Output Options**:
   - Rerun visualization (3D viewer)
   - OpenCV window (2D overlay)
   - CSV export for analysis
8. **Performance Monitoring**: Shows FPS counter for tracking performance

## Visualization

**OpenCV Windows** (--show-window): 
- **Main Window**: Shows two panels side by side:
  - **Left Panel (Eye Camera)**:
    - Eye camera image
    - Gaze direction arrow (green)
    - Uncertainty bounds (orange lines)
  - **Right Panel (RGB Camera)**:
    - Switchable between fisheye and undistorted views
    - Color correction with CLAHE enhancement (toggleable)
    - Projected gaze point (green circle with crosshair)
    - View mode indicator (FISHEYE VIEW / UNDISTORTED VIEW)
    - Color correction status indicator
    - Yaw/Pitch angles in degrees
    - FPS counter

- **Plain Fisheye Window** (toggleable):
  - Shows raw fisheye RGB image without any processing
  - No color correction or undistortion applied
  - Gaze point overlay (when calibration available)
  - Toggle on/off with 'p' key

## Controls

- Press **'f'** to toggle between fisheye and undistorted RGB views (main window)
- Press **'c'** to toggle color correction on/off
- Press **'p'** to toggle plain fisheye window on/off
- Press **'q'** or Ctrl+C to quit

## Notes

- Ensure your device is properly connected before running
- The model will be downloaded on first run (~50MB)
- WiFi streaming requires the device to be on the same network
- USB streaming typically provides better performance