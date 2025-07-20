#!/usr/bin/env python3
"""
Aria Eye Tracking + AgentVox Integration Example

This script demonstrates how to integrate Project Aria eye tracking
with AgentVox voice assistant to create a multimodal AI assistant
that can see what you're looking at and respond accordingly.

Usage:
    python aria_agentvox_integration.py --device-ip <ARIA_IP>

Requirements:
    - Project Aria glasses connected via WiFi
    - AgentVox with multimodal support enabled
    - Gemma 3 model with mmproj downloaded
"""

import argparse
import cv2
import numpy as np
import threading
import time
import logging
from PIL import Image
from pathlib import Path
import sys
import os

# Reduce DDS warnings
logging.getLogger('SubListener').setLevel(logging.CRITICAL)
logging.getLogger('dds').setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.ERROR)

# Suppress specific DDS warnings
import warnings
warnings.filterwarnings("ignore", message=".*sample lost.*")
warnings.filterwarnings("ignore", message=".*CRITICAL DDS.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# Add paths for Project Aria modules
base_dir = os.path.dirname(__file__)
sys.path.extend([
    os.path.join(base_dir, "projectaria/projectaria_eyetracking/projectaria_eyetracking"),
    os.path.join(base_dir, "agentvox"),
    os.path.join(base_dir, "projectaria/projectaria_client_sdk_samples")
])

# Project Aria imports
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core.calibration import device_calibration_from_json_string
from common import update_iptables

# AgentVox imports
from agentvox.voice_assistant import VoiceAssistant, ModelConfig, AudioConfig

# Eye tracking import
from inference.infer import EyeGazeInference

logger = logging.getLogger(__name__)


class AriaAgentVoxBridge:
    """Bridge class to connect Aria eye tracking with AgentVox"""
    
    def __init__(self, voice_assistant: VoiceAssistant, eye_gaze_inference: EyeGazeInference, 
                 device_calibration=None, resize=896):
        self.voice_assistant = voice_assistant
        self.eye_gaze_inference = eye_gaze_inference
        self.device_calibration = device_calibration
        self.resize = resize
        
        # Latest data
        self.latest_rgb_image = None
        self.latest_eye_image = None
        self.latest_gaze_data = None
        
        # Threading locks
        self.image_lock = threading.Lock()
        self.gaze_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        
        # Audio buffer for streaming data
        self.audio_buffer = []
        self.audio_sample_rate = 48000  # Aria default audio sample rate
        self.audio_enabled = False  # Flag to enable/disable Aria audio
        self.audio_max_buffer_size = self.audio_sample_rate * 5  # 5 seconds max buffer
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Language check
        self.is_korean = voice_assistant.model_config.stt_language.startswith('ko')
    
    def print_bilingual(self, kr_msg: str, en_msg: str):
        """Print message in appropriate language"""
        print(kr_msg if self.is_korean else en_msg)
    
    def clean_response(self, response: str) -> str:
        """Clean response by removing markdown formatting"""
        return response.replace("*", "").replace("--", "").strip()
    
    def _convert_to_pil_image(self, cv_image: np.array) -> Image.Image:
        """Convert OpenCV image to PIL Image, handling color format"""
        try:
            # Try direct conversion first (Aria might already be RGB)
            return Image.fromarray(cv_image)
        except:
            # Fallback to BGR2RGB conversion
            return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
    def on_image_received(self, image: np.array, record: ImageDataRecord):
        """Handle incoming images from Aria device"""
        camera_id = record.camera_id
        
        if camera_id == aria.CameraId.Rgb:
            # Process RGB image
            # Rotate 90 degrees clockwise (Aria RGB camera orientation)
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
            with self.image_lock:
                # Convert to PIL Image for AgentVox
                self.latest_rgb_image = self._convert_to_pil_image(rotated_image)
                    
        elif camera_id == aria.CameraId.EyeTrack:
            # Process eye tracking image
            self.process_eye_tracking(image, record)
    
    def process_eye_tracking(self, image: np.array, record: ImageDataRecord):
        """Process eye tracking data"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        with self.image_lock:
            self.latest_eye_image = image
        
        # Run eye gaze inference
        try:
            timestamp_ns = record.capture_timestamp_ns
            preds_main, preds_lower, preds_upper = self.eye_gaze_inference.predict(image)
            
            # Convert tensors to numpy if needed
            if hasattr(preds_main, 'cpu'):
                preds_main = preds_main.cpu().detach().numpy().flatten()
            else:
                preds_main = preds_main.flatten()
            
            # Extract gaze angles
            yaw_rads = float(preds_main[0])
            pitch_rads = float(preds_main[1])
            yaw_deg = np.degrees(yaw_rads)
            pitch_deg = np.degrees(pitch_rads)
            
            with self.gaze_lock:
                self.latest_gaze_data = {
                    "timestamp_ns": timestamp_ns,
                    "yaw_deg": yaw_deg,
                    "pitch_deg": pitch_deg,
                    "yaw_rads": yaw_rads,
                    "pitch_rads": pitch_rads
                }
        
        except Exception as e:
            logger.error(f"Error in eye tracking inference: {e}")
            
        # Update FPS
        self.update_fps()
    
    def on_audio_received(self, audio_data, data_record):
        """Handle incoming audio data from Aria device"""
        try:
            # Extract raw audio data from Aria AudioData object
            audio_array = np.array(audio_data.data, dtype=np.float32)
            
            # Add to audio buffer
            with self.audio_lock:
                self.audio_buffer.extend(audio_array.flatten())
                
                # Keep buffer size reasonable (5 seconds max)
                if len(self.audio_buffer) > self.audio_max_buffer_size:
                    self.audio_buffer = self.audio_buffer[-self.audio_max_buffer_size:]
                    
        except Exception as e:
            print(f"Error processing audio data: {e}")
    
    def get_audio_chunk(self, chunk_size):
        """Get a chunk of audio data for processing"""
        with self.audio_lock:
            if len(self.audio_buffer) >= chunk_size:
                chunk = self.audio_buffer[:chunk_size]
                self.audio_buffer = self.audio_buffer[chunk_size:]
                return np.array(chunk, dtype=np.float32)
        return None
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed > 2.0:  # Update every 2 seconds
            fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time
            print(f"Eye tracking FPS: {fps:.1f}")
    
    def capture_current_view(self):
        """Capture current RGB view and add to AgentVox"""
        with self.image_lock:
            if self.latest_rgb_image is not None:
                self.voice_assistant.add_image(self.latest_rgb_image.copy())
                print("📸 Current view captured and added to voice assistant!")
    
    def project_gaze_to_image(self, image_width: int, image_height: int) -> tuple:
        """Project gaze angles to image coordinates - simplified approach from streaming_eye_gaze_demo.py"""
        with self.gaze_lock:
            if not self.latest_gaze_data:
                return None, None
            
            yaw_deg = self.latest_gaze_data["yaw_deg"]
            pitch_deg = self.latest_gaze_data["pitch_deg"]
            
            # Use the simple projection from streaming_eye_gaze_demo.py
            # Map gaze angles to image coordinates
            # Assuming field of view ~90 degrees (divide by 45)
            
            # The image has been rotated 90° clockwise, need another 90° = 180° total
            # After 180° rotation, everything is inverted:
            # - Yaw: inverted (positive becomes negative)
            # - Pitch: inverted (positive becomes negative)
            
            # After 180° rotation:
            gaze_x = int(image_width/2 - (yaw_deg / 45.0) * image_width/2)    # yaw inverted
            gaze_y = int(image_height/2 - (pitch_deg / 45.0) * image_height/2)  # pitch inverted
            
            # Clamp to image bounds
            gaze_x = max(0, min(image_width - 1, int(gaze_x)))
            gaze_y = max(0, min(image_height - 1, int(gaze_y)))
            
            return int(gaze_x), int(gaze_y)
    
    def get_gaze_or_center(self, width: int, height: int) -> tuple:
        """Get gaze coordinates or center if not available"""
        gaze_x, gaze_y = self.project_gaze_to_image(width, height)
        if gaze_x is None or gaze_y is None:
            return width // 2, height // 2
        return gaze_x, gaze_y
    
    def resize_image_with_gaze(self, image: Image.Image, size: int = None) -> Image.Image:
        """Resize image and add gaze point marker"""
        if size is None:
            size = self.resize
        if not image:
            return None
            
        # First add gaze point to the image
        image_with_gaze = self.add_gaze_overlay(image)
        
        # Then resize the image
        if image_with_gaze.size[0] != size or image_with_gaze.size[1] != size:
            # Keep aspect ratio
            image_with_gaze.thumbnail((size, size), Image.Resampling.LANCZOS)
            
        return image_with_gaze
    
    def add_gaze_overlay(self, image: Image.Image) -> Image.Image:
        """Add green dot at gaze point on image"""
        if not image:
            return None
            
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = cv_image.shape[:2]
        
        # Get gaze point coordinates
        gaze_x, gaze_y = self.project_gaze_to_image(width, height)
        
        if gaze_x is not None and gaze_y is not None:
            # Draw large green dot at gaze point
            cv2.circle(cv_image, (gaze_x, gaze_y), 20, (0, 255, 0), -1)  # Large filled green circle
            cv2.circle(cv_image, (gaze_x, gaze_y), 25, (0, 255, 0), 3)   # Outer ring for visibility
        
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    
    def add_gaze_point_to_image(self, image: Image.Image) -> tuple:
        """Add gaze point to image and return image with relative coordinates"""
        if not image:
            return None, None, None
            
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = cv_image.shape[:2]
        
        # Get gaze point coordinates
        gaze_x, gaze_y = self.project_gaze_to_image(width, height)
        
        if gaze_x is None or gaze_y is None:
            # No gaze data available
            return image, None, None
        
        # Draw gaze point (bright green circle)
        cv2.circle(cv_image, (gaze_x, gaze_y), 12, (0, 255, 0), 3)  # Green circle
        cv2.circle(cv_image, (gaze_x, gaze_y), 4, (0, 255, 0), -1)  # Filled center
        
        # Add crosshair for better visibility
        cv2.line(cv_image, (gaze_x - 20, gaze_y), (gaze_x + 20, gaze_y), (0, 255, 0), 2)
        cv2.line(cv_image, (gaze_x, gaze_y - 20), (gaze_x, gaze_y + 20), (0, 255, 0), 2)
        
        # Calculate relative coordinates (0.0 to 1.0)
        rel_x = gaze_x / width
        rel_y = gaze_y / height
        
        # Convert back to PIL
        annotated_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        return annotated_image, rel_x, rel_y
    
    def get_current_view_with_gaze(self) -> tuple:
        """Get current RGB image with gaze point overlay and coordinates"""
        with self.image_lock:
            if not self.latest_rgb_image:
                return None, None, None
            
            # Create annotated image with gaze point
            annotated_image, rel_x, rel_y = self.add_gaze_point_to_image(self.latest_rgb_image.copy())
            
            return annotated_image, rel_x, rel_y
    
    def get_full_image_with_gaze_overlay(self) -> Image.Image:
        """Get full RGB image with large green dot at gaze point for debug"""
        with self.image_lock:
            if not self.latest_rgb_image:
                return None
            
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(self.latest_rgb_image), cv2.COLOR_RGB2BGR)
            height, width = cv_image.shape[:2]
            
            # Get gaze point coordinates
            gaze_x, gaze_y = self.project_gaze_to_image(width, height)
            
            if gaze_x is not None and gaze_y is not None:
                # Draw large green dot at gaze point
                cv2.circle(cv_image, (gaze_x, gaze_y), 20, (0, 255, 0), -1)  # Large filled green circle
                cv2.circle(cv_image, (gaze_x, gaze_y), 25, (0, 255, 0), 3)   # Outer ring for visibility
                
                # Add text showing gaze coordinates
                text = f"Gaze: ({gaze_x}, {gaze_y})"
                cv2.putText(cv_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # No gaze data - add warning text
                cv2.putText(cv_image, "No gaze data", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Convert back to PIL
            return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    
    def get_current_resized_view(self) -> Image.Image:
        """Get current RGB image resized with gaze overlay"""
        with self.image_lock:
            if not self.latest_rgb_image:
                return None
            
            # Resize image with gaze overlay
            return self.resize_image_with_gaze(self.latest_rgb_image.copy())
    
    def get_gaze_description(self) -> str:
        """Get a description of current gaze direction"""
        with self.gaze_lock:
            if not self.latest_gaze_data:
                return "No gaze data available"
            
            yaw = self.latest_gaze_data["yaw_deg"]
            pitch = self.latest_gaze_data["pitch_deg"]
            
            # Simple gaze direction description
            h_direction = "center"
            if yaw > 10:
                h_direction = "right"
            elif yaw < -10:
                h_direction = "left"
                
            v_direction = "center"
            if pitch > 10:
                v_direction = "down"
            elif pitch < -10:
                v_direction = "up"
            
            return f"Looking {v_direction}-{h_direction} (yaw: {yaw:.1f}°, pitch: {pitch:.1f}°)"
    
    def get_status(self) -> str:
        """Get current status"""
        with self.image_lock:
            rgb_status = "✓" if self.latest_rgb_image is not None else "✗"
            eye_status = "✓" if self.latest_eye_image is not None else "✗"
        
        with self.gaze_lock:
            gaze_status = "✓" if self.latest_gaze_data else "✗"
        
        return f"RGB: {rgb_status} | Eye: {eye_status} | Gaze: {gaze_status}"



def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Aria Eye Tracking + AgentVox Integration"
    )
    
    # Aria connection parameters
    parser.add_argument(
        "--device-ip",
        type=str,
        help="Aria device IP address (required for WiFi interface)"
    )
    parser.add_argument(
        "--interface",
        type=str,
        choices=["usb", "wifi"],
        default="wifi",
        help="Connection interface (default: wifi)"
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        default="profile18",
        help="Streaming profile name (default: profile18)"
    )
    
    # Eye tracking model parameters
    parser.add_argument(
        "--eye-model-path",
        type=str,
        default="projectaria/projectaria_eyetracking/projectaria_eyetracking/inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth",
        help="Path to eye tracking model weights"
    )
    parser.add_argument(
        "--eye-config-path", 
        type=str,
        default="projectaria/projectaria_eyetracking/projectaria_eyetracking/inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml",
        help="Path to eye tracking model config"
    )
    
    # Image processing parameters
    parser.add_argument(
        "--resize",
        type=int,
        default=896,
        help="Size to resize image (default: 896)"
    )
    
    # AgentVox parameters
    parser.add_argument(
        "--llm-model",
        type=str,
        help="Path to LLM model (default: auto-detect)"
    )
    parser.add_argument(
        "--mmproj-model",
        type=str,
        default="mmproj-gemma-3-12b-it-F16.gguf",
        help="Path to multimodal projection model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device for inference (default: auto)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ko",
        help="Language for voice assistant (default: ko)"
    )
    parser.add_argument(
        "--speaker-wav",
        type=str,
        default=None,
        help="Speaker voice sample file for voice cloning"
    )
    
    # Integration parameters
    parser.add_argument(
        "--update-iptables",
        action="store_true",
        help="Update iptables for streaming (Linux only)"
    )
    parser.add_argument(
        "--use-aria-mic",
        action="store_true",
        help="Use Aria microphone instead of computer microphone (experimental)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode - save captured images with gaze overlay"
    )
    
    return parser


def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.interface == "wifi" and not args.device_ip:
        parser.error("--device-ip is required when using WiFi interface")
    
    # Update iptables if requested
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()
    
    print("🚀 Starting Aria + AgentVox Integration")
    print("=" * 50)
    
    try:
        # 1. Initialize AgentVox with multimodal support
        print("📢 Initializing AgentVox with multimodal support...")
        
        model_config = ModelConfig(
            llm_model=args.llm_model,
            mmproj_model=args.mmproj_model,
            is_multimodal=True,  # Enable multimodal support
            device=args.device,
            stt_language=args.language,
            speaker_wav=args.speaker_wav
        )
        
        audio_config = AudioConfig()
        voice_assistant = VoiceAssistant(model_config, audio_config)
        
        # 2. Initialize eye tracking model
        print("👁️ Loading eye tracking model...")
        eye_gaze_inference = EyeGazeInference(
            args.eye_model_path,
            args.eye_config_path,
            "cpu"
        )
        
        # 3. Connect to Aria device
        print("📱 Connecting to Aria device...")
        device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        
        # Set IP address only for WiFi connection
        if args.device_ip:
            client_config.ip_v4_address = args.device_ip
        
        device_client.set_client_config(client_config)
        
        device = device_client.connect()
        if not device:
            raise RuntimeError("Failed to connect to Aria device")
        
        connection_info = f"via {args.interface.upper()}"
        if args.device_ip:
            connection_info += f" at {args.device_ip}"
        print(f"✅ Connected to Aria device {connection_info}")
        
        # 4. Set up streaming
        print("📺 Setting up streaming...")
        streaming_manager = device.streaming_manager
        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = args.profile_name
        
        if args.interface == "usb":
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        
        streaming_config.security_options.use_ephemeral_certs = True
        streaming_manager.streaming_config = streaming_config
        
        # Get device calibration
        device_calibration = None
        try:
            sensors_calib_json = streaming_manager.sensors_calibration()
            device_calibration = device_calibration_from_json_string(sensors_calib_json)
            print("✅ Device calibration loaded")
        except Exception as e:
            print(f"⚠️ Could not load device calibration: {e}")
        
        # 5. Create bridge and set up observer
        bridge = AriaAgentVoxBridge(voice_assistant, eye_gaze_inference, device_calibration, resize=args.resize)
        
        streaming_client = streaming_manager.streaming_client
        streaming_client.set_streaming_client_observer(bridge)
        
        # Configure subscription
        config = streaming_client.subscription_config
        # Temporarily disable audio streaming to reduce DDS warnings
        config.subscriber_data_type = (
            aria.StreamingDataType.EyeTrack | aria.StreamingDataType.Rgb
        )
        config.message_queue_size[aria.StreamingDataType.EyeTrack] = 3
        config.message_queue_size[aria.StreamingDataType.Rgb] = 1
        # config.message_queue_size[aria.StreamingDataType.Audio] = 20  # Disabled for now
        
        # Enable security
        options = aria.StreamingSecurityOptions()
        options.use_ephemeral_certs = True
        config.security_options = options
        
        streaming_client.subscription_config = config
        
        # 6. Start streaming
        print("🎬 Starting streaming...")
        streaming_manager.start_streaming()
        streaming_client.subscribe()
        
        # Enable Aria audio if requested
        if args.use_aria_mic:
            print("🎤 Enabling Aria microphone...")
            bridge.audio_enabled = True
            voice_assistant.set_external_audio_source(bridge)
        
        print("\n🎯 Integration ready!")
        print("📋 How it works:")
        print("  ✨ When you speak, the system automatically:")
        print("     1. Captures your current RGB camera view")
        print("     2. Adds a GREEN DOT showing where you're looking")
        print("     3. Analyzes the image with your speech input")
        print("     4. Provides contextual responses based on your gaze")
        print()
        print("📋 Special commands:")
        print("  - Say 'status' to get system status")
        print("  - Say 'clear images' to clear image buffer")
        print("  - Say 'exit' to quit")
        print()
        print("🔍 Example questions:")
        print("  - '이것이 뭐야?' (What is this?) - Analyzes what you're looking at")
        print("  - '여기에 쓰여진 내용을 읽어줘' (Read what's written here)")
        print("  - '이 화면에서 중요한 부분은?' (What's important in this screen?)")
        print("=" * 50)
        
        # Create debug directory if debug mode is enabled
        debug_dir = None
        if args.debug:
            debug_dir = Path("aria_debug_captures")
            debug_dir.mkdir(exist_ok=True)
            print(f"📁 Debug mode enabled - saving captures to: {debug_dir}")
        
        # 7. Enhanced conversation loop with Aria integration
        def enhanced_conversation_loop():
            """Enhanced conversation loop with Aria integration"""
            is_korean = voice_assistant.model_config.stt_language.startswith('ko')
            
            # Variable to store captured image
            captured_image = None
            capture_count = 0
            
            # Callback for when recording starts
            def on_recording_start():
                nonlocal captured_image, capture_count
                bridge.print_bilingual("🎤 녹음 시작 - 시선 위치 캡처 중...", "🎤 Recording started - capturing gaze position...")
                # Capture resized image WITH green dot for AI
                captured_image = bridge.get_current_resized_view()
                
                if args.debug and captured_image:
                    # Save debug image
                    capture_count += 1
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    
                    # Save resized image with gaze
                    filename = debug_dir / f"capture_{capture_count:04d}_{timestamp}.png"
                    captured_image.save(filename)
                    print(f"💾 Debug: Saved capture to {filename}")
                
                if captured_image:
                    bridge.print_bilingual(f"📷 {bridge.resize}x{bridge.resize} 이미지 캡처 완료 (시선 표시 포함)", 
                                         f"📷 Captured {bridge.resize}x{bridge.resize} image (with gaze marker)")
            
            # Set the callback
            voice_assistant.stt.recorder.on_recording_start = on_recording_start
            
            while True:
                # Reset captured image
                captured_image = None
                
                # Listen to user
                user_input = voice_assistant.stt.transcribe_once()
                
                if not user_input:
                    continue
                
                user_lower = user_input.lower()
                
                # Check for special commands
                if "exit" in user_lower or "종료" in user_input:
                    bridge.print_bilingual("\n대화를 종료합니다.", "\nEnding conversation.")
                    break
                elif "status" in user_lower or "상태" in user_input:
                    status = bridge.get_status()
                    print(f"📊 System Status: {status}")
                    continue
                elif "clear images" in user_lower or "이미지 클리어" in user_input:
                    voice_assistant.clear_images()
                    continue
                
                # Use the image captured when recording started
                cropped_image = captured_image
                
                # Clear previous images
                voice_assistant.clear_images()
                
                if cropped_image:
                    # Add cropped image to voice assistant
                    voice_assistant.add_image(cropped_image)
                    
                    # Use original input without coordinates
                    enhanced_input = user_input
                    
                    bridge.print_bilingual(
                        f"📷 시선 표시가 포함된 {bridge.resize}x{bridge.resize} 이미지로 응답 생성 중...",
                        f"📷 Generating response with {bridge.resize}x{bridge.resize} image with gaze marker..."
                    )
                    
                    # Generate response with the reloaded image and gaze coordinates
                    response = voice_assistant.llm.generate_response(enhanced_input, images=voice_assistant.image_buffer)
                    response = bridge.clean_response(response)
                else:
                    # No image or gaze data available, proceed with text only
                    bridge.print_bilingual(
                        "⚠️ 이미지나 시선 데이터가 없어 텍스트만으로 응답합니다.",
                        "⚠️ No image or gaze data available, proceeding with text-only response."
                    )
                    
                    response = voice_assistant.llm.generate_response(user_input)
                    response = bridge.clean_response(response)
                bridge.print_bilingual(f"\n어시스턴트: {response}", f"\nAssistant: {response}")
                
                # Clear images after use
                voice_assistant.clear_images()
                
                # Speak response
                voice_assistant.tts.speak(response)
        
        # Run the enhanced conversation loop
        enhanced_conversation_loop()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            streaming_client.unsubscribe()
            streaming_manager.stop_streaming()
            device_client.disconnect(device)
            print("✅ Cleanup completed")
        except:
            pass


if __name__ == "__main__":
    main()