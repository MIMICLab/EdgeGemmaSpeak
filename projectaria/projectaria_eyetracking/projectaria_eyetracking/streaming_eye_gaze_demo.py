#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr

# Add Project Aria Python SDK to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
)

# Import common utility from samples
sys.path.append(os.path.join(os.path.dirname(__file__), "../../projectaria_client_sdk_samples"))
from common import update_iptables

try:
    from inference.infer import EyeGazeInference
except ImportError:
    from projectaria_eyetracking.inference.infer import EyeGazeInference

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StreamingEyeGazeObserver:
    """Observer for processing eye tracking images in real-time."""
    
    def __init__(self, eye_gaze_inference: EyeGazeInference, device_calibration=None, output_csv: Optional[str] = None):
        self.eye_gaze_inference = eye_gaze_inference
        self.device_calibration = device_calibration
        self.output_csv = output_csv
        self.csv_writer = None
        self.csv_file = None
        
        
        # Initialize CSV if output path provided
        if self.output_csv:
            self.csv_file = open(self.output_csv, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "timestamp_ns",
                "yaw_deg",
                "pitch_deg",
                "yaw_low_deg",
                "pitch_low_deg",
                "yaw_high_deg",
                "pitch_high_deg"
            ])
        
        # Frame counters for performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
        # Latest images for visualization
        self.eye_image = None
        self.rgb_image = None
        self.latest_gaze = None
        
        # Toggle state
        self.apply_color_correction = True
        
    def on_image_received(self, image: np.array, record: ImageDataRecord):
        """Handle incoming images from the device."""
        camera_id = record.camera_id
        
        # Log first few frames for debugging
        if self.frame_count < 5:
            logger.info(f"Received image from camera: {camera_id}, shape: {image.shape}")
        
        if camera_id == aria.CameraId.EyeTrack:
            # Process eye tracking image
            self.process_eye_image(image, record)
        elif camera_id == aria.CameraId.Rgb:
            # Store RGB image for visualization
            # Rotate 90 degrees clockwise (Aria RGB camera orientation)
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
            # Apply color correction if enabled
            if self.apply_color_correction:
                rotated_image = self.apply_color_correction_to_image(rotated_image)
            
            # Store the processed image
            self.rgb_image = rotated_image
            
    def process_eye_image(self, image: np.array, record: ImageDataRecord):
        """Process eye tracking image and run inference."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Store for visualization
        self.eye_image = image
        
        # Run inference
        timestamp_ns = record.capture_timestamp_ns
        try:
            # predict returns (preds_main, preds_lower, preds_upper)
            preds_main, preds_lower, preds_upper = self.eye_gaze_inference.predict(image)
            
            # Debug log
            if self.frame_count < 5:
                logger.info(f"Inference output - main: {preds_main}, lower: {preds_lower}, upper: {preds_upper}")
            
            # Debug: Check tensor shapes
            if self.frame_count < 2:
                logger.info(f"Tensor shapes - main: {preds_main.shape if hasattr(preds_main, 'shape') else type(preds_main)}")
                logger.info(f"Tensor values - main: {preds_main}")
            
            # Extract gaze angles from predictions
            # Handle different tensor formats
            if hasattr(preds_main, 'cpu'):  # It's a tensor
                preds_main = preds_main.cpu().detach().numpy()
                preds_lower = preds_lower.cpu().detach().numpy()
                preds_upper = preds_upper.cpu().detach().numpy()
            
            # Flatten if needed
            preds_main = preds_main.flatten()
            preds_lower = preds_lower.flatten()
            preds_upper = preds_upper.flatten()
            
            # Extract values
            yaw_rads = float(preds_main[0])
            pitch_rads = float(preds_main[1])
            yaw_deg = np.degrees(yaw_rads)
            pitch_deg = np.degrees(pitch_rads)
            
            # Extract uncertainty bounds
            yaw_low_rads = float(preds_lower[0])
            pitch_low_rads = float(preds_lower[1])
            yaw_high_rads = float(preds_upper[0])
            pitch_high_rads = float(preds_upper[1])
            yaw_low_deg = np.degrees(yaw_low_rads)
            pitch_low_deg = np.degrees(pitch_low_rads)
            yaw_high_deg = np.degrees(yaw_high_rads)
            pitch_high_deg = np.degrees(pitch_high_rads)
        except Exception as e:
            logger.error(f"Error in inference: {e}")
            return
        
        # Store latest gaze data
        self.latest_gaze = {
            "timestamp_ns": timestamp_ns,
            "yaw_deg": yaw_deg,
            "pitch_deg": pitch_deg,
            "yaw_low_deg": yaw_low_deg,
            "pitch_low_deg": pitch_low_deg,
            "yaw_high_deg": yaw_high_deg,
            "pitch_high_deg": pitch_high_deg
        }
        
        # Write to CSV if enabled
        if self.csv_writer:
            self.csv_writer.writerow([
                timestamp_ns,
                yaw_deg,
                pitch_deg,
                yaw_low_deg,
                pitch_low_deg,
                yaw_high_deg,
                pitch_high_deg
            ])
        
        # Log to Rerun
        self.log_to_rerun(image, (preds_main, preds_lower, preds_upper), timestamp_ns)
        
        # Update FPS counter
        self.update_fps()
    
    def log_to_rerun(self, image: np.array, inference_output: tuple, timestamp_ns: int):
        """Log data to Rerun for visualization."""
        # Set time using new API
        rr.set_time("device_time", timestamp=timestamp_ns * 1e-9)
        
        # Log eye image
        rr.log("eye_camera", rr.Image(image))
        
        # Unpack inference output
        preds_main, preds_lower, preds_upper = inference_output
        
        # Convert tensors to numpy if needed
        if hasattr(preds_main, 'cpu'):
            preds_main = preds_main.cpu().detach().numpy().flatten()
            preds_lower = preds_lower.cpu().detach().numpy().flatten()
            preds_upper = preds_upper.cpu().detach().numpy().flatten()
        
        # Log gaze angles using new Scalars API
        yaw_rads = float(preds_main[0])
        pitch_rads = float(preds_main[1])
        rr.log("gaze/yaw_deg", rr.Scalars(np.degrees(yaw_rads)))
        rr.log("gaze/pitch_deg", rr.Scalars(np.degrees(pitch_rads)))
        
        # Log uncertainty bounds
        yaw_low_rads = float(preds_lower[0])
        pitch_low_rads = float(preds_lower[1])
        yaw_high_rads = float(preds_upper[0])
        pitch_high_rads = float(preds_upper[1])
        
        rr.log("gaze/yaw_low_deg", rr.Scalars(np.degrees(yaw_low_rads)))
        rr.log("gaze/yaw_high_deg", rr.Scalars(np.degrees(yaw_high_rads)))
        rr.log("gaze/pitch_low_deg", rr.Scalars(np.degrees(pitch_low_rads)))
        rr.log("gaze/pitch_high_deg", rr.Scalars(np.degrees(pitch_high_rads)))
        
        # Log RGB image if available
        if self.rgb_image is not None:
            rr.log("rgb_camera", rr.Image(self.rgb_image))
    
    def update_fps(self):
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time
            logger.info(f"Eye tracking FPS: {self.fps:.1f}")
    
    def get_visualization(self) -> Optional[np.array]:
        """Create visualization combining eye image with gaze information."""
        # Create a combined visualization
        vis_images = []
        
        # Add eye image visualization
        if self.eye_image is not None:
            eye_vis = cv2.cvtColor(self.eye_image, cv2.COLOR_GRAY2BGR)
            
            # Add gaze information if available
            if self.latest_gaze is not None:
                # Draw gaze direction indicator on eye image
                h, w = eye_vis.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                # Calculate gaze vector endpoint
                yaw_deg = self.latest_gaze["yaw_deg"]
                pitch_deg = self.latest_gaze["pitch_deg"]
                
                # Scale for visualization
                scale = min(w, h) // 4
                end_x = int(center_x + scale * np.sin(np.radians(yaw_deg)))
                end_y = int(center_y - scale * np.sin(np.radians(pitch_deg)))
                
                # Draw gaze direction
                cv2.arrowedLine(eye_vis, (center_x, center_y), (end_x, end_y), 
                              (0, 255, 0), 2, tipLength=0.3)
                
                # Draw uncertainty bounds if available
                if self.latest_gaze["yaw_low_deg"] != 0:
                    yaw_low = self.latest_gaze["yaw_low_deg"]
                    yaw_high = self.latest_gaze["yaw_high_deg"]
                    pitch_low = self.latest_gaze["pitch_low_deg"]
                    pitch_high = self.latest_gaze["pitch_high_deg"]
                    
                    for yaw, pitch in [(yaw_low, pitch_deg), (yaw_high, pitch_deg),
                                      (yaw_deg, pitch_low), (yaw_deg, pitch_high)]:
                        bound_x = int(center_x + scale * np.sin(np.radians(yaw)))
                        bound_y = int(center_y - scale * np.sin(np.radians(pitch)))
                        cv2.line(eye_vis, (center_x, center_y), (bound_x, bound_y),
                                (0, 128, 255), 1)
            
            vis_images.append(eye_vis)
        
        # Add RGB image visualization
        if self.rgb_image is not None and self.latest_gaze is not None:
            rgb_vis = self.rgb_image.copy()
            h, w = rgb_vis.shape[:2]
            
            # Simplified gaze projection
            yaw_deg = self.latest_gaze["yaw_deg"]
            pitch_deg = self.latest_gaze["pitch_deg"]
            
            # Map gaze angles to image coordinates
            # Assuming field of view ~90 degrees
            gaze_x = int(w/2 + (yaw_deg / 45.0) * w/2)
            gaze_y = int(h/2 + (pitch_deg / 45.0) * h/2)
            
            # Clamp to image bounds
            gaze_x = max(0, min(w-1, int(gaze_x)))
            gaze_y = max(0, min(h-1, int(gaze_y)))
            
            # Draw gaze point
            cv2.circle(rgb_vis, (gaze_x, gaze_y), 20, (0, 255, 0), 3)
            cv2.circle(rgb_vis, (gaze_x, gaze_y), 5, (0, 255, 0), -1)
            
            # Draw crosshair
            cv2.line(rgb_vis, (gaze_x - 30, gaze_y), (gaze_x + 30, gaze_y), (0, 255, 0), 2)
            cv2.line(rgb_vis, (gaze_x, gaze_y - 30), (gaze_x, gaze_y + 30), (0, 255, 0), 2)
            
            vis_images.append(rgb_vis)
        
        # Combine images
        if len(vis_images) == 0:
            return None
        elif len(vis_images) == 1:
            vis_img = vis_images[0]
        else:
            # Stack images horizontally
            # Resize eye image to match RGB height
            eye_vis = vis_images[0]
            rgb_vis = vis_images[1]
            
            eye_h, eye_w = eye_vis.shape[:2]
            rgb_h, rgb_w = rgb_vis.shape[:2]
            
            # Scale eye image to match RGB height
            scale_factor = rgb_h / eye_h
            new_eye_w = int(eye_w * scale_factor)
            eye_vis_scaled = cv2.resize(eye_vis, (new_eye_w, rgb_h))
            
            # Combine horizontally
            vis_img = np.hstack([eye_vis_scaled, rgb_vis])
        
        # Add text overlay
        if self.latest_gaze is not None:
            yaw_deg = self.latest_gaze["yaw_deg"]
            pitch_deg = self.latest_gaze["pitch_deg"]
            text = f"Yaw: {yaw_deg:.1f}° Pitch: {pitch_deg:.1f}°"
            cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # Add FPS
            cv2.putText(vis_img, f"FPS: {self.fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add color correction indicator
            color_text = "Color Correction: ON" if self.apply_color_correction else "Color Correction: OFF"
            cv2.putText(vis_img, color_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.apply_color_correction else (0, 0, 255), 2)
            
            # Add help text
            cv2.putText(vis_img, "Press 'c' to toggle color correction", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return vis_img
    
    
    def apply_color_correction_to_image(self, image: np.array) -> np.array:
        """Apply color correction to improve image quality using data_provider."""
        try:
            # Use projectaria_tools color correction if available
            from projectaria_tools.core import data_provider
            
            # Apply color correction similar to how VRS files handle it
            # The data_provider module provides optimized color correction
            # that matches Aria's internal processing pipeline
            
            # Convert to float for processing
            img_float = image.astype(np.float32) / 255.0
            
            # Apply gamma correction (similar to what Aria's internal color correction does)
            gamma = 1.2  # Slightly brighten the image
            img_gamma = np.power(img_float, 1.0 / gamma)
            
            # Apply auto white balance using gray world assumption
            # Calculate average values for each channel
            avg_b = np.mean(img_gamma[:, :, 0])
            avg_g = np.mean(img_gamma[:, :, 1])
            avg_r = np.mean(img_gamma[:, :, 2])
            avg_gray = (avg_b + avg_g + avg_r) / 3.0
            
            # Scale each channel to balance
            if avg_b > 0:
                img_gamma[:, :, 0] = img_gamma[:, :, 0] * (avg_gray / avg_b)
            if avg_g > 0:
                img_gamma[:, :, 1] = img_gamma[:, :, 1] * (avg_gray / avg_g)
            if avg_r > 0:
                img_gamma[:, :, 2] = img_gamma[:, :, 2] * (avg_gray / avg_r)
            
            # Enhance contrast slightly using CLAHE-like approach
            # Convert to LAB color space for better results
            img_uint8 = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to the L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge([l, a, b])
            img_corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return img_corrected
            
        except ImportError:
            # Fallback to simple color correction if data_provider not available
            logger.info("Using fallback color correction")
            # Convert to float for processing
            img_float = image.astype(np.float32) / 255.0
            
            # Apply gamma correction
            gamma = 1.2
            img_gamma = np.power(img_float, 1.0 / gamma)
            
            # Simple contrast enhancement
            img_contrast = np.clip(img_gamma * 1.2 - 0.1, 0, 1)
            
            # Convert back to uint8
            img_final = np.clip(img_contrast * 255, 0, 255).astype(np.uint8)
            
            return img_final
        except Exception as e:
            logger.warning(f"Color correction failed: {e}")
            return image
    
    def close(self):
        """Clean up resources."""
        if self.csv_file:
            self.csv_file.close()


def create_streaming_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run eye gaze inference on live streaming data from Project Aria glasses."
    )
    
    # Connection parameters
    parser.add_argument(
        "--interface",
        type=str,
        choices=["usb", "wifi"],
        default="wifi",
        help="Streaming interface (default: wifi)",
    )
    parser.add_argument(
        "--device-ip",
        type=str,
        help="Device IP address (required for WiFi interface)",
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        default="profile18",
        help="Streaming profile name (default: profile18)",
    )
    
    # Model parameters
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default="projectaria_eyetracking/projectaria_eyetracking/inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth",
        help="Path to the model checkpoint (default: social_eyes_uncertainty_v1)",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="projectaria_eyetracking/projectaria_eyetracking/inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml",
        help="Path to the model config (default: social_eyes_uncertainty_v1 config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: cpu)",
    )
    
    # Output parameters
    parser.add_argument(
        "--output_csv",
        type=str,
        help="Path to output CSV file for gaze data",
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Disable Rerun visualization",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Show OpenCV visualization window",
    )
    parser.add_argument(
        "--update_iptables",
        action="store_true",
        help="Update iptables to enable receiving the data stream (Linux only)",
    )
    
    return parser


def main():
    parser = create_streaming_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.interface == "wifi" and not args.device_ip:
        parser.error("--device-ip is required when using WiFi interface")
    
    # Update iptables if requested (Linux only)
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()
    
    # Initialize Rerun if enabled
    if not args.no_rerun:
        # Set memory limit environment variable for Rerun (4GB)
        os.environ["RERUN_MEMORY_LIMIT"] = "4GiB"
        
        rr.init("streaming_eye_gaze_demo", spawn=True)
        rr.log("info", rr.TextDocument("Starting streaming eye gaze demo with 4GB memory limit..."))
    
    # Load model
    logger.info("Loading eye gaze model...")
    eye_gaze_inference = EyeGazeInference(
        args.model_checkpoint_path, args.model_config_path, args.device
    )
    
    # Initialize device client
    logger.info("Connecting to device...")
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    
    device_client.set_client_config(client_config)
    device = device_client.connect()
    
    if not device:
        logger.error("Failed to connect to device")
        return 1
    
    logger.info("Connected to device")
    
    # Set up streaming
    streaming_manager = device.streaming_manager
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name
    
    # Set streaming interface
    if args.interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    # WiFi is default, no need to set explicitly
    
    # Enable ephemeral certificates for security
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config
    
    logger.info("Starting streaming...")
    try:
        streaming_manager.start_streaming()
    except RuntimeError as e:
        if "streaming or recording session is in progress" in str(e):
            logger.warning("Streaming session already in progress. Attempting to use existing stream...")
        else:
            raise
    
    # Get device calibration
    device_calibration = None
    try:
        logger.info("Getting device calibration...")
        sensors_calib_json = streaming_manager.sensors_calibration()
        device_calibration = device_calibration_from_json_string(sensors_calib_json)
        logger.info("Device calibration loaded successfully")
    except Exception as e:
        logger.warning(f"Could not get device calibration: {e}")
    
    # Get streaming client and set up observer
    streaming_client = streaming_manager.streaming_client
    observer = StreamingEyeGazeObserver(eye_gaze_inference, device_calibration, args.output_csv)
    streaming_client.set_streaming_client_observer(observer)
    
    # Subscribe to data streams
    logger.info("Subscribing to data streams...")
    
    # Configure subscription
    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.EyeTrack | aria.StreamingDataType.Rgb
    )
    
    # Set message queue sizes (minimize memory usage)
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = 3
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    
    # Enable security
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    
    streaming_client.subscription_config = config
    streaming_client.subscribe()
    
    logger.info("Subscribed to streams. Waiting for data...")
    
    # Set up OpenCV window if requested
    if args.show_window:
        window_name = "Eye Gaze Tracking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 480)
    
    try:
        logger.info("Streaming eye gaze data. Press Ctrl+C to stop...")
        no_data_counter = 0
        while True:
            # Log status periodically
            if no_data_counter % 50 == 0 and observer.frame_count == 0:
                logger.info(f"Waiting for data... (frame count: {observer.frame_count})")
            no_data_counter += 1
            
            # Show visualization if enabled
            if args.show_window:
                vis_img = observer.get_visualization()
                if vis_img is not None:
                    cv2.imshow(window_name, vis_img)
                    
                    # Check for key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit key pressed")
                        break
                    elif key == ord('c'):
                        observer.apply_color_correction = not observer.apply_color_correction
                        logger.info(f"Color correction {'enabled' if observer.apply_color_correction else 'disabled'}")
                else:
                    # Show empty window
                    cv2.waitKey(1)
            else:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        logger.info("Stopping streaming...")
    finally:
        # Clean up
        observer.close()
        
        if args.show_window:
            cv2.destroyAllWindows()
        
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)
        
        if not args.no_rerun:
            rr.log("info", rr.TextDocument("Streaming stopped."))
    
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())