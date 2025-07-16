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

"""
This sample demonstrates how to apply color correction to RGB streams
from Aria glasses in real-time. It includes various color correction
techniques such as brightness/contrast adjustment, gamma correction,
white balance, and histogram equalization.
"""

import argparse
import sys

import aria.sdk as aria
import cv2
import numpy as np

from common import ctrl_c_handler, quit_keypress, update_iptables
from projectaria_tools.core.sensor_data import ImageDataRecord


class ColorCorrection:
    """Class containing various color correction methods"""
    
    @staticmethod
    def adjust_brightness_contrast(image, brightness=0, contrast=0):
        """
        Adjust brightness and contrast of an image
        brightness: -100 to 100
        contrast: -100 to 100
        """
        brightness = int((brightness - 0) * (255 - (-255)) / (100 - (-100)) + (-255))
        contrast = int((contrast - 0) * (127 - (-127)) / (100 - (-100)) + (-127))
        
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                max_val = 255
            else:
                shadow = 0
                max_val = 255 + brightness
            alpha = (max_val - shadow) / 255
            gamma = shadow
            image = cv2.addWeighted(image, alpha, image, 0, gamma)
        
        if contrast != 0:
            alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            gamma = 127 * (1 - alpha)
            image = cv2.addWeighted(image, alpha, image, 0, gamma)
        
        return image
    
    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        """Apply gamma correction to an image"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    @staticmethod
    def auto_white_balance(image):
        """Apply automatic white balance using the Gray World algorithm"""
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    
    @staticmethod
    def histogram_equalization(image):
        """Apply histogram equalization to improve contrast"""
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Apply histogram equalization to the Y channel
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        # Convert back to BGR
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    @staticmethod
    def clahe_correction(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Split the channels
        l, a, b = cv2.split(lab)
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        # Merge the channels
        lab = cv2.merge([l, a, b])
        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def adjust_saturation(image, saturation_scale=1.0):
        """Adjust color saturation"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        # Scale saturation
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_scale
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        # Convert back to BGR
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    # Color correction parameters
    parser.add_argument(
        "--brightness", type=int, default=0, help="Brightness adjustment (-100 to 100)"
    )
    parser.add_argument(
        "--contrast", type=int, default=0, help="Contrast adjustment (-100 to 100)"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="Gamma correction value"
    )
    parser.add_argument(
        "--saturation", type=float, default=1.0, help="Saturation scale factor"
    )
    parser.add_argument(
        "--auto-white-balance", action="store_true", help="Apply auto white balance"
    )
    parser.add_argument(
        "--histogram-eq", action="store_true", help="Apply histogram equalization"
    )
    parser.add_argument(
        "--clahe", action="store_true", help="Apply CLAHE correction"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    # Set SDK's log level
    aria.set_log_level(aria.Level.Info)

    # Create DeviceClient instance
    device_client = aria.DeviceClient()
    
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    # Connect to the device
    device = device_client.connect()

    # Get streaming manager and client
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    # Configure streaming
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    # Start streaming
    streaming_manager.start_streaming()

    # Configure subscription for RGB stream
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb
    streaming_client.subscription_config = config

    # Create color correction instance
    color_corrector = ColorCorrection()

    # Create observer for streaming data
    class StreamingClientObserver:
        def __init__(self):
            self.rgb_image = None

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.rgb_image = image

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    # Create windows for display
    original_window = "Original RGB"
    corrected_window = "Color Corrected RGB"
    
    cv2.namedWindow(original_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(original_window, 640, 640)
    cv2.moveWindow(original_window, 50, 50)
    
    cv2.namedWindow(corrected_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(corrected_window, 640, 640)
    cv2.moveWindow(corrected_window, 750, 50)

    print("Streaming RGB with color correction. Press 'q' or ESC to quit.")
    print("Color correction settings:")
    print(f"  Brightness: {args.brightness}")
    print(f"  Contrast: {args.contrast}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Saturation: {args.saturation}")
    print(f"  Auto White Balance: {args.auto_white_balance}")
    print(f"  Histogram Equalization: {args.histogram_eq}")
    print(f"  CLAHE: {args.clahe}")

    with ctrl_c_handler() as ctrl_c:
        while not (quit_keypress() or ctrl_c):
            if observer.rgb_image is not None:
                # Convert from BGR to RGB
                rgb_image = cv2.cvtColor(observer.rgb_image, cv2.COLOR_BGR2RGB)
                
                # Apply color corrections
                corrected_image = rgb_image.copy()
                
                # Apply brightness and contrast adjustments
                if args.brightness != 0 or args.contrast != 0:
                    corrected_image = color_corrector.adjust_brightness_contrast(
                        corrected_image, args.brightness, args.contrast
                    )
                
                # Apply gamma correction
                if args.gamma != 1.0:
                    corrected_image = color_corrector.adjust_gamma(
                        corrected_image, args.gamma
                    )
                
                # Apply saturation adjustment
                if args.saturation != 1.0:
                    corrected_image = color_corrector.adjust_saturation(
                        corrected_image, args.saturation
                    )
                
                # Apply auto white balance
                if args.auto_white_balance:
                    corrected_image = color_corrector.auto_white_balance(corrected_image)
                
                # Apply histogram equalization
                if args.histogram_eq:
                    corrected_image = color_corrector.histogram_equalization(corrected_image)
                
                # Apply CLAHE
                if args.clahe:
                    corrected_image = color_corrector.clahe_correction(corrected_image)
                
                # Rotate images for proper display
                original_display = np.rot90(rgb_image, -1)
                corrected_display = np.rot90(corrected_image, -1)
                
                # Display images
                cv2.imshow(original_window, original_display)
                cv2.imshow(corrected_window, corrected_display)
                
                observer.rgb_image = None

    # Cleanup
    print("Stopping streaming...")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()