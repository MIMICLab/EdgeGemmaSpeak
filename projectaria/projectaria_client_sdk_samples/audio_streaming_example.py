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

"""
Audio Streaming Example for Project Aria glasses
This example demonstrates how to stream and capture audio data from Aria glasses.
"""

import argparse
import sys
import time
import numpy as np
from collections import deque
from threading import Lock

import aria.sdk as aria
from common import update_iptables, quit_keypress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream audio data from Project Aria glasses"
    )
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
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--device-ip", 
        help="IP address to connect to the device over wifi"
    )
    parser.add_argument(
        "--buffer-duration",
        type=float,
        default=10.0,
        help="Duration of audio buffer to maintain in seconds (default: 10.0)"
    )
    return parser.parse_args()


class AudioStreamingObserver:
    """
    Observer class to handle incoming audio data from Aria glasses.
    """
    
    def __init__(self, buffer_duration_sec=10.0):
        self.audio_buffer = deque()
        self.buffer_duration = buffer_duration_sec
        self.sample_rate = None
        self.num_channels = None
        self.audio_lock = Lock()
        self.total_samples = 0
        self.last_timestamp = None
        
    def on_audio_received(self, audio_data: np.ndarray, sample_rate: int, num_channels: int, timestamp_ns: int):
        """
        Callback function that receives audio data from the Aria glasses.
        
        Args:
            audio_data: numpy array of audio samples
            sample_rate: Sample rate of the audio (e.g., 48000 Hz)
            num_channels: Number of audio channels
            timestamp_ns: Timestamp in nanoseconds
        """
        with self.audio_lock:
            # Store audio configuration
            if self.sample_rate is None:
                self.sample_rate = sample_rate
                self.num_channels = num_channels
                print(f"Audio configuration: {sample_rate} Hz, {num_channels} channels")
            
            # Add new audio data to buffer
            self.audio_buffer.append({
                'data': audio_data.copy(),
                'timestamp_ns': timestamp_ns
            })
            
            # Calculate buffer duration in samples
            if self.sample_rate:
                max_samples = int(self.buffer_duration * self.sample_rate)
                
                # Remove old data if buffer exceeds duration
                while self.get_buffer_duration() > self.buffer_duration:
                    self.audio_buffer.popleft()
            
            self.total_samples += len(audio_data)
            self.last_timestamp = timestamp_ns
            
            # Print statistics periodically
            if self.total_samples % (self.sample_rate or 48000) == 0:
                print(f"Received {self.total_samples} audio samples, "
                      f"buffer size: {self.get_buffer_duration():.2f}s")
    
    def get_buffer_duration(self):
        """Calculate the current buffer duration in seconds."""
        if not self.audio_buffer or not self.sample_rate:
            return 0.0
        
        total_samples = sum(len(item['data']) for item in self.audio_buffer)
        return total_samples / self.sample_rate
    
    def get_audio_data(self):
        """Get all buffered audio data as a continuous numpy array."""
        with self.audio_lock:
            if not self.audio_buffer:
                return None
            
            # Concatenate all audio chunks
            audio_chunks = [item['data'] for item in self.audio_buffer]
            return np.concatenate(audio_chunks)


def main():
    args = parse_args()
    
    # Update iptables if needed (Linux only)
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()
    
    # Set SDK log level
    aria.set_log_level(aria.Level.Info)
    
    # 1. Create DeviceClient instance
    device_client = aria.DeviceClient()
    
    # Configure client if IP address is provided
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)
    
    # 2. Connect to the device
    print("Connecting to Aria device...")
    device = device_client.connect()
    print(f"Connected to device: {device.device_info}")
    
    # 3. Get streaming manager and client
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client
    
    # 4. Configure streaming for audio
    streaming_config = aria.StreamingConfig()
    # Note: Audio streaming might require a specific profile that includes audio
    # Common profiles with audio: profile0, profile2, profile9, profile10
    streaming_config.profile_name = "profile10"  # Profile with audio support
    
    # Set interface (USB or WiFi)
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    
    # Use ephemeral certificates
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config
    
    # 5. Configure subscription for audio stream
    print("Configuring audio stream subscription...")
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Audio
    
    # Set message queue size for audio
    config.message_queue_size[aria.StreamingDataType.Audio] = 10
    
    # Set security options
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config
    
    # 6. Create and attach audio observer
    audio_observer = AudioStreamingObserver(buffer_duration_sec=args.buffer_duration)
    
    # Note: The actual observer attachment depends on the SDK version
    # This is a conceptual implementation - the actual API might differ
    class StreamingClientObserver:
        def __init__(self, audio_observer):
            self.audio_observer = audio_observer
            
        def on_audio_received(self, audio_data, audio_config):
            # Extract audio parameters from the config
            sample_rate = audio_config.sample_rate
            num_channels = audio_config.num_channels
            timestamp_ns = audio_config.capture_timestamp_ns
            
            self.audio_observer.on_audio_received(
                audio_data, sample_rate, num_channels, timestamp_ns
            )
    
    observer = StreamingClientObserver(audio_observer)
    streaming_client.set_streaming_client_observer(observer)
    
    # 7. Start streaming
    print("Starting audio streaming...")
    streaming_manager.start_streaming()
    
    # Get streaming state
    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")
    
    # 8. Subscribe to start receiving data
    streaming_client.subscribe()
    
    # 9. Stream audio until user quits
    print("\nStreaming audio data. Press 'q' to quit...")
    print("Audio data is being buffered and statistics will be displayed.")
    
    try:
        while not quit_keypress():
            time.sleep(0.1)
            
            # Optionally: Access buffered audio data here
            # audio_data = audio_observer.get_audio_data()
            # if audio_data is not None:
            #     # Process audio data (e.g., save to file, analyze, etc.)
            #     pass
    
    except KeyboardInterrupt:
        print("\nStopping audio stream...")
    
    # 10. Clean up
    print("Unsubscribing and stopping stream...")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)
    print("Disconnected successfully.")


if __name__ == "__main__":
    main()