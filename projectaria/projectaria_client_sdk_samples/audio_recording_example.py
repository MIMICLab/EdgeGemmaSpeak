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
Audio Recording Example for Project Aria glasses
This example demonstrates how to stream audio from Aria glasses and save it to a WAV file.
"""

import argparse
import sys
import time
import wave
import numpy as np
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from queue import Queue

import aria.sdk as aria
from common import update_iptables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record audio from Project Aria glasses to a WAV file"
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
        "--duration",
        type=int,
        default=10,
        help="Recording duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output WAV file path (default: aria_audio_TIMESTAMP.wav)",
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
        "--profile",
        type=str,
        default="profile10",
        help="Streaming profile to use (default: profile10)",
    )
    return parser.parse_args()


class AudioRecorder:
    """
    Class to handle audio recording from Aria glasses.
    """
    
    def __init__(self, output_path=None):
        self.audio_queue = Queue()
        self.sample_rate = None
        self.num_channels = None
        self.recording = False
        self.lock = Lock()
        self.total_frames = 0
        self.output_path = output_path or self._generate_filename()
        self.wav_file = None
        self.writer_thread = None
        
    def _generate_filename(self):
        """Generate a filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"aria_audio_{timestamp}.wav"
    
    def start_recording(self):
        """Start recording audio to file."""
        with self.lock:
            if self.recording:
                return
            
            self.recording = True
            self.total_frames = 0
            
            # Start writer thread
            self.writer_thread = Thread(target=self._write_audio_thread)
            self.writer_thread.start()
            
            print(f"Started recording to: {self.output_path}")
    
    def stop_recording(self):
        """Stop recording and close the file."""
        with self.lock:
            if not self.recording:
                return
            
            self.recording = False
        
        # Signal writer thread to stop
        self.audio_queue.put(None)
        
        # Wait for writer thread to finish
        if self.writer_thread:
            self.writer_thread.join()
        
        print(f"Recording saved to: {self.output_path}")
        print(f"Total frames recorded: {self.total_frames}")
        if self.sample_rate:
            duration = self.total_frames / self.sample_rate
            print(f"Duration: {duration:.2f} seconds")
    
    def _write_audio_thread(self):
        """Thread function to write audio data to WAV file."""
        try:
            while True:
                # Get audio data from queue
                item = self.audio_queue.get()
                
                # Check for stop signal
                if item is None:
                    break
                
                audio_data, sample_rate, num_channels = item
                
                # Initialize WAV file on first audio chunk
                if self.wav_file is None:
                    self.wav_file = wave.open(self.output_path, 'wb')
                    self.wav_file.setnchannels(num_channels)
                    self.wav_file.setsampwidth(2)  # 16-bit audio
                    self.wav_file.setframerate(sample_rate)
                    print(f"WAV file initialized: {sample_rate} Hz, {num_channels} channels")
                
                # Convert float audio to 16-bit PCM
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    # Normalize to [-1, 1] range if needed
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    # Convert to 16-bit PCM
                    audio_data = (audio_data * 32767).astype(np.int16)
                
                # Write to WAV file
                self.wav_file.writeframes(audio_data.tobytes())
                self.total_frames += len(audio_data)
                
        finally:
            # Close WAV file
            if self.wav_file:
                self.wav_file.close()
    
    def on_audio_received(self, audio_data: np.ndarray, sample_rate: int, 
                         num_channels: int, timestamp_ns: int):
        """
        Callback function that receives audio data from the Aria glasses.
        """
        with self.lock:
            if not self.recording:
                return
            
            # Store audio configuration
            if self.sample_rate is None:
                self.sample_rate = sample_rate
                self.num_channels = num_channels
            
            # Add audio data to queue for writing
            self.audio_queue.put((audio_data.copy(), sample_rate, num_channels))


class AriaAudioStreamingObserver:
    """
    Observer for Aria streaming client that handles audio data.
    """
    
    def __init__(self, audio_recorder):
        self.audio_recorder = audio_recorder
        self.audio_info_printed = False
    
    def on_audio_received(self, audio_data: np.ndarray, audio_config):
        """
        Handle received audio data from Aria glasses.
        
        Note: The actual callback signature depends on the SDK implementation.
        This is a conceptual example.
        """
        # Extract audio parameters
        sample_rate = getattr(audio_config, 'sample_rate', 48000)
        num_channels = getattr(audio_config, 'num_channels', 1)
        timestamp_ns = getattr(audio_config, 'capture_timestamp_ns', 0)
        
        if not self.audio_info_printed:
            print(f"Receiving audio: {sample_rate} Hz, {num_channels} channel(s)")
            self.audio_info_printed = True
        
        # Forward to recorder
        self.audio_recorder.on_audio_received(
            audio_data, sample_rate, num_channels, timestamp_ns
        )


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
    print(f"Connected to device")
    
    # 3. Get streaming manager and client
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client
    
    # 4. Configure streaming
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile
    
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
    
    # 6. Create audio recorder and observer
    audio_recorder = AudioRecorder(output_path=args.output)
    observer = AriaAudioStreamingObserver(audio_recorder)
    
    # Attach observer to streaming client
    streaming_client.set_streaming_client_observer(observer)
    
    # 7. Start streaming
    print("Starting audio streaming...")
    streaming_manager.start_streaming()
    
    # Get streaming state
    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")
    
    # 8. Subscribe and start recording
    streaming_client.subscribe()
    audio_recorder.start_recording()
    
    # 9. Record for specified duration
    print(f"\nRecording for {args.duration} seconds...")
    print("Press Ctrl+C to stop early.")
    
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    
    # 10. Stop recording and clean up
    print("\nStopping recording...")
    audio_recorder.stop_recording()
    
    print("Cleaning up...")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)
    print("Disconnected successfully.")


if __name__ == "__main__":
    main()