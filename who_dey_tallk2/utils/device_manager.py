#!/usr/bin/env python3
"""
Device Manager for Who Dey Tallk v2

Handles listing, selecting and initializing audio and video devices.
"""

import cv2
import sounddevice as sd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger('device_manager')

class Colors:
    """Terminal colors for output formatting"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class DeviceManager:
    """Manages audio and video devices for the application"""
    
    @staticmethod
    def list_audio_devices():
        """
        List available audio input devices
        
        Returns:
            List of audio device information dictionaries
        """
        print(f"{Colors.HEADER}Available audio input devices:{Colors.ENDC}")
        print("-" * 60)
        
        devices = sd.query_devices()
        audio_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"ID: {i}, {Colors.BOLD}Name: {device['name']}{Colors.ENDC}, "
                      f"Channels: {device['max_input_channels']}, "
                      f"Sample Rate: {device['default_samplerate']}")
                audio_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        
        print("-" * 60)
        return audio_devices
    
    @staticmethod
    def list_video_devices(max_devices=10):
        """
        List available video input devices
        
        Args:
            max_devices: Maximum number of devices to check
            
        Returns:
            List of available video device IDs
        """
        print(f"{Colors.HEADER}Available video input devices:{Colors.ENDC}")
        print("-" * 60)
        
        video_devices = []
        
        for i in range(max_devices):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera information
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"ID: {i}, {Colors.BOLD}Camera {i}{Colors.ENDC}, "
                      f"Resolution: {width}x{height}, FPS: {fps:.1f}")
                
                video_devices.append({
                    'id': i,
                    'name': f"Camera {i}",
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                
                # Release the camera
                cap.release()
        
        print("-" * 60)
        return video_devices
    
    @staticmethod
    def list_devices():
        """List both audio and video devices"""
        DeviceManager.list_audio_devices()
        print()
        DeviceManager.list_video_devices()
    
    @staticmethod
    def initialize_video_device(device_id, width=None, height=None, fps=None):
        """
        Initialize a video capture device with specified parameters
        
        Args:
            device_id: Video device ID
            width: Desired capture width
            height: Desired capture height
            fps: Desired capture framerate
            
        Returns:
            OpenCV VideoCapture object or None if initialization fails
        """
        try:
            cap = cv2.VideoCapture(device_id)
            
            if not cap.isOpened():
                logger.error(f"Failed to open video device {device_id}")
                return None
            
            # Set properties if specified
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            if fps:
                cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Log actual values
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video device {device_id} initialized: "
                        f"{actual_width}x{actual_height} @ {actual_fps:.1f} fps")
            
            return cap
            
        except Exception as e:
            logger.error(f"Error initializing video device {device_id}: {e}")
            return None
    
    @staticmethod
    def test_audio_device(device_id, duration=3, sample_rate=16000):
        """
        Test audio input device by recording for a few seconds
        
        Args:
            device_id: Audio device ID
            duration: Recording duration in seconds
            sample_rate: Recording sample rate
            
        Returns:
            Average volume level (0-100) or None if failed
        """
        try:
            print(f"Testing audio device {device_id} for {duration} seconds...")
            
            # Record audio
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.int16,
                device=device_id
            )
            
            # Wait for recording to complete
            sd.wait()
            
            # Calculate volume level
            if audio_data.size > 0:
                audio_data = audio_data.flatten()
                max_val = np.abs(audio_data).max()
                normalized_level = min(100, int((max_val / 32768) * 100))
                
                # Print volume meter
                meter_length = 40
                meter = "#" * int(normalized_level * meter_length / 100)
                print(f"Volume level: [{meter.ljust(meter_length)}] {normalized_level}%")
                
                return normalized_level
            
            return None
            
        except Exception as e:
            logger.error(f"Error testing audio device {device_id}: {e}")
            return None


if __name__ == "__main__":
    # Test functionality when script is run directly
    print("Testing Device Manager...")
    DeviceManager.list_devices()
    
    # Get first video device and test it
    video_devices = DeviceManager.list_video_devices()
    if video_devices:
        device_id = video_devices[0]['id']
        cap = DeviceManager.initialize_video_device(device_id)
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Successfully captured frame from video device {device_id}")
                # Display the frame for 2 seconds
                cv2.imshow('Test Frame', frame)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
            cap.release()
    
    # Test first audio device
    audio_devices = DeviceManager.list_audio_devices()
    if audio_devices:
        device_id = audio_devices[0]['id']
        level = DeviceManager.test_audio_device(device_id)
        if level is not None:
            print(f"Audio device {device_id} tested successfully")