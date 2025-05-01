#!/usr/bin/env python3
"""
Video Capture module for Who Dey Tallk 2

Handles video capture from camera devices
"""
import os
import time
import logging
import threading
import numpy as np
from pathlib import Path

# Import OpenCV if available
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not found, video capture will be disabled")

logger = logging.getLogger(__name__)

class VideoCapture:
    """
    Video capture class for Who Dey Tallk system
    
    Captures frames from a camera device and provides them for processing
    """
    
    def __init__(self, device_id=0, width=640, height=480, fps=30, buffer_size=5):
        """
        Initialize the video capture
        
        Args:
            device_id: ID of the video device (camera)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second
            buffer_size: Size of the frame buffer (number of frames to keep)
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        # Internal state
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.last_frame_time = 0
        self.current_fps = 0
        
        # Frame buffer (stores recent frames)
        self.frame_buffer = []
        self.buffer_lock = threading.RLock()
        
        # Initialize video capture if OpenCV is available
        if OPENCV_AVAILABLE:
            try:
                self._init_capture()
            except Exception as e:
                logger.error(f"Error initializing video capture: {e}")
                self.cap = None
        else:
            logger.warning("OpenCV not available, video capture disabled")
            self.cap = None
    
    def _init_capture(self):
        """Initialize the OpenCV video capture"""
        if not OPENCV_AVAILABLE:
            return False
        
        try:
            # If we already have a capture object, release it first
            if self.cap is not None:
                self.cap.release()
            
            # Create video capture object
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video device {self.device_id}")
                self.cap = None
                return False
            
            # Set capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Read actual properties (which may differ from requested)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video capture initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video capture: {e}")
            self.cap = None
            return False
    
    def release(self):
        """Release video capture resources"""
        self.running = False
        
        if self.cap is not None:
            try:
                self.cap.release()
                self.cap = None
                logger.info("Video capture released")
            except Exception as e:
                logger.error(f"Error releasing video capture: {e}")
    
    def capture_frame(self):
        """
        Capture a single frame from the video device
        
        Returns:
            bool: True if frame was captured successfully
        """
        if not OPENCV_AVAILABLE or self.cap is None:
            return False
        
        try:
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                logger.warning("Failed to capture frame")
                return False
            
            # Calculate FPS
            current_time = time.time()
            if self.last_frame_time > 0:
                time_diff = current_time - self.last_frame_time
                if time_diff > 0:
                    instantaneous_fps = 1.0 / time_diff
                    # Smooth FPS calculation with moving average
                    self.current_fps = 0.9 * self.current_fps + 0.1 * instantaneous_fps
            
            self.last_frame_time = current_time
            self.frame_count += 1
            
            # Add frame to buffer
            with self.buffer_lock:
                self.frame_buffer.append(frame)
                
                # Keep buffer size limited
                while len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)
            
            return True
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return False
    
    def get_current_frame(self):
        """
        Get the most recent frame
        
        Returns:
            numpy.ndarray: Frame as a numpy array or None if no frames available
        """
        with self.buffer_lock:
            if not self.frame_buffer:
                return None
            
            # Return the most recent frame (deep copy to avoid modifying buffer)
            return self.frame_buffer[-1].copy()
    
    def get_frame_at_index(self, index):
        """
        Get a frame at a specific index in the buffer
        
        Args:
            index: Index in buffer (0 is oldest, -1 is newest)
            
        Returns:
            numpy.ndarray: Frame as a numpy array or None if index is out of bounds
        """
        with self.buffer_lock:
            if not self.frame_buffer or index >= len(self.frame_buffer) or abs(index) > len(self.frame_buffer):
                return None
            
            return self.frame_buffer[index].copy()
    
    def get_stats(self):
        """
        Get capture statistics
        
        Returns:
            dict: Dictionary of statistics
        """
        return {
            "fps": self.current_fps,
            "frame_count": self.frame_count,
            "buffer_size": len(self.frame_buffer) if hasattr(self, 'frame_buffer') else 0,
            "resolution": (self.width, self.height)
        }
    
    def save_frame(self, frame, output_path):
        """
        Save a frame to disk
        
        Args:
            frame: Frame to save (numpy array)
            output_path: Path to save the frame
            
        Returns:
            bool: True if saved successfully
        """
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV not available, cannot save frame")
            return False
        
        try:
            if frame is None:
                logger.error("Cannot save None frame")
                return False
                
            # Ensure directory exists
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save frame
            cv2.imwrite(output_path, frame)
            logger.debug(f"Saved frame to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return False
    
    @staticmethod
    def list_devices():
        """List available video devices"""
        if not OPENCV_AVAILABLE:
            print("OpenCV not available, cannot list devices")
            return
        
        try:
            # This is system-dependent, but we can try a few methods
            
            # Method 1: Enumerate devices on Linux/Mac
            if os.name == "posix":
                # Check /dev/video* on Linux
                video_devices = list(Path("/dev").glob("video*"))
                if video_devices:
                    print("Available video devices:")
                    for device in sorted(video_devices):
                        device_id = int(device.name.replace("video", ""))
                        print(f"  {device_id}: {device}")
                    return
            
            # Method 2: Try opening cameras sequentially
            print("Scanning for available cameras...")
            for i in range(8):  # Check first 8 camera indices
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"  {i}: Camera {i} available")
                    cap.release()
                else:
                    print(f"  {i}: Not available")
            
        except Exception as e:
            print(f"Error listing video devices: {e}")
    
    def add_overlay_text(self, frame, text, position=(10, 30), font_scale=1.0, 
                         color=(0, 255, 0), thickness=2):
        """
        Add text overlay to a frame
        
        Args:
            frame: Frame to modify
            text: Text to add
            position: Position (x, y) for the text
            font_scale: Font scale
            color: Text color (BGR)
            thickness: Text thickness
            
        Returns:
            numpy.ndarray: Modified frame
        """
        if not OPENCV_AVAILABLE:
            return frame
        
        try:
            # Make a copy to avoid modifying the original
            annotated_frame = frame.copy()
            
            # Add text
            cv2.putText(
                annotated_frame,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness
            )
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error adding text overlay: {e}")
            return frame  # Return original frame if there's an error