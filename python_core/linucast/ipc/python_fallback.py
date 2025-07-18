"""
Fallback implementation that uses Python only for video processing.
This module is used when the C++ backend is not available or fails to initialize.
"""

import logging
import cv2
import numpy as np
import threading
import time
from typing import Dict, Any, List, Optional, Tuple

class PythonFallbackProcessor:
    """Pure Python fallback processor when C++ backend is not available."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Python fallback processor."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.camera = None
        self.is_running = False
        self.processing_thread = None
        self.current_frame = None
        self.lock = threading.Lock()
        self.fps = 30
        self.frame_time = 1.0 / self.fps
        
        # Camera switching
        self.available_cameras = []
        self.current_camera_index = 0
        self.last_camera_switch = time.time()
        
    def _scan_available_cameras(self) -> List[int]:
        """Scan for available camera devices."""
        available_cameras = []
        # Try the first 10 camera indexes (common range)
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    self.logger.info(f"Found camera at index {i}")
                    available_cameras.append(i)
                cap.release()
            except Exception:
                pass
        
        # Try common device paths as string indices
        for path in ["/dev/video0", "/dev/video1", "/dev/video2"]:
            if path == str(path) and path not in [str(i) for i in available_cameras]:
                try:
                    cap = cv2.VideoCapture(path)
                    if cap.isOpened():
                        self.logger.info(f"Found camera at path {path}")
                        available_cameras.append(path)
                    cap.release()
                except Exception:
                    pass
                    
        return available_cameras
        
    def initialize(self, input_device: str) -> bool:
        """Initialize the camera and processing."""
        try:
            # Scan for available cameras
            self.available_cameras = self._scan_available_cameras()
            self.logger.info(f"Found {len(self.available_cameras)} cameras: {self.available_cameras}")
            
            # Try to use the specified input device first
            if input_device and input_device in self.available_cameras:
                self.current_camera_index = self.available_cameras.index(input_device)
            elif input_device and input_device.isdigit() and int(input_device.split('video')[-1]) in self.available_cameras:
                self.current_camera_index = self.available_cameras.index(int(input_device.split('video')[-1]))
            elif len(self.available_cameras) > 0:
                # Default to first available camera if specified one isn't available
                self.current_camera_index = 0
            else:
                # No cameras available
                self.logger.warning("No cameras found, using test pattern")
                self.use_test_image = True
                self.test_image = self._create_test_image()
                return True
            
            # Open the selected camera
            device = self.available_cameras[self.current_camera_index]
            self.logger.info(f"Initializing camera {device}")
            self.camera = cv2.VideoCapture(device)
            
            if not self.camera.isOpened():
                self.logger.warning(f"Failed to open camera {device}, using test pattern")
                self.use_test_image = True
                self.test_image = self._create_test_image()
                return True
            
            self.use_test_image = False
                
            # Set camera properties
            camera_config = self.config.get("camera", {})
            resolution = camera_config.get("resolution", [1280, 720])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, camera_config.get("fps", 30))
            
            self.logger.info("Python fallback processor initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing camera: {e}")
            self.use_test_image = True
            self.test_image = self._create_test_image()
            return True  # Return True to continue with test image
    
    def start(self) -> bool:
        """Start the processing thread."""
        if self.is_running:
            return True
            
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.logger.info("Processing thread started")
        return True
        
    def handle_key_press(self, key_code: int) -> bool:
        """Handle key press events for camera control.
        
        Key codes:
        - 99: 'c' key for switching cameras
        - 114: 'r' key for resetting/rescanning cameras
        
        Returns True if the key was handled.
        """
        if key_code == 99:  # 'c' key
            self.logger.info("Switching camera due to key press")
            return self.switch_camera()
        elif key_code == 114:  # 'r' key
            self.logger.info("Rescanning cameras")
            self.available_cameras = self._scan_available_cameras()
            return True
        return False
        
    def stop(self) -> bool:
        """Stop the processing thread."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
            
        self.logger.info("Processing stopped")
        return True
        
    def switch_camera(self) -> bool:
        """Switch to the next available camera."""
        if not self.available_cameras or len(self.available_cameras) <= 1:
            self.logger.info("No other cameras available to switch to")
            return False
            
        with self.lock:
            # Close current camera if it exists
            if self.camera is not None and not self.use_test_image:
                self.camera.release()
                
            # Switch to next camera
            self.current_camera_index = (self.current_camera_index + 1) % len(self.available_cameras)
            device = self.available_cameras[self.current_camera_index]
            
            self.logger.info(f"Switching to camera {device}")
            
            # Open new camera
            self.camera = cv2.VideoCapture(device)
            if not self.camera.isOpened():
                self.logger.warning(f"Failed to open camera {device}")
                return False
                
            # Set camera properties
            camera_config = self.config.get("camera", {})
            resolution = camera_config.get("resolution", [1280, 720])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, camera_config.get("fps", 30))
            
            # Disable test image mode
            self.use_test_image = False
            
            self.last_camera_switch = time.time()
            return True
            
    def _process_frames(self):
        """Process frames from the camera."""
        self.logger.info("Frame processing started")
        
        while self.is_running:
            start_time = time.time()
            
            if hasattr(self, 'use_test_image') and self.use_test_image:
                # In test image mode, we don't need to read from camera
                # The get_frame method will handle returning the test image
                time.sleep(0.03)  # ~30 FPS
                continue
            
            # Capture frame from camera
            if self.camera is None:
                self.logger.warning("Camera is None, sleeping")
                time.sleep(0.1)
                continue
                
            success, frame = self.camera.read()
            if not success:
                self.logger.warning("Failed to read frame, trying to switch cameras")
                if self.switch_camera():
                    continue
                else:
                    # If switching failed, use test pattern
                    self.use_test_image = True
                    self.test_image = self._create_test_image()
                    time.sleep(0.01)
                    continue
                
            # Add camera info overlay
            camera_info = f"Camera {self.current_camera_index + 1}/{len(self.available_cameras)}"
            if isinstance(self.available_cameras[self.current_camera_index], str):
                camera_info += f" ({self.available_cameras[self.current_camera_index]})"
            
            # Simple processing (just add text overlays)
            cv2.putText(
                frame, 
                "Linucast - Python Fallback Mode", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                frame,
                camera_info,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Add hint for camera switching
            cv2.putText(
                frame,
                "Press 'C' to switch cameras",
                (frame.shape[1] - 250, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
                
            # Update current frame
            with self.lock:
                self.current_frame = frame.copy()
                
            # Sleep to maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        self.logger.info("Frame processing stopped")
    
    def _create_test_image(self) -> np.ndarray:
        """Create a test pattern image when no camera is available."""
        # Create a 720p test image
        height, width = 720, 1280
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a colorful gradient background
        for y in range(height):
            for x in range(width):
                image[y, x, 0] = int(255 * (x / width))  # Blue gradient
                image[y, x, 1] = int(255 * (y / height))  # Green gradient
                image[y, x, 2] = int(255 * (1 - (x / width) * (y / height)))  # Red gradient
        
        # Add text
        cv2.putText(
            image,
            "Linucast - Test Pattern", 
            (width // 2 - 200, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3
        )
        
        cv2.putText(
            image,
            "No camera input available", 
            (width // 2 - 180, height // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        return image
        
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """Get the current processed frame."""
        # Check if it's time for an automatic camera switch (every 60 seconds)
        if (not hasattr(self, 'use_test_image') or not self.use_test_image) and \
           time.time() - self.last_camera_switch > 60 and \
           len(self.available_cameras) > 1:
            self.switch_camera()
            
        with self.lock:
            if hasattr(self, 'use_test_image') and self.use_test_image:
                # Return a copy of the test image with a timestamp
                img = self.test_image.copy()
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                cv2.putText(img, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # If no cameras available, show instructions to connect a camera
                if len(self.available_cameras) == 0:
                    cv2.putText(
                        img,
                        "No cameras detected. Please connect a camera.",
                        (img.shape[1]//2 - 220, img.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 200, 0),
                        2
                    )
                
                return True, img
                
            if self.current_frame is None:
                return False, np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Return the current frame with some additional info
            frame = self.current_frame.copy()
            
            # Add FPS info
            fps = 1.0 / self.frame_time if self.frame_time > 0 else 0
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
                
            return True, frame
