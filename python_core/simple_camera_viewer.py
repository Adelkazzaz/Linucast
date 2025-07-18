#!/usr/bin/env python3
"""
Simple console-based camera viewer that works without Qt dependencies.
"""

import sys
import logging
import cv2
import numpy as np
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleCameraViewer:
    """A simple camera viewer class that supports camera switching."""
    
    def __init__(self):
        self.current_camera_index = 0
        self.available_cameras = self._scan_available_cameras()
        self.camera = None
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.open_camera()
        
    def _scan_available_cameras(self):
        """Scan for available camera devices."""
        available_cameras = []
        # Try the first 10 camera indexes (common range)
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    logger.info(f"Found camera at index {i}")
                    available_cameras.append(i)
                cap.release()
            except Exception:
                pass
        
        return available_cameras
    
    def open_camera(self):
        """Open the current camera."""
        if len(self.available_cameras) == 0:
            logger.warning("No cameras found.")
            return False
        
        if self.camera is not None:
            self.camera.release()
        
        index = self.available_cameras[self.current_camera_index]
        logger.info(f"Opening camera {index}")
        self.camera = cv2.VideoCapture(index)
        
        if not self.camera.isOpened():
            logger.error(f"Failed to open camera {index}")
            return False
            
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return True
    
    def switch_camera(self):
        """Switch to the next available camera."""
        if len(self.available_cameras) <= 1:
            logger.info("No other cameras available to switch to.")
            return False
        
        self.current_camera_index = (self.current_camera_index + 1) % len(self.available_cameras)
        return self.open_camera()
    
    def rescan_cameras(self):
        """Rescan available cameras."""
        current_index = self.current_camera_index
        if self.camera:
            self.camera.release()
        
        self.available_cameras = self._scan_available_cameras()
        if len(self.available_cameras) == 0:
            logger.warning("No cameras found during rescan.")
            return False
        
        # Try to keep the same index if possible
        if current_index < len(self.available_cameras):
            self.current_camera_index = current_index
        else:
            self.current_camera_index = 0
        
        return self.open_camera()
    
    def run(self):
        """Run the camera viewer."""
        if len(self.available_cameras) == 0:
            logger.error("No cameras available.")
            print("No cameras were detected. Please connect a camera and try again.")
            print("You can also try running with elevated permissions:")
            print("sudo python3 simple_camera_viewer.py")
            return
        
        logger.info("Starting camera viewer. Press 'c' to switch cameras, 'r' to rescan, 'q' to quit.")
        
        while True:
            # Get frame
            if self.camera is None or not self.camera.isOpened():
                logger.error("Camera not open")
                time.sleep(1)
                continue
            
            success, frame = self.camera.read()
            
            if not success:
                logger.warning("Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Update FPS calculation
            self.frame_count += 1
            current_time = time.time()
            time_diff = current_time - self.last_fps_time
            
            if time_diff >= 1.0:
                self.fps = self.frame_count / time_diff
                self.frame_count = 0
                self.last_fps_time = current_time
            
            # Add camera info and FPS overlay
            camera_info = f"Camera {self.current_camera_index + 1}/{len(self.available_cameras)}"
            fps_info = f"FPS: {self.fps:.1f}"
            
            cv2.putText(
                frame, 
                camera_info, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                frame, 
                fps_info, 
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                frame,
                "Press 'c' to switch cameras, 'r' to rescan, 'q' to quit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Show frame
            cv2.imshow("Camera Viewer", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                logger.info("Switching camera")
                self.switch_camera()
            elif key == ord('r'):
                logger.info("Rescanning cameras")
                self.rescan_cameras()
        
        # Clean up
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    viewer = SimpleCameraViewer()
    viewer.run()
