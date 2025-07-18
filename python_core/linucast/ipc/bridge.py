"""Bridge for communication between Python and C++ components."""

import logging
import threading
import time
import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

try:
    # Try to import the compiled C++ module
    import linucast_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    linucast_cpp = None

from ..ai.face_detector import DetectedFace


@dataclass
class ProcessingConfig:
    """Configuration for the C++ processing core."""
    enable_face_tracking: bool = True
    enable_background_removal: bool = True
    enable_smoothing: bool = True
    smoothing_factor: float = 0.7
    target_fps: int = 30
    background_mode: str = "blur"  # "blur", "replace", "none"
    background_image_path: str = ""
    
    def to_cpp_config(self):
        """Convert to C++ configuration object."""
        if not CPP_AVAILABLE:
            return None
        
        config = linucast_cpp.ProcessingConfig()
        config.enable_face_tracking = self.enable_face_tracking
        config.enable_background_removal = self.enable_background_removal
        config.enable_smoothing = self.enable_smoothing
        config.smoothing_factor = self.smoothing_factor
        config.target_fps = self.target_fps
        config.background_mode = self.background_mode
        config.background_image_path = self.background_image_path
        return config


class CppBridge:
    """Bridge between Python AI components and C++ processing core."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # C++ core components
        self.cpp_core: Optional[Any] = None
        self.frame_processor: Optional[Any] = None
        self.virtual_camera: Optional[Any] = None
        
        # Processing configuration
        self.processing_config = ProcessingConfig()
        self._load_config_from_dict(config)
        
        # Threading
        self.running = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Status
        self.last_fps = 0.0
        
        # Fallback mode
        self.using_fallback = False
        self.python_processor = None
        self.last_processing_time = 0.0
        
    def _load_config_from_dict(self, config: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        output_config = config.get("output", {})
        background_config = config.get("background", {})
        
        self.processing_config.target_fps = output_config.get("fps", 30)
        self.processing_config.background_mode = background_config.get("mode", "blur")
        self.processing_config.background_image_path = background_config.get("replacement_image", "")
        
    def initialize(self, input_device: str = "/dev/video0", 
                  output_device: str = "/dev/video10") -> bool:
        """Initialize the C++ bridge."""
        try:
            if not CPP_AVAILABLE:
                self.logger.error("C++ module not available. Please compile the C++ backend.")
                return False
            
            self.logger.info("Initializing C++ bridge...")
            
            # Check if we should use output device from config
            output_config = self.config.get("output", {})
            config_device = output_config.get("virtual_device")
            if config_device:
                output_device = config_device
                self.logger.info(f"Using output device from config: {output_device}")
            
            # Check if the device exists and is accessible before trying to use it
            if output_device:
                if not os.path.exists(output_device):
                    self.logger.warning(f"Output device {output_device} does not exist. Falling back to /dev/video10")
                    output_device = "/dev/video10"
                elif not os.access(output_device, os.R_OK | os.W_OK):
                    self.logger.warning(f"Output device {output_device} is not accessible. Check permissions. Falling back to /dev/video10")
                    output_device = "/dev/video10"
            
            # Use output device from config if not explicitly provided
            if output_device is None:
                output_config = self.config.get("output", {})
                output_device = output_config.get("virtual_device", "/dev/video10")
            
            # Create C++ core
            self.cpp_core = linucast_cpp.LinucastCore()
            
            # Convert configuration
            cpp_config = self.processing_config.to_cpp_config()
            if cpp_config is None:
                self.logger.error("Failed to create C++ configuration")
                return False
            
            # Try to initialize C++ core with both input and output devices
            self.logger.info(f"Initializing C++ core with input={input_device}, output={output_device}")
            success = False
            
            try:
                success = self.cpp_core.initialize(cpp_config, input_device, output_device)
                
                # If initialization failed, try again without the virtual camera output
                if not success:
                    self.logger.warning(f"Failed to initialize C++ core with output device. Trying again without virtual camera.")
                    # Use empty string to disable virtual camera output
                    success = self.cpp_core.initialize(cpp_config, input_device, "")
            except Exception as e:
                self.logger.error(f"Error during C++ core initialization: {e}")
                success = False
            
            if not success:
                self.logger.error("Failed to initialize C++ core, falling back to Python-only mode")
                
                # Initialize Python fallback processor
                from .python_fallback import PythonFallbackProcessor
                self.python_processor = PythonFallbackProcessor(self.config)
                if not self.python_processor.initialize(input_device):
                    self.logger.error("Failed to initialize Python fallback processor")
                    return False
                
                self.logger.info("Python fallback processor initialized successfully")
                self.using_fallback = True
                return True
            
            # Create individual components for direct access
            self.frame_processor = linucast_cpp.FrameProcessor()
            self.frame_processor.initialize(cpp_config)
            
            # Initialize virtual camera with the configured output device only if we have a valid device
            if output_device:
                try:
                    self.virtual_camera = linucast_cpp.VirtualCamera()
                    output_res = linucast_cpp.Size(1280, 720)  # Default resolution
                    self.logger.info(f"Initializing virtual camera on {output_device} at {output_res.width}x{output_res.height} @ {self.processing_config.target_fps}fps")
                    self.virtual_camera.initialize(output_device, output_res, self.processing_config.target_fps)
                except Exception as e:
                    self.logger.error(f"Failed to initialize virtual camera: {e}")
                    self.logger.info("Continuing without virtual camera output")
                    # Continue even if virtual camera failed - it might be a permission issue or missing device
            else:
                self.logger.info("No output device specified, running without virtual camera")
            
            self.logger.info("C++ bridge initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing C++ bridge: {e}")
            return False
    
    def start_processing(self) -> bool:
        """Start the background processing thread."""
        if self.running:
            self.logger.warning("Processing already running")
            return True
            
        try:
            self.running = True
            
            # Use Python fallback if C++ core initialization failed
            if self.using_fallback:
                self.logger.info("Starting Python fallback processor")
                return self.python_processor.start()
            
            # C++ processing
            if not self.cpp_core:
                self.logger.error("C++ core not initialized")
                return False
                
            # Start C++ processing in a separate thread
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.logger.info("Background processing started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting processing: {e}")
            self.running = False
            return False
    
    def stop_processing(self) -> None:
        """Stop the background processing."""
        self.running = False
        
        # Stop Python fallback if using it
        if self.using_fallback and self.python_processor:
            self.python_processor.stop()
            self.logger.info("Python fallback processor stopped")
            return
        
        # Stop C++ processing
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            self.processing_thread = None
        
        if self.cpp_core:
            self.cpp_core.shutdown()
        
        self.logger.info("Background processing stopped")
    
    def _processing_loop(self) -> None:
        """Main processing loop running in C++."""
        try:
            if self.cpp_core:
                self.cpp_core.run()
        except Exception as e:
            self.logger.error(f"Error in processing loop: {e}")
        finally:
            self.running = False
    
    def update_faces(self, faces: List[DetectedFace]) -> None:
        """Update detected faces in the C++ core."""
        if not self.cpp_core or not CPP_AVAILABLE:
            return
        
        try:
            # Convert Python faces to C++ faces
            cpp_faces = []
            for face in faces:
                cpp_face = linucast_cpp.Face()
                
                # Convert bbox
                x, y, w, h = face.bbox
                cpp_face.bbox = linucast_cpp.Rect(x, y, w, h)
                
                # Convert landmarks
                cpp_landmarks = []
                for lm_x, lm_y in face.landmarks:
                    cpp_landmarks.append(linucast_cpp.Point2f(lm_x, lm_y))
                cpp_face.landmarks = cpp_landmarks
                
                cpp_face.confidence = face.confidence
                cpp_face.id = -1  # Will be assigned by tracker
                
                cpp_faces.append(cpp_face)
            
            # Update in C++ core
            self.cpp_core.update_faces_from_python(cpp_faces)
            
        except Exception as e:
            self.logger.error(f"Error updating faces: {e}")
    
    def update_background_mask(self, mask: np.ndarray) -> None:
        """Update background segmentation mask in the C++ core."""
        if not self.cpp_core or not CPP_AVAILABLE:
            return
        
        try:
            # Convert numpy array to format expected by C++
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            
            # Update in C++ core
            self.cpp_core.update_background_mask_from_python(mask)
            
        except Exception as e:
            self.logger.error(f"Error updating background mask: {e}")
    
    def set_background_image(self, image: np.ndarray) -> None:
        """Set background replacement image."""
        if not self.frame_processor or not CPP_AVAILABLE:
            return
        
        try:
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            self.frame_processor.set_background_image(image)
            
        except Exception as e:
            self.logger.error(f"Error setting background image: {e}")
    
    def update_config(self, config: ProcessingConfig) -> None:
        """Update processing configuration."""
        self.processing_config = config
        
        if self.cpp_core and CPP_AVAILABLE:
            try:
                cpp_config = config.to_cpp_config()
                if cpp_config:
                    self.cpp_core.set_config_from_python(cpp_config)
            except Exception as e:
                self.logger.error(f"Error updating config: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        status = {
            "running": self.running,
            "cpp_available": CPP_AVAILABLE,
            "fps": 0.0,
            "processing_time_ms": 0.0
        }
        
        if self.cpp_core and CPP_AVAILABLE:
            try:
                status["running"] = self.cpp_core.is_running()
                status["fps"] = self.cpp_core.get_fps()
            except Exception as e:
                self.logger.error(f"Error getting status: {e}")
        
        return status
    
    def is_running(self) -> bool:
        """Check if processing is running."""
        if self.cpp_core and CPP_AVAILABLE:
            try:
                return self.cpp_core.is_running()
            except Exception:
                pass
        return self.running
    
    def get_fps(self) -> float:
        """Get current FPS."""
        if self.using_fallback and self.python_processor:
            return self.python_processor.fps
            
        if self.cpp_core and CPP_AVAILABLE:
            try:
                return self.cpp_core.get_fps()
            except Exception:
                pass
        return 0.0
        
    def get_latest_frame(self) -> Tuple[bool, np.ndarray]:
        """Get the latest processed frame."""
        # Use Python fallback processor if C++ failed
        if self.using_fallback and self.python_processor:
            return self.python_processor.get_frame()
            
        # Use C++ processing
        if self.cpp_core and CPP_AVAILABLE:
            try:
                success, frame_data = self.cpp_core.get_latest_frame()
                if success and frame_data is not None:
                    return True, frame_data
            except Exception as e:
                self.logger.error(f"Error getting latest frame: {e}")
                
        return False, np.zeros((720, 1280, 3), dtype=np.uint8)
        
    def handle_key_press(self, key_code: int) -> bool:
        """Handle key press for camera control.
        
        In fallback mode, this allows switching between cameras.
        """
        if self.using_fallback and self.python_processor:
            return self.python_processor.handle_key_press(key_code)
        return False


# Fallback implementation when C++ module is not available
class FallbackBridge:
    """Fallback bridge implementation using pure Python."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        self.logger.warning("Using fallback Python-only implementation")
    
    def initialize(self, input_device: str = "/dev/video0", 
                  output_device: str = "/dev/video10") -> bool:
        """Initialize fallback bridge."""
        self.logger.info("Fallback bridge initialized (limited functionality)")
        return True
    
    def start_processing(self) -> bool:
        """Start fallback processing."""
        self.running = True
        self.logger.info("Fallback processing started")
        return True
    
    def stop_processing(self) -> None:
        """Stop fallback processing."""
        self.running = False
        self.logger.info("Fallback processing stopped")
    
    def update_faces(self, faces: List[DetectedFace]) -> None:
        """Update faces (no-op in fallback)."""
        pass
    
    def update_background_mask(self, mask: np.ndarray) -> None:
        """Update background mask (no-op in fallback)."""
        pass
    
    def set_background_image(self, image: np.ndarray) -> None:
        """Set background image (no-op in fallback)."""
        pass
    
    def update_config(self, config: ProcessingConfig) -> None:
        """Update config (no-op in fallback)."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get status."""
        return {
            "running": self.running,
            "cpp_available": False,
            "fps": 0.0,
            "processing_time_ms": 0.0
        }
    
    def is_running(self) -> bool:
        """Check if running."""
        return self.running
    
    def get_fps(self) -> float:
        """Get FPS."""
        return 0.0


# Factory function to create appropriate bridge
def create_bridge(config: Dict[str, Any]) -> CppBridge:
    """Create appropriate bridge based on availability."""
    if CPP_AVAILABLE:
        return CppBridge(config)
    else:
        return FallbackBridge(config)
