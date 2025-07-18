#!/usr/bin/env python3
"""
Linucast Simplified - Camera app with face tracking and background effects.

This implementation is based on MediaPipe and OpenCV, providing:
1. Face tracking
2. Background blur/removal
3. Virtual camera output support
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LinucastSimplified:
    """Simple camera app with face tracking and background effects."""
    
    def __init__(self, 
                camera_index=1, 
                blur_strength=55,
                segmentation_threshold=0.6,
                show_landmarks=False,
                output_resolution=(640, 480),
                virtual_cam=False,
                virtual_cam_device="/dev/video10",
                face_tracking=False,
                zoom_ratio=1.8,
                smoothing_factor=0.2,
                fps_target=30):
        """Initialize the LinucastSimplified app.
        
        Args:
            camera_index: Index of the camera to use
            blur_strength: Strength of background blur (higher = stronger blur)
            segmentation_threshold: Threshold for background segmentation (0.0-1.0)
            show_landmarks: Whether to show face landmarks
            output_resolution: Resolution of output video (width, height)
            virtual_cam: Whether to output to a virtual camera
            virtual_cam_device: Path to virtual camera device
            face_tracking: Whether to enable face tracking and auto-framing
            zoom_ratio: Zoom ratio for face tracking (higher = closer zoom)
            smoothing_factor: Smoothing factor for face tracking (higher = faster tracking)
            fps_target: Target FPS (30 or 60)
        """
        self.camera_index = camera_index
        self.blur_strength = blur_strength
        self.segmentation_threshold = segmentation_threshold
        self.show_landmarks = show_landmarks
        self.output_resolution = output_resolution
        self.use_virtual_cam = virtual_cam
        self.virtual_cam_device = virtual_cam_device
        self.face_tracking = face_tracking
        self.zoom_ratio = zoom_ratio
        self.ema_alpha = smoothing_factor
        self.fps_target = fps_target
        self.high_fps_mode = (fps_target == 60)
        
        # Face tracking variables
        self.frame_w, self.frame_h = output_resolution
        self.smoothed_cx = self.frame_w // 2
        self.smoothed_cy = self.frame_h // 2
        self.last_valid_cx = self.frame_w // 2
        self.last_valid_cy = self.frame_h // 2
        self.frame_lost_count = 0
        
        # Zoom control min/max values
        self.min_zoom_ratio = 1.2
        self.max_zoom_ratio = 3.0
        self.zoom_step = 0.1
        
        # Performance settings
        self.use_lightweight_mode = False  # Will bypass heavy processing when needed
        
        self.virtual_cam = None
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Initialize MediaPipe components
        self.mp_face_landmarker = mp.solutions.face_mesh
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize models with optimized parameters
        self.face_mesh = self.mp_face_landmarker.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            refine_landmarks=False,  # Disable refined landmarks for performance
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3  # Lower tracking confidence to maintain detection
        )
        
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1  # 0=general, 1=landscape optimized
        )
        
        # Available cameras
        self.available_cameras = self._scan_available_cameras()
        if len(self.available_cameras) == 0:
            logger.error("No cameras found!")
        else:
            logger.info(f"Found {len(self.available_cameras)} cameras: {self.available_cameras}")
            if self.camera_index not in self.available_cameras:
                self.camera_index = self.available_cameras[0]
                logger.info(f"Selected camera index {self.camera_index} is not available. Using camera {self.camera_index} instead.")
        
        # Initialize camera
        self.cap = None
        
        # Background image for replacement mode
        self.background_image = None
        self.bg_mode = "blur"  # "blur", "remove", "replace"
    
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
        """Open the camera."""
        if self.cap is not None:
            self.cap.release()
        
        logger.info(f"Opening camera {self.camera_index}")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.output_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.output_resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
        
        logger.info(f"Camera set to {self.output_resolution[0]}x{self.output_resolution[1]} @ {self.fps_target}fps")
        return True
    
    def switch_camera(self):
        """Switch to the next available camera."""
        if len(self.available_cameras) <= 1:
            logger.info("No other cameras available to switch to.")
            return False
        
        # Find the index of current camera in the available_cameras list
        try:
            current_idx = self.available_cameras.index(self.camera_index)
            next_idx = (current_idx + 1) % len(self.available_cameras)
            self.camera_index = self.available_cameras[next_idx]
        except ValueError:
            # Current camera index not in list, use first available
            self.camera_index = self.available_cameras[0]
        
        return self.open_camera()
    
    def set_background_mode(self, mode):
        """Set the background mode.
        
        Args:
            mode: Background mode ("blur", "remove", or "replace")
        """
        if mode not in ["blur", "remove", "replace"]:
            logger.warning(f"Unknown background mode: {mode}")
            return
        
        self.bg_mode = mode
        logger.info(f"Background mode set to: {mode}")
    
    def set_background_image(self, image_path):
        """Set the background image for replacement mode.
        
        Args:
            image_path: Path to background image file
        """
        if not os.path.exists(image_path):
            logger.error(f"Background image not found: {image_path}")
            return False
        
        try:
            bg_img = cv2.imread(image_path)
            if bg_img is None:
                logger.error(f"Failed to load background image: {image_path}")
                return False
            
            # Resize to match output resolution
            self.background_image = cv2.resize(bg_img, self.output_resolution)
            logger.info(f"Background image loaded: {image_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading background image: {e}")
            return False
    
    def adjust_zoom(self, increase=True):
        """Adjust zoom ratio for face tracking.
        
        Args:
            increase: Whether to increase (True) or decrease (False) the zoom
        """
        if increase:
            self.zoom_ratio = min(self.zoom_ratio + self.zoom_step, self.max_zoom_ratio)
            logger.info(f"Zoom increased to {self.zoom_ratio:.1f}x")
        else:
            self.zoom_ratio = max(self.zoom_ratio - self.zoom_step, self.min_zoom_ratio)
            logger.info(f"Zoom decreased to {self.zoom_ratio:.1f}x")
    
    def toggle_fps_mode(self):
        """Toggle between 30fps and 60fps modes."""
        self.high_fps_mode = not self.high_fps_mode
        self.fps_target = 60 if self.high_fps_mode else 30
        
        # Update camera FPS if possible
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
            
        # Reset FPS calculation variables
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0  # Reset FPS measurement
        self.use_lightweight_mode = False  # Reset performance mode
        
        # When switching to high FPS mode, automatically reduce blur strength
        if self.high_fps_mode and self.bg_mode == "blur":
            self.blur_strength = min(self.blur_strength, 25)
            
        logger.info(f"FPS mode set to {self.fps_target}fps")
    
    def _handle_key_press(self, key):
        """Handle key press events.
        
        Returns:
            True if the app should quit, False otherwise
        """
        if key == 27 or key == ord('q'):  # ESC or Q to quit
            return True
        elif key == ord('c'):  # C to switch camera
            self.switch_camera()
        elif key == ord('b'):  # B for blur mode
            self.set_background_mode("blur")
        elif key == ord('r'):  # R for remove mode
            self.set_background_mode("remove")
        elif key == ord('i'):  # I for image background
            self.set_background_mode("replace")
            # Use a default image if none is set
            if self.background_image is None:
                # You could add a file dialog here for selecting an image
                print("Please set a background image first using set_background_image()")
        elif key == ord('l'):  # L to toggle landmarks
            self.show_landmarks = not self.show_landmarks
            logger.info(f"Face landmarks {'enabled' if self.show_landmarks else 'disabled'}")
        elif key == ord('t'):  # T to toggle face tracking
            self.face_tracking = not self.face_tracking
            logger.info(f"Face tracking {'enabled' if self.face_tracking else 'disabled'}")
        elif key == ord('+') or key == ord('='):  # +/= to increase zoom
            if self.face_tracking:
                self.adjust_zoom(increase=True)
        elif key == ord('-') or key == ord('_'):  # -/_ to decrease zoom
            if self.face_tracking:
                self.adjust_zoom(increase=False)
        elif key == ord('f'):  # F to toggle FPS mode
            self.toggle_fps_mode()
        
        return False
    
    def process_frame(self, frame):
        """Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        start_time = time.time()
        
        # Flip for natural mirror view
        frame = cv2.flip(frame, 1)
        
        # Always process at full resolution to avoid shape mismatches
        input_frame = cv2.resize(frame, self.output_resolution)
        processing_frame = input_frame.copy()
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
        
        # Determine if we should do full or lightweight processing
        processing_time_budget = 1.0 / max(15, min(self.fps_target, 30))  # Aim for at least 15 FPS
        
        # ---- Face tracking ----
        # Only do face mesh if we're in face tracking mode or showing landmarks
        if self.face_tracking or self.show_landmarks:
            face_results = self.face_mesh.process(rgb)
        else:
            # Create an empty results object
            class EmptyResults:
                def __init__(self):
                    self.multi_face_landmarks = None
            face_results = EmptyResults()
        
        # ---- Background segmentation ----
        # Skip background processing if we're already running too slow
        current_processing_time = time.time() - start_time
        
        # Initialize condition_3ch
        condition_3ch = None
        
        if current_processing_time > processing_time_budget * 0.5 and self.bg_mode == "blur":
            # Use a simpler blur if we're too slow
            self.use_lightweight_mode = True
            # Create a simple mask that treats everything as foreground
            condition_3ch = np.ones((processing_frame.shape[0], processing_frame.shape[1], 3), dtype=bool)
        else:
            self.use_lightweight_mode = False
            segmentation_results = self.segmentation.process(rgb)
            mask = segmentation_results.segmentation_mask
            # Create the condition mask
            condition = mask > self.segmentation_threshold
            # Convert boolean mask to 3 channels
            condition_3ch = np.stack((condition,) * 3, axis=-1)
        
        # Convert boolean mask to 3 channels
        condition_3ch = np.stack((condition,) * 3, axis=-1)
        
        # ---- Apply background effect based on mode ----
        if self.use_lightweight_mode:
            # Use a faster, lower quality effect when in lightweight mode
            if self.bg_mode == "blur":
                # Fast blur with smaller kernel
                blurred_frame = cv2.blur(input_frame, (15, 15))
                output = blurred_frame
            else:
                output = input_frame
        else:
            # Full quality effects when performance allows
            if self.bg_mode == "blur":
                # Optimize blur - use smaller kernel for better performance
                blur_size = min(self.blur_strength, 25)  # Cap blur size for performance
                if blur_size % 2 == 0:  # Ensure odd kernel size
                    blur_size += 1
                blurred_frame = cv2.GaussianBlur(input_frame, (blur_size, blur_size), 0)
                output = np.where(condition_3ch, input_frame, blurred_frame)
            elif self.bg_mode == "remove":
                # Remove background (replace with black)
                black_bg = np.zeros_like(input_frame)
                output = np.where(condition_3ch, input_frame, black_bg)
            elif self.bg_mode == "replace" and self.background_image is not None:
                # Replace background with image
                bg_resized = cv2.resize(self.background_image, (input_frame.shape[1], input_frame.shape[0]))
                output = np.where(condition_3ch, input_frame, bg_resized)
            else:
                # Default to original frame
                output = input_frame
        
        # ---- Apply face tracking auto-frame if enabled ----
        if self.face_tracking:
            if face_results.multi_face_landmarks:
                self.frame_lost_count = 0  # Reset lost frame counter
                for face_landmarks in face_results.multi_face_landmarks:
                    # Use landmarks 33 (left eye) and 263 (right eye) as face center
                    x1 = int(face_landmarks.landmark[33].x * self.frame_w)
                    x2 = int(face_landmarks.landmark[263].x * self.frame_w)
                    y1 = int(face_landmarks.landmark[33].y * self.frame_h)
                    y2 = int(face_landmarks.landmark[263].y * self.frame_h)
                    
                    face_cx = (x1 + x2) // 2
                    face_cy = (y1 + y2) // 2
                    
                    # Store last valid position
                    self.last_valid_cx = face_cx
                    self.last_valid_cy = face_cy

                    # Adjust smoothing based on FPS - smoother when FPS is low
                    adaptive_alpha = min(self.ema_alpha * 2.0, 0.4) if self.fps < 20 else self.ema_alpha
                    
                    # Apply smoothing (Exponential Moving Average)
                    self.smoothed_cx = int(adaptive_alpha * face_cx + (1 - adaptive_alpha) * self.smoothed_cx)
                    self.smoothed_cy = int(adaptive_alpha * face_cy + (1 - adaptive_alpha) * self.smoothed_cy)
            else:
                # If face detection was lost, increment counter
                self.frame_lost_count += 1
                
                # If lost for too many frames, slow down tracking to avoid jitter
                if self.frame_lost_count > 10:
                    # Use very slow tracking when face is lost
                    recovery_alpha = 0.05
                    
                    # Gently move toward last known position
                    self.smoothed_cx = int(recovery_alpha * self.last_valid_cx + (1 - recovery_alpha) * self.smoothed_cx)
                    self.smoothed_cy = int(recovery_alpha * self.last_valid_cy + (1 - recovery_alpha) * self.smoothed_cy)

            # Calculate zoom window
            crop_w = int(self.frame_w / self.zoom_ratio)
            crop_h = int(self.frame_h / self.zoom_ratio)

            x1 = max(self.smoothed_cx - crop_w // 2, 0)
            y1 = max(self.smoothed_cy - crop_h // 2, 0)
            x2 = min(x1 + crop_w, self.frame_w)
            y2 = min(y1 + crop_h, self.frame_h)

            # Ensure crop stays in bounds
            x1 = max(min(x1, self.frame_w - crop_w), 0)
            y1 = max(min(y1, self.frame_h - crop_h), 0)

            # Crop and resize
            if x2 > x1 and y2 > y1:  # Ensure valid crop dimensions
                cropped = output[y1:y2, x1:x2]
                output = cv2.resize(cropped, (self.frame_w, self.frame_h))
                
        # ---- Draw face landmarks if enabled ----
        if self.show_landmarks and face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=landmarks,
                    connections=self.mp_face_landmarker.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                )
        
        # Calculate processing time for this frame
        processing_time = time.time() - start_time
        
        # Update FPS calculation - use an exponential moving average for smoother updates
        self.frame_count += 1
        current_time = time.time()
        time_diff = current_time - self.last_fps_time
        
        if time_diff >= 0.25:  # Update FPS more frequently (every 250ms)
            instantaneous_fps = self.frame_count / time_diff
            # Smooth FPS using EMA
            if self.fps == 0:  # First calculation
                self.fps = instantaneous_fps
            else:
                # Smooth the FPS calculation (70% previous, 30% new)
                self.fps = 0.7 * self.fps + 0.3 * instantaneous_fps
            
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # Add info text
        tracking_status = "ON" if self.face_tracking else "OFF"
        fps_mode = "60fps" if self.high_fps_mode else "30fps"
        zoom_info = f"Zoom: {self.zoom_ratio:.1f}x" if self.face_tracking else ""
        
        cv2.putText(
            output,
            f"FPS: {self.fps:.1f} | {fps_mode} | Camera: {self.camera_index} | Mode: {self.bg_mode} | Tracking: {tracking_status} {zoom_info}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Add key controls info
        controls_text = "C: Camera | B: Blur | R: Remove | I: Image BG | L: Landmarks | T: Tracking | +/-: Zoom | F: FPS | Q/ESC: Quit"
        cv2.putText(
            output,
            controls_text,
            (10, output.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return output
    
    def setup_virtual_camera(self):
        """Set up virtual camera output."""
        try:
            import pyvirtualcam
            self.virtual_cam = pyvirtualcam.Camera(
                width=self.output_resolution[0], 
                height=self.output_resolution[1], 
                fps=self.fps_target,
                device=self.virtual_cam_device
            )
            logger.info(f"Virtual camera set up successfully: {self.virtual_cam_device} @ {self.fps_target}fps")
            return True
        except ImportError:
            logger.error("pyvirtualcam not installed. Install with: pip install pyvirtualcam")
            return False
        except Exception as e:
            logger.error(f"Failed to set up virtual camera: {e}")
            return False
    
    def run(self):
        """Run the app."""
        # Check if any cameras are available
        if len(self.available_cameras) == 0:
            print("No cameras found. Please connect a camera and try again.")
            print("You might need to run with elevated permissions:")
            print("sudo python3 linucast_simple.py")
            return
        
        # Open the camera
        if not self.open_camera():
            return
        
        # Set up virtual camera if requested
        if self.use_virtual_cam:
            if not self.setup_virtual_camera():
                logger.warning("Continuing without virtual camera output.")
        
        logger.info("Starting Linucast Simplified...")
        logger.info("Controls:")
        logger.info("  C: Switch Camera")
        logger.info("  B: Blur Background")
        logger.info("  R: Remove Background")
        logger.info("  I: Background Image")
        logger.info("  L: Toggle Face Landmarks")
        logger.info("  T: Toggle Face Tracking/Auto-frame")
        logger.info("  +/-: Zoom In/Out (when tracking is enabled)")
        logger.info("  F: Toggle between 30/60 FPS mode")
        logger.info("  Q/ESC: Quit")
        
        # Performance tracking variables
        frame_times = []  # Keep track of recent frame processing times
        max_frame_history = 10  # Number of frames to average
        
        while True:
            frame_start = time.time()
            
            # Get frame from camera
            success, frame = self.cap.read()
            if not success:
                logger.warning("Failed to read frame from camera.")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Track frame processing time
            frame_end = time.time()
            frame_time = frame_end - frame_start
            frame_times.append(frame_time)
            if len(frame_times) > max_frame_history:
                frame_times.pop(0)
            
            # Calculate average frame time for adaptive processing
            avg_frame_time = sum(frame_times) / len(frame_times)
            
            # Send to virtual camera if enabled
            if self.use_virtual_cam and self.virtual_cam:
                # Convert BGR to RGB for pyvirtualcam
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                self.virtual_cam.send(frame_rgb)
                
                # Let virtual camera handle timing
                self.virtual_cam.sleep_until_next_frame()
            
            # Show the frame
            cv2.imshow("Linucast Simplified", processed_frame)
            
            # Adaptive wait time based on processing speed
            target_frame_time = 1.0 / self.fps_target
            remaining_time = target_frame_time - avg_frame_time
            
            # Convert to milliseconds for cv2.waitKey (minimum of 1ms)
            wait_time = max(1, int(remaining_time * 1000))
            
            # Handle key presses with minimal wait time to keep responsive
            key = cv2.waitKey(1) & 0xFF
            if self._handle_key_press(key):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        if self.use_virtual_cam and self.virtual_cam:
            self.virtual_cam.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Linucast Simplified - Camera app with face tracking and background effects")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 1)")
    parser.add_argument("--blur", type=int, default=55, help="Blur strength (default: 55)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Segmentation threshold (default: 0.6)")
    parser.add_argument("--landmarks", action="store_true", help="Show face landmarks")
    parser.add_argument("--resolution", type=str, default="640x480", help="Output resolution (default: 640x480)")
    parser.add_argument("--virtual-cam", action="store_true", help="Output to virtual camera")
    parser.add_argument("--virtual-device", type=str, default="/dev/video10", help="Virtual camera device (default: /dev/video10)")
    parser.add_argument("--bg-image", type=str, help="Background image for replacement mode")
    parser.add_argument("--mode", type=str, default="blur", choices=["blur", "remove", "replace"], help="Background mode (default: blur)")
    parser.add_argument("--face-tracking", action="store_true", help="Enable face tracking and auto-framing")
    parser.add_argument("--zoom-ratio", type=float, default=1.8, help="Zoom ratio for face tracking (default: 1.8)")
    parser.add_argument("--smoothing", type=float, default=0.2, help="Smoothing factor for face tracking (default: 0.2)")
    parser.add_argument("--fps", type=int, default=30, choices=[30, 60], help="Target FPS (30 or 60, default: 30)")
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split("x"))
        resolution = (width, height)
    except ValueError:
        print(f"Invalid resolution format: {args.resolution}. Using default 640x480.")
        resolution = (640, 480)
    
    # Create and run the app
    app = LinucastSimplified(
        camera_index=args.camera,
        blur_strength=args.blur,
        segmentation_threshold=args.threshold,
        show_landmarks=args.landmarks,
        output_resolution=resolution,
        virtual_cam=args.virtual_cam,
        virtual_cam_device=args.virtual_device,
        face_tracking=args.face_tracking,
        zoom_ratio=args.zoom_ratio,
        smoothing_factor=args.smoothing,
        fps_target=args.fps
    )
    
    # Set background mode
    app.set_background_mode(args.mode)
    
    # Set background image if provided
    if args.bg_image:
        app.set_background_image(args.bg_image)
    
    app.run()


if __name__ == "__main__":
    main()
