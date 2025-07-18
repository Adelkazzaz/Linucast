"""Main GUI window for Linucast."""

import sys
import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QComboBox, QSlider, QGroupBox, QFileDialog,
        QCheckBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTabWidget,
        QListWidget, QListWidgetItem, QProgressBar, QStatusBar
    )
    from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize
    from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


class VideoProcessingThread(QThread):
    """Thread for handling video processing and AI inference."""
    
    frame_processed = pyqtSignal(np.ndarray)  # Emitted when a frame is processed
    faces_detected = pyqtSignal(list)  # Emitted when faces are detected
    fps_updated = pyqtSignal(float)  # Emitted when FPS is updated
    
    def __init__(self, face_detector, face_identifier, background_segmenter, cpp_bridge):
        super().__init__()
        self.face_detector = face_detector
        self.face_identifier = face_identifier
        self.background_segmenter = background_segmenter
        self.cpp_bridge = cpp_bridge
        
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Video capture
        self.cap = None
        self.input_device = "/dev/video0"
        
    def set_input_device(self, device: str) -> None:
        """Set input video device."""
        self.input_device = device
        
    def start_capture(self) -> bool:
        """Start video capture."""
        try:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.input_device)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera: {self.input_device}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting capture: {e}")
            return False
    
    def stop_capture(self) -> None:
        """Stop video capture."""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def run(self) -> None:
        """Main processing loop."""
        self.running = True
        
        if not self.start_capture():
            return
        
        frame_count = 0
        import time
        start_time = time.time()
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Process frame with AI
                processed_frame = self.process_frame(frame)
                
                # Emit processed frame
                if processed_frame is not None:
                    self.frame_processed.emit(processed_frame)
                
                # Calculate and emit FPS
                frame_count += 1
                if frame_count % 30 == 0:  # Update FPS every 30 frames
                    current_time = time.time()
                    fps = 30 / (current_time - start_time)
                    self.fps_updated.emit(fps)
                    start_time = current_time
                
                # Small delay to prevent excessive CPU usage
                self.msleep(33)  # ~30 FPS
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                break
        
        self.stop_capture()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with AI."""
        try:
            processed_frame = frame.copy()
            
            # Detect faces
            faces = []
            if self.face_detector:
                faces = self.face_detector.detect_faces(frame)
                self.faces_detected.emit(faces)
                
                # Update C++ bridge with faces
                if self.cpp_bridge:
                    self.cpp_bridge.update_faces(faces)
            
            # Segment background
            if self.background_segmenter:
                mask = self.background_segmenter.segment_background(frame)
                if mask is not None:
                    # Update C++ bridge with mask
                    if self.cpp_bridge:
                        self.cpp_bridge.update_background_mask(mask)
                    
                    # Apply effect for preview (this will be done in C++ for actual output)
                    processed_frame = self.background_segmenter.apply_background_effect(
                        frame, mask, "blur"
                    )
            
            # Draw faces for preview
            if faces:
                if self.face_detector:
                    processed_frame = self.face_detector.draw_faces(processed_frame, faces)
            
            return processed_frame
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame
    
    def stop(self) -> None:
        """Stop the processing thread."""
        self.running = False


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, config: Dict[str, Any], face_detector, face_identifier, 
                 background_segmenter, cpp_bridge):
        super().__init__()
        
        self.config = config
        self.face_detector = face_detector
        self.face_identifier = face_identifier
        self.background_segmenter = background_segmenter
        self.cpp_bridge = cpp_bridge
        
        self.logger = logging.getLogger(__name__)
        
        # Current state
        self.current_faces = []
        self.selected_face_id = -1
        self.background_image = None
        
        # Video processing thread
        self.video_thread = VideoProcessingThread(
            face_detector, face_identifier, background_segmenter, cpp_bridge
        )
        self.video_thread.frame_processed.connect(self.update_preview)
        self.video_thread.faces_detected.connect(self.update_faces_list)
        self.video_thread.fps_updated.connect(self.update_fps)
        
        # Initialize UI
        self.init_ui()
        
        # Start video processing
        self.start_processing()
    
    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Linucast - AI Virtual Camera")
        self.setMinimumSize(1200, 800)
        
        # Set application icon
        icon_path = Path(__file__).parent.parent.parent / "assets" / "linucast-logo.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - video preview and controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 2)
        
        # Right panel - settings and face list
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.fps_label = QLabel("FPS: 0.0")
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.fps_label)
        self.status_bar.addPermanentWidget(self.status_label)
    
    def create_left_panel(self) -> QWidget:
        """Create the left panel with video preview."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Video preview
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("Camera Preview")
        layout.addWidget(self.video_label)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.toggle_camera)
        controls_layout.addWidget(self.start_btn)
        
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setEnabled(False)  # TODO: Implement recording
        controls_layout.addWidget(self.record_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create the right panel with settings."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different settings
        tab_widget = QTabWidget()
        
        # Camera settings tab
        camera_tab = self.create_camera_settings_tab()
        tab_widget.addTab(camera_tab, "Camera")
        
        # Background settings tab
        background_tab = self.create_background_settings_tab()
        tab_widget.addTab(background_tab, "Background")
        
        # Face settings tab
        face_tab = self.create_face_settings_tab()
        tab_widget.addTab(face_tab, "Faces")
        
        # Performance tab
        performance_tab = self.create_performance_tab()
        tab_widget.addTab(performance_tab, "Performance")
        
        layout.addWidget(tab_widget)
        
        return panel
    
    def create_camera_settings_tab(self) -> QWidget:
        """Create camera settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Input device selection
        device_group = QGroupBox("Input Device")
        device_layout = QVBoxLayout(device_group)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["/dev/video0", "/dev/video1", "/dev/video2"])
        device_layout.addWidget(self.device_combo)
        
        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.clicked.connect(self.refresh_devices)
        device_layout.addWidget(refresh_btn)
        
        layout.addWidget(device_group)
        
        # Resolution settings
        resolution_group = QGroupBox("Resolution")
        resolution_layout = QVBoxLayout(resolution_group)
        
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1280x720", "1920x1080", "640x480"])
        resolution_layout.addWidget(self.resolution_combo)
        
        layout.addWidget(resolution_group)
        
        # FPS settings
        fps_group = QGroupBox("Frame Rate")
        fps_layout = QVBoxLayout(fps_group)
        
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(15, 60)
        self.fps_spinbox.setValue(30)
        fps_layout.addWidget(self.fps_spinbox)
        
        layout.addWidget(fps_group)
        
        layout.addStretch()
        return tab
    
    def create_background_settings_tab(self) -> QWidget:
        """Create background settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Background mode
        mode_group = QGroupBox("Background Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.bg_mode_combo = QComboBox()
        self.bg_mode_combo.addItems(["None", "Blur", "Replace"])
        self.bg_mode_combo.currentTextChanged.connect(self.on_background_mode_changed)
        mode_layout.addWidget(self.bg_mode_combo)
        
        layout.addWidget(mode_group)
        
        # Background image selection
        self.bg_image_group = QGroupBox("Background Image")
        bg_image_layout = QVBoxLayout(self.bg_image_group)
        
        self.bg_image_path = QLabel("No image selected")
        bg_image_layout.addWidget(self.bg_image_path)
        
        select_bg_btn = QPushButton("Select Image")
        select_bg_btn.clicked.connect(self.select_background_image)
        bg_image_layout.addWidget(select_bg_btn)
        
        layout.addWidget(self.bg_image_group)
        self.bg_image_group.setEnabled(False)
        
        # Blur settings
        blur_group = QGroupBox("Blur Settings")
        blur_layout = QVBoxLayout(blur_group)
        
        blur_layout.addWidget(QLabel("Blur Strength:"))
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(1, 101)
        self.blur_slider.setValue(51)
        self.blur_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.blur_slider.setTickInterval(25)
        blur_layout.addWidget(self.blur_slider)
        
        layout.addWidget(blur_group)
        
        layout.addStretch()
        return tab
    
    def create_face_settings_tab(self) -> QWidget:
        """Create face settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Face detection settings
        detection_group = QGroupBox("Face Detection")
        detection_layout = QVBoxLayout(detection_group)
        
        self.face_detection_enabled = QCheckBox("Enable Face Detection")
        self.face_detection_enabled.setChecked(True)
        detection_layout.addWidget(self.face_detection_enabled)
        
        detection_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        detection_layout.addWidget(self.confidence_slider)
        
        layout.addWidget(detection_group)
        
        # Detected faces list
        faces_group = QGroupBox("Detected Faces")
        faces_layout = QVBoxLayout(faces_group)
        
        self.faces_list = QListWidget()
        faces_layout.addWidget(self.faces_list)
        
        add_face_btn = QPushButton("Add Selected Face")
        add_face_btn.clicked.connect(self.add_selected_face)
        faces_layout.addWidget(add_face_btn)
        
        layout.addWidget(faces_group)
        
        layout.addStretch()
        return tab
    
    def create_performance_tab(self) -> QWidget:
        """Create performance monitoring tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Performance metrics
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.fps_display = QLabel("FPS: 0.0")
        self.fps_display.setFont(QFont("monospace", 12))
        metrics_layout.addWidget(self.fps_display)
        
        self.processing_time = QLabel("Processing Time: 0.0 ms")
        self.processing_time.setFont(QFont("monospace", 12))
        metrics_layout.addWidget(self.processing_time)
        
        layout.addWidget(metrics_group)
        
        # System info
        system_group = QGroupBox("System Information")
        system_layout = QVBoxLayout(system_group)
        
        self.system_info = QTextEdit()
        self.system_info.setMaximumHeight(200)
        self.system_info.setReadOnly(True)
        self.update_system_info()
        system_layout.addWidget(self.system_info)
        
        layout.addWidget(system_group)
        
        layout.addStretch()
        return tab
    
    def start_processing(self) -> None:
        """Start video processing."""
        try:
            # Set input device
            device = self.device_combo.currentText()
            self.video_thread.set_input_device(device)
            
            # Start C++ processing
            if self.cpp_bridge:
                self.cpp_bridge.start_processing()
            
            # Start video thread
            self.video_thread.start()
            
            self.start_btn.setText("Stop Camera")
            self.status_label.setText("Processing")
            
        except Exception as e:
            self.logger.error(f"Error starting processing: {e}")
            self.status_label.setText("Error starting processing")
    
    def stop_processing(self) -> None:
        """Stop video processing."""
        try:
            # Stop video thread
            self.video_thread.stop()
            self.video_thread.wait()
            
            # Stop C++ processing
            if self.cpp_bridge:
                self.cpp_bridge.stop_processing()
            
            self.start_btn.setText("Start Camera")
            self.status_label.setText("Stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping processing: {e}")
    
    def toggle_camera(self) -> None:
        """Toggle camera on/off."""
        if self.video_thread.running:
            self.stop_processing()
        else:
            self.start_processing()
    
    def update_preview(self, frame: np.ndarray) -> None:
        """Update video preview with processed frame."""
        try:
            # Convert frame to QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Convert to pixmap and display
            pixmap = QPixmap.fromImage(q_image.rgbSwapped())
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.logger.error(f"Error updating preview: {e}")
    
    def update_faces_list(self, faces) -> None:
        """Update the list of detected faces."""
        self.current_faces = faces
        
        # Update faces list widget
        self.faces_list.clear()
        for i, face in enumerate(faces):
            item_text = f"Face {i+1} (Confidence: {face.confidence:.2f})"
            item = QListWidgetItem(item_text)
            self.faces_list.addItem(item)
    
    def update_fps(self, fps: float) -> None:
        """Update FPS display."""
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.fps_display.setText(f"FPS: {fps:.1f}")
    
    def on_background_mode_changed(self, mode: str) -> None:
        """Handle background mode change."""
        self.bg_image_group.setEnabled(mode == "Replace")
        
        # Update C++ bridge configuration
        if self.cpp_bridge:
            from ..ipc.bridge import ProcessingConfig
            config = ProcessingConfig()
            config.background_mode = mode.lower()
            self.cpp_bridge.update_config(config)
    
    def select_background_image(self) -> None:
        """Select background replacement image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Background Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            try:
                # Load image
                image = cv2.imread(file_path)
                if image is not None:
                    self.background_image = image
                    self.bg_image_path.setText(Path(file_path).name)
                    
                    # Update C++ bridge
                    if self.cpp_bridge:
                        self.cpp_bridge.set_background_image(image)
                else:
                    self.logger.error("Failed to load background image")
            except Exception as e:
                self.logger.error(f"Error loading background image: {e}")
    
    def add_selected_face(self) -> None:
        """Add selected face to the database."""
        current_row = self.faces_list.currentRow()
        if current_row >= 0 and current_row < len(self.current_faces):
            face = self.current_faces[current_row]
            
            # TODO: Implement face addition to database
            self.logger.info(f"Adding face {current_row} to database")
    
    def refresh_devices(self) -> None:
        """Refresh available camera devices."""
        # TODO: Implement device enumeration
        self.logger.info("Refreshing camera devices")
    
    def update_system_info(self) -> None:
        """Update system information display."""
        info = []
        info.append("Linucast System Information")
        info.append("=" * 30)
        
        # Check component availability
        info.append(f"C++ Backend: {'Available' if self.cpp_bridge else 'Not Available'}")
        info.append(f"Face Detection: {'Available' if self.face_detector else 'Not Available'}")
        info.append(f"Face Recognition: {'Available' if self.face_identifier else 'Not Available'}")
        info.append(f"Background Segmentation: {'Available' if self.background_segmenter else 'Not Available'}")
        
        # System info
        info.append("")
        info.append("Python: " + sys.version.split()[0])
        
        try:
            import cv2
            info.append("OpenCV: " + cv2.__version__)
        except:
            info.append("OpenCV: Not Available")
        
        try:
            import torch
            info.append("PyTorch: " + torch.__version__)
        except:
            info.append("PyTorch: Not Available")
        
        self.system_info.setPlainText("\n".join(info))
    
    def closeEvent(self, event) -> None:
        """Handle window close event."""
        self.stop_processing()
        event.accept()


# Create main window function
def create_main_window(config: Dict[str, Any], face_detector, face_identifier, 
                      background_segmenter, cpp_bridge) -> Optional[QMainWindow]:
    """Create and return main window if PyQt6 is available."""
    if not PYQT_AVAILABLE:
        print("PyQt6 not available. GUI disabled.")
        return None
    
    return MainWindow(config, face_detector, face_identifier, background_segmenter, cpp_bridge)
