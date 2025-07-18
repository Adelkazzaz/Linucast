#!/usr/bin/env python3
"""
Simple GUI launcher for Linucast with fallback mode.
Run this script to start Linucast with a simplified GUI that works even without
the C++ backend.
"""

import sys
import os
import logging
import cv2
import yaml
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, Qt

# Import Linucast components
from linucast.ipc.bridge import create_bridge
from linucast.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LinucastSimpleGUI(QMainWindow):
    """Simple GUI for Linucast with fallback mode support."""
    
    def __init__(self):
        super().__init__()
        
        # Load configuration
        self.config = load_config("config_novirt.yaml")
        
        # Initialize bridge
        self.bridge = create_bridge(self.config)
        if not self.bridge.initialize():
            logger.error("Failed to initialize bridge")
            sys.exit(1)
        
        # Start processing
        if not self.bridge.start_processing():
            logger.error("Failed to start processing")
            sys.exit(1)
        
        # Setup UI
        self.init_ui()
        
        # Setup timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS
    
    def init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("Linucast Simple GUI")
        self.setGeometry(100, 100, 800, 600)
        
        # Set application icon
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "assets", "linucast-logo.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.video_label)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Quit button
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        controls_layout.addWidget(self.quit_button)
        
        # FPS display
        self.fps_label = QLabel("FPS: 0.0")
        controls_layout.addWidget(self.fps_label)
        
        # Add controls to main layout
        main_layout.addLayout(controls_layout)
        
        # Create central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def update_frame(self):
        """Update the video frame."""
        success, frame = self.bridge.get_latest_frame()
        
        if success and frame is not None:
            # Convert frame to QImage
            h, w, c = frame.shape
            bytes_per_line = c * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # Scale to fit the label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), 
                                                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Update FPS display
        fps = self.bridge.get_fps()
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.timer.stop()
        self.bridge.stop_processing()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LinucastSimpleGUI()
    window.show()
    sys.exit(app.exec_())
