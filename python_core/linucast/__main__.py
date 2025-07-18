#!/usr/bin/env python3
"""Main entry point for Linucast application."""

import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Optional

from .gui.main_window import MainWindow
from .ai.face_detector import FaceDetector
from .ai.face_id import FaceIdentifier
from .ai.segmenter import BackgroundSegmenter
from .ipc.bridge import CppBridge

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
except ImportError:
    print("PyQt6 not found. Please install it: pip install PyQt6")
    sys.exit(1)


def setup_logging(config: dict) -> None:
    """Setup logging configuration."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get("file", "logs/linucast.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    handlers = []
    if log_config.get("console", True):
        handlers.append(logging.StreamHandler())
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        return {}


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Linucast - AI Virtual Camera for Linux"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--nogui",
        action="store_true",
        help="Run without GUI (headless mode)"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="/dev/video0",
        help="Input camera device (default: /dev/video0)"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        default="/dev/video10",
        help="Output virtual camera device (default: /dev/video10)"
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index (default: 0)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Linucast 0.1.0"
    )
    
    return parser


class OpenCamFXApp:
    """Main Linucast application class."""
    
    def __init__(self, config: dict, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI components
        self.face_detector: Optional[FaceDetector] = None
        self.face_identifier: Optional[FaceIdentifier] = None
        self.background_segmenter: Optional[BackgroundSegmenter] = None
        
        # Initialize C++ bridge
        self.cpp_bridge: Optional[CppBridge] = None
        
    def initialize_ai_components(self) -> bool:
        """Initialize AI processing components."""
        try:
            self.logger.info("Initializing AI components...")
            
            # Initialize face detector
            self.face_detector = FaceDetector(self.config.get("ai", {}))
            if not self.face_detector.initialize():
                self.logger.error("Failed to initialize face detector")
                return False
            
            # Initialize face identifier
            self.face_identifier = FaceIdentifier(self.config.get("ai", {}))
            if not self.face_identifier.initialize():
                self.logger.error("Failed to initialize face identifier")
                return False
            
            # Initialize background segmenter
            self.background_segmenter = BackgroundSegmenter(self.config.get("ai", {}))
            if not self.background_segmenter.initialize():
                self.logger.error("Failed to initialize background segmenter")
                return False
            
            self.logger.info("AI components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing AI components: {e}")
            return False
    
    def initialize_cpp_bridge(self) -> bool:
        """Initialize C++ backend bridge."""
        try:
            self.logger.info("Initializing C++ bridge...")
            
            self.cpp_bridge = CppBridge(self.config)
            if not self.cpp_bridge.initialize(
                input_device=self.args.input,
                output_device=self.args.output
            ):
                self.logger.error("Failed to initialize C++ bridge")
                return False
            
            self.logger.info("C++ bridge initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing C++ bridge: {e}")
            return False
    
    def run_gui(self) -> int:
        """Run the application with GUI."""
        app = QApplication(sys.argv)
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        
        # Create main window
        main_window = MainWindow(
            config=self.config,
            face_detector=self.face_detector,
            face_identifier=self.face_identifier,
            background_segmenter=self.background_segmenter,
            cpp_bridge=self.cpp_bridge
        )
        
        main_window.show()
        return app.exec()
    
    def run_headless(self) -> int:
        """Run the application in headless mode."""
        self.logger.info("Running in headless mode...")
        
        if not self.cpp_bridge:
            self.logger.error("C++ bridge not initialized")
            return 1
        
        try:
            # Start the C++ processing loop
            self.cpp_bridge.start_processing()
            
            # Keep the application running
            import signal
            import time
            
            def signal_handler(signum, frame):
                self.logger.info("Shutdown signal received")
                self.cpp_bridge.stop_processing()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            while self.cpp_bridge.is_running():
                time.sleep(1)
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            return 0
        except Exception as e:
            self.logger.error(f"Error in headless mode: {e}")
            return 1
        finally:
            if self.cpp_bridge:
                self.cpp_bridge.stop_processing()
    
    def run(self) -> int:
        """Run the main application."""
        # Initialize components
        if not self.initialize_ai_components():
            return 1
        
        if not self.initialize_cpp_bridge():
            return 1
        
        # Run GUI or headless mode
        if self.args.nogui:
            return self.run_headless()
        else:
            return self.run_gui()


def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.debug:
        config.setdefault("logging", {})["level"] = "DEBUG"
    
    # Setup logging
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Linucast...")
    
    try:
        # Create and run application
        app = OpenCamFXApp(config, args)
        return app.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
