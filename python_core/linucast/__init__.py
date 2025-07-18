"""Linucast - AI Virtual Camera for Linux."""

__version__ = "0.1.0"
__author__ = "Linucast Team"
__license__ = "Apache-2.0"

import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import C++ module using the custom loader
from ._cpp_loader import cpp_module as linucast_cpp

# Only import Python modules if they exist
try:
    from .ai.face_detector import FaceDetector
    from .ai.face_id import FaceIdentifier
    from .ai.segmenter import BackgroundSegmenter
    
    __all__ = [
        "FaceDetector",
        "FaceIdentifier", 
        "BackgroundSegmenter",
        "linucast_cpp",
    ]
except ImportError as e:
    logging.warning(f"Some Python modules could not be imported: {e}")
    __all__ = ["linucast_cpp"]
