import logging
logging.basicConfig(level=logging.DEBUG)
#!/usr/bin/env python3
"""Simple test script for Linucast AI components."""

import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import linucast
sys.path.insert(0, str(Path(__file__).parent.parent / "python_core"))

def test_imports():
    """Test that all required modules can be imported."""
    tests = []
    
    # Test basic Python imports
    try:
        import numpy as np
        tests.append(("NumPy", "✓", np.__version__))
    except ImportError as e:
        tests.append(("NumPy", "✗", str(e)))
    
    try:
        import cv2
        tests.append(("OpenCV", "✓", cv2.__version__))
    except ImportError as e:
        tests.append(("OpenCV", "✗", str(e)))
    
    try:
        import yaml
        tests.append(("PyYAML", "✓", "Available"))
    except ImportError as e:
        tests.append(("PyYAML", "✗", str(e)))
    
    try:
        from PyQt6.QtWidgets import QApplication
        tests.append(("PyQt6", "✓", "Available"))
    except ImportError as e:
        tests.append(("PyQt6", "✗", str(e)))
    
    try:
        import mediapipe as mp
        tests.append(("MediaPipe", "✓", mp.__version__))
    except ImportError as e:
        tests.append(("MediaPipe", "✗", str(e)))
    
    try:
        import onnxruntime as ort
        tests.append(("ONNX Runtime", "✓", ort.__version__))
    except ImportError as e:
        tests.append(("ONNX Runtime", "✗", str(e)))
    
    # Test Linucast modules
    try:
        from linucast.ai.face_detector import FaceDetector
        tests.append(("Linucast Face Detector", "✓", "Available"))
    except ImportError as e:
        tests.append(("Linucast Face Detector", "✗", str(e)))
    
    try:
        from linucast.ai.face_id import FaceIdentifier
        tests.append(("Linucast Face ID", "✓", "Available"))
    except ImportError as e:
        tests.append(("Linucast Face ID", "✗", str(e)))
    
    try:
        from linucast.ai.segmenter import BackgroundSegmenter
        tests.append(("Linucast Segmenter", "✓", "Available"))
    except ImportError as e:
        tests.append(("Linucast Segmenter", "✗", str(e)))
    
    try:
        from linucast.ipc.bridge import CppBridge
        tests.append(("Linucast C++ Bridge", "✓", "Available"))
    except ImportError as e:
        tests.append(("Linucast C++ Bridge", "✗", str(e)))
    
    # Test C++ module
    try:
        import linucast
        if hasattr(linucast, 'linucast_cpp') and linucast.linucast_cpp is not None:
            # Additional check to make sure it's a valid module
            module_attrs = dir(linucast.linucast_cpp)
            if len(module_attrs) > 0:
                tests.append(("Linucast C++ Module", "✓", "Available"))
            else:
                tests.append(("Linucast C++ Module", "✗", "Module loaded but empty"))
        else:
            tests.append(("Linucast C++ Module", "✗", "Module found but not accessible"))
    except ImportError as e:
        tests.append(("Linucast C++ Module", "✗", str(e)))
    
    return tests

def test_system_requirements():
    """Test system requirements."""
    tests = []
    
    # Check for v4l2loopback
    try:
        import subprocess
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        if 'v4l2loopback' in result.stdout:
            tests.append(("v4l2loopback", "✓", "Loaded"))
        else:
            tests.append(("v4l2loopback", "✗", "Not loaded"))
    except Exception as e:
        tests.append(("v4l2loopback", "✗", str(e)))
    
    # Check for virtual camera device
    try:
        from pathlib import Path
        if Path("/dev/video10").exists():
            tests.append(("Virtual Camera (/dev/video10)", "✓", "Available"))
        else:
            tests.append(("Virtual Camera (/dev/video10)", "✗", "Not found"))
    except Exception as e:
        tests.append(("Virtual Camera", "✗", str(e)))
    
    # Check for input camera
    try:
        if Path("/dev/video0").exists():
            tests.append(("Input Camera (/dev/video0)", "✓", "Available"))
        else:
            tests.append(("Input Camera (/dev/video0)", "?", "Not found (may be normal)"))
    except Exception as e:
        tests.append(("Input Camera", "✗", str(e)))
    
    return tests

def print_results(title, tests):
    """Print test results in a formatted table."""
    print(f"\n{title}")
    print("=" * len(title))
    
    for name, status, info in tests:
        print(f"{name:<25} {status:<3} {info}")

def main():
    """Main test function."""
    print("Linucast System Test")
    print("===================")
    
    # Test imports
    import_tests = test_imports()
    print_results("Python Dependencies", import_tests)
    
    # Test system requirements
    system_tests = test_system_requirements()
    print_results("System Requirements", system_tests)
    
    # Summary
    total_tests = len(import_tests) + len(system_tests)
    passed_tests = sum(1 for _, status, _ in import_tests + system_tests if status == "✓")
    
    print(f"\nSummary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✓ All tests passed! Linucast should work correctly.")
        return 0
    else:
        failed_tests = [name for name, status, _ in import_tests + system_tests if status == "✗"]
        print("✗ Some tests failed. Please check the following:")
        for test in failed_tests:
            print(f"  - {test}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
