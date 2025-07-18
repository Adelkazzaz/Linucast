"""Module to load the C++ extension."""

import os
import importlib.util
import sys
import logging
import ctypes
from types import ModuleType

logger = logging.getLogger(__name__)

class CppModuleProxy(ModuleType):
    """A proxy class that simulates the C++ module when it cannot be loaded."""
    
    def __init__(self, name="linucast_cpp_proxy"):
        super().__init__(name)
        self.__file__ = "Not available"
        self.__doc__ = "Proxy for the linucast C++ module (not loaded)"
        
    def __getattr__(self, name):
        logger.warning(f"C++ module not loaded, '{name}' is not available")
        return None

def load_cpp_module():
    """Load the C++ extension module."""
    # Get current directory and other possible locations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to load liblinucast.so first as it's a dependency
    lib_paths = [
        os.path.join(current_dir, "liblinucast.so"),
        os.path.join(current_dir, "..", "..", "cpp_core", "build", "liblinucast.so"),
    ]
    
    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            try:
                logger.info(f"Loading dependency: {lib_path}")
                ctypes.CDLL(lib_path)
                logger.info("Successfully loaded C++ library dependency")
                break
            except Exception as e:
                logger.error(f"Failed to load C++ library dependency: {e}")
    
    # Now try to load the actual module
    module_name = "linucast_cpp"
    module_paths = [
        os.path.join(current_dir, "linucast_cpp.cpython-310-x86_64-linux-gnu.so"),
        os.path.join(current_dir, "..", "..", "cpp_core", "build", "linucast_cpp.cpython-310-x86_64-linux-gnu.so"),
    ]
    
    for path in module_paths:
        if os.path.exists(path):
            logger.info(f"Found C++ module at {path}")
            try:
                # Make sure the file is accessible
                logger.info(f"Module file size: {os.path.getsize(path)} bytes")
                
                # Try to load the module
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    logger.info("Successfully loaded C++ module")
                    return module
            except Exception as e:
                logger.error(f"Error loading C++ module from {path}: {e}")
                import traceback
                traceback.print_exc()
    
    logger.warning("C++ module not found or could not be loaded, using proxy")
    return CppModuleProxy()

# Try to load the module
try:
    cpp_module = load_cpp_module()
except Exception as e:
    logger.error(f"Failed to load C++ module: {e}")
    cpp_module = CppModuleProxy()
