#!/bin/bash

set -e

echo "Building Linucast C++ Backend"
echo "============================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CPP_DIR="$PROJECT_DIR/cpp_core"

# Check if we're in the right directory
if [ ! -d "$CPP_DIR" ]; then
    echo "Error: cpp_core directory not found at $CPP_DIR"
    exit 1
fi

echo "Project directory: $PROJECT_DIR"
echo "C++ source directory: $CPP_DIR"

# Navigate to cpp_core directory
cd "$CPP_DIR"

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Check if CMAKE is available
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install cmake."
    exit 1
fi

# Configure build
echo "Configuring build with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building C++ components..."
make -j$(nproc)

echo "✓ C++ build completed!"

# Check if pybind11 module was built
if [ -f "linucast_cpp*.so" ]; then
    echo "Python module built successfully!"
    
    # Copy to Python package directory
    PYTHON_PKG_DIR="$PROJECT_DIR/python_core/linucast"
    if [ -d "$PYTHON_PKG_DIR" ]; then
        echo "Copying Python module to package directory..."
        cp linucast_cpp*.so "$PYTHON_PKG_DIR/"
        echo "Python module installed to $PYTHON_PKG_DIR/"
    fi
else
    echo "Warning: Python module not built. Make sure pybind11 is installed."
fi

echo "Build process completed!"
