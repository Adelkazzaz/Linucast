#!/bin/bash

# Install dependencies for Linucast
# This script installs system dependencies and Python packages

set -e

echo "Installing Linucast Dependencies"
echo "==============================="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "Please do not run this script as root"
   exit 1
fi

# Update package list
echo "Updating package list..."
sudo apt update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libopencv-dev \
    libopencv-contrib-dev \
    libeigen3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libonnx-dev \
    ffmpeg \
    v4l-utils \
    curl \
    wget

# Install v4l2loopback kernel module
echo "Installing v4l2loopback..."
if ! lsmod | grep -q v4l2loopback; then
    sudo apt install -y v4l2loopback-dkms
    echo "v4l2loopback installed. You may need to reboot."
fi

# Install Python dependencies
echo "Installing Python dependencies..."
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install pybind11
echo "Installing pybind11..."
pip3 install pybind11[global]

# Install ONNX Runtime (CPU version by default)
echo "Installing ONNX Runtime..."
pip3 install onnxruntime

# Check for GPU support
echo "Checking for GPU support..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. You may want to install onnxruntime-gpu:"
    echo "pip3 install onnxruntime-gpu"
fi

if command -v rocm-smi &> /dev/null; then
    echo "AMD GPU with ROCm detected."
fi

# Install MediaPipe
echo "Installing MediaPipe..."
pip3 install mediapipe

# Install additional Python packages
echo "Installing additional Python packages..."
pip3 install \
    opencv-python \
    numpy \
    PyQt6 \
    pyyaml \
    pillow

echo "System dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "1. Run './install/build_cpp.sh' to build the C++ backend"
echo "2. Run './install/setup_v4l2loopback.sh' to configure virtual camera"
echo "3. Navigate to python_core/ and run 'poetry install' to install Python dependencies"
echo ""
echo "Note: You may need to reboot if v4l2loopback was newly installed."
