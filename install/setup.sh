#!/bin/bash

# Linucast Complete Setup Script
# This script installs all dependencies and sets up Linucast

set -e

echo "============================================"
echo "  Linucast - AI Virtual Camera for Linux"
echo "============================================"
echo ""

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ Please do not run this script as root"
   exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting Linucast setup..."
echo "Project directory: $PROJECT_DIR"
echo ""

# Step 1: Install system dependencies
echo "📦 Step 1: Installing system dependencies..."
"$SCRIPT_DIR/install_deps.sh"

if [ $? -ne 0 ]; then
    echo "❌ Failed to install system dependencies"
    exit 1
fi

echo "✅ System dependencies installed"
echo ""

# Step 2: Setup virtual camera
echo "📹 Step 2: Setting up virtual camera..."
"$SCRIPT_DIR/setup_v4l2loopback.sh"

if [ $? -ne 0 ]; then
    echo "❌ Failed to setup virtual camera"
    exit 1
fi

echo "✅ Virtual camera setup complete"
echo ""

# Step 3: Install Python dependencies
echo "🐍 Step 3: Installing Python dependencies..."
cd "$PROJECT_DIR/python_core"

# Install Poetry if not available
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install Python packages
poetry install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

echo "✅ Python dependencies installed"
echo ""

# Step 4: Build C++ backend
echo "⚙️  Step 4: Building C++ backend..."
"$SCRIPT_DIR/build_cpp.sh"

if [ $? -ne 0 ]; then
    echo "❌ Failed to build C++ backend"
    exit 1
fi

echo "✅ C++ backend built successfully"
echo ""

# Step 5: Run tests
echo "🧪 Step 5: Running system tests..."
cd "$PROJECT_DIR"
python3 tools/test_setup.py

echo ""

# Final instructions
echo "🎉 Linucast setup complete!"
echo ""
echo "Next steps:"
echo "1. Reboot your system (recommended for v4l2loopback)"
echo "2. Run the test: python3 tools/test_setup.py"
echo "3. Start Linucast with GUI: cd python_core && poetry run linucast"
echo "4. Or run headless: cd python_core && poetry run linucast --nogui"
echo ""
echo "For video conferencing:"
echo "- Select 'Linucast' as your camera in Zoom, Discord, etc."
echo "- The virtual camera device is /dev/video10"
echo ""
echo "Documentation:"
echo "- README.md - Main documentation"
echo "- docs/ - Detailed specifications"
echo ""
echo "Troubleshooting:"
echo "- Check logs in python_core/logs/"
echo "- Run test script: python3 tools/test_setup.py"
echo "- For issues, check: /dev/video* devices"
