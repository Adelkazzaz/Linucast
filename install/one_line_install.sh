#!/bin/bash
#
# Linucast One-Line Installer Script
# This script handles the complete installation of Linucast in a single command

set -e # Exit on any error

echo "
██╗     ██╗███╗   ██╗██╗   ██╗ ██████╗ █████╗ ███████╗████████╗
██║     ██║████╗  ██║██║   ██║██╔════╝██╔══██╗██╔════╝╚══██╔══╝
██║     ██║██╔██╗ ██║██║   ██║██║     ███████║███████╗   ██║   
██║     ██║██║╚██╗██║██║   ██║██║     ██╔══██║╚════██║   ██║   
███████╗██║██║ ╚████║╚██████╔╝╚██████╗██║  ██║███████║   ██║   
╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝   
"
echo "=== Linucast One-Line Installer ==="
echo "Installing Linucast - AI Virtual Camera for Linux..."

# Create temporary directory
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

# Detect package manager
if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt"
    INSTALL_CMD="sudo apt-get install -y"
    UPDATE_CMD="sudo apt-get update"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
    INSTALL_CMD="sudo dnf install -y"
    UPDATE_CMD="sudo dnf check-update"
elif command -v pacman &> /dev/null; then
    PKG_MANAGER="pacman"
    INSTALL_CMD="sudo pacman -Sy --noconfirm"
    UPDATE_CMD="sudo pacman -Sy"
else
    echo "Unsupported package manager. Please install dependencies manually."
    echo "Required: git, python3, pip, cmake, build-essential, libopencv-dev"
    exit 1
fi

# Check if git is installed, install if not
if ! command -v git &> /dev/null; then
    echo "Git not found. Installing git..."
    $UPDATE_CMD
    $INSTALL_CMD git
fi

# Clone the repository
echo "Cloning Linucast repository..."
git clone https://github.com/Adelkazzaz/Linucast.git
cd Linucast

# Install system dependencies
echo "Installing system dependencies..."

case $PKG_MANAGER in
    apt)
        $UPDATE_CMD
        $INSTALL_CMD \
            python3-dev \
            python3-pip \
            python3-venv \
            cmake \
            build-essential \
            libopencv-dev \
            libjpeg-dev \
            libpng-dev \
            libavcodec-dev \
            libavformat-dev \
            libswscale-dev \
            v4l2loopback-dkms \
            python3-pyqt5
        ;;
    dnf)
        $UPDATE_CMD
        $INSTALL_CMD \
            python3-devel \
            python3-pip \
            python3-virtualenv \
            cmake \
            gcc-c++ \
            make \
            opencv-devel \
            libjpeg-turbo-devel \
            libpng-devel \
            ffmpeg-devel \
            v4l2loopback \
            qt5-qtbase-devel
        ;;
    pacman)
        $UPDATE_CMD
        $INSTALL_CMD \
            python \
            python-pip \
            python-virtualenv \
            cmake \
            base-devel \
            opencv \
            libjpeg-turbo \
            libpng \
            ffmpeg \
            v4l2loopback-dkms \
            qt5-base
        ;;
esac

# Set up v4l2loopback
echo "Setting up virtual camera..."
if lsmod | grep -q v4l2loopback; then
    echo "v4l2loopback already loaded."
else
    echo "Loading v4l2loopback module..."
    if ! sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="LinucastCam" exclusive_caps=1; then
        echo "Warning: Failed to load v4l2loopback. Virtual camera may not be available."
        echo "Check if v4l2loopback-dkms is properly installed for your kernel."
        VIRTUAL_CAM_FAILED=1
    fi
fi

# Make v4l2loopback persistent
if [ -z "$VIRTUAL_CAM_FAILED" ]; then
    echo "Making virtual camera persistent..."
    if ! grep -q "v4l2loopback" /etc/modules; then
        echo "v4l2loopback" | sudo tee -a /etc/modules > /dev/null
    fi
    
    if ! grep -q "options v4l2loopback" /etc/modprobe.d/v4l2loopback.conf 2>/dev/null; then
        echo "options v4l2loopback devices=1 video_nr=10 card_label=LinucastCam exclusive_caps=1" | sudo tee -a /etc/modprobe.d/v4l2loopback.conf > /dev/null
    fi
    
    echo "Virtual camera setup complete."
fi

# Set up Python environment
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install mediapipe opencv-python numpy pyvirtualcam

# Build C++ components
echo "Building C++ components..."
mkdir -p cpp_core/build
cd cpp_core/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..

# Copy C++ library to Python module
echo "Installing C++ modules into Python package..."
cp cpp_core/build/liblinucast.so python_core/linucast/
cp cpp_core/build/linucast_cpp*.so python_core/linucast/

# Ensure local directories exist
mkdir -p ~/.local/bin
mkdir -p ~/.local/share/applications

# Remove any existing desktop entries first to avoid duplicates
echo "Removing any existing desktop shortcuts..."
rm -f ~/.local/share/applications/linucast*.desktop 2>/dev/null || true
rm -f ~/Desktop/linucast*.desktop 2>/dev/null || true
rm -f ~/Desktop/Linucast*.desktop 2>/dev/null || true

# Create desktop shortcut
echo "Creating desktop shortcut..."
cat > ~/.local/share/applications/linucast.desktop << EOF
[Desktop Entry]
Version=1.0
Name=Linucast
Comment=AI Virtual Camera for Linux
Exec=$(pwd)/venv/bin/python $(pwd)/python_core/linucast_simple.py
Icon=$(pwd)/python_core/assets/linucast-logo.png
Terminal=false
Type=Application
Categories=Video;AudioVideo;Graphics;
Keywords=camera;webcam;virtual camera;AI;background removal;
StartupNotify=true
EOF

# Make sure desktop file is executable and properly formatted
chmod 644 ~/.local/share/applications/linucast.desktop

echo "Desktop shortcut created successfully"

# Create a launcher script in user's PATH
echo "Creating launcher script..."
cat > ~/.local/bin/linucast << EOF
#!/bin/bash
cd $(pwd)
source venv/bin/activate
cd python_core
python linucast_simple.py "\$@"
EOF
chmod +x ~/.local/bin/linucast

# Add ~/.local/bin to PATH if not already there
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    echo "Adding ~/.local/bin to your PATH..."
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "Please run 'source ~/.bashrc' after installation to update your PATH"
fi

echo ""
echo "🎉 Installation complete! 🎉"
echo ""
echo "You can now run Linucast by:"
echo "1. Using the desktop shortcut in your applications menu"
echo "2. Running the 'linucast' command in your terminal"
echo "3. Manually with: cd $(pwd) && source venv/bin/activate && cd python_core && python linucast_simple.py"
echo ""

if [ -z "$VIRTUAL_CAM_FAILED" ]; then
    echo "✅ To use as a virtual camera, select 'LinucastCam' in your video conferencing app."
else
    echo "⚠️  Virtual camera setup failed. Please install v4l2loopback manually for this feature."
fi

echo ""
echo "For help and documentation, see: https://github.com/Adelkazzaz/Linucast"
echo "Enjoy your new AI-powered virtual camera!"
