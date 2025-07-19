#!/bin/bash

# Linucast Uninstall Script
# This script removes Linucast and all its components

set -e

echo "============================================"
echo "      Linucast Uninstall Script"
echo "============================================"
echo ""

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ Please do not run this script as root"
   exit 1
fi

echo "🗑️  Starting Linucast uninstallation..."
echo ""

# Function to ask for confirmation
confirm_removal() {
    echo "⚠️  This will remove:"
    echo "   • Linucast application files"
    echo "   • Desktop shortcuts"
    echo "   • Launcher scripts"
    echo "   • Python virtual environment"
    echo "   • V4L2 loopback configuration (optional)"
    echo ""
    read -p "❓ Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Uninstallation cancelled."
        exit 0
    fi
}

# Get the installation directory (assume script is in install/ subdirectory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Try to detect installation directory if current directory is not the source
if [ ! -f "$PROJECT_DIR/README.md" ] || [ ! -d "$PROJECT_DIR/python_core" ]; then
    echo "🔍 Searching for Linucast installation..."
    
    # Common installation locations
    SEARCH_PATHS=(
        "$HOME/Linucast"
        "$HOME/linucast"
        "$HOME/Downloads/Linucast"
        "$HOME/Downloads/linucast"
        "/tmp/Linucast"
        "/opt/linucast"
    )
    
    PROJECT_DIR=""
    for path in "${SEARCH_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/README.md" ] && [ -d "$path/python_core" ]; then
            PROJECT_DIR="$path"
            echo "✅ Found Linucast installation at: $PROJECT_DIR"
            break
        fi
    done
    
    if [ -z "$PROJECT_DIR" ]; then
        echo "❌ Could not find Linucast installation directory."
        echo "Please run this script from the Linucast installation directory,"
        echo "or manually specify the path:"
        read -p "Enter Linucast installation path (or press Enter to skip): " PROJECT_DIR
        
        if [ -z "$PROJECT_DIR" ] || [ ! -d "$PROJECT_DIR" ]; then
            echo "⚠️  Will only remove system-wide components."
            PROJECT_DIR=""
        fi
    fi
fi

# Show confirmation
confirm_removal

echo "🧹 Removing Linucast components..."

# Step 1: Remove desktop entries
echo "📋 Removing desktop shortcuts..."
DESKTOP_FILES=(
    "$HOME/.local/share/applications/linucast.desktop"
    "$HOME/.local/share/applications/linucast-virtual-camera.desktop"
    "$HOME/Desktop/linucast.desktop"
    "$HOME/Desktop/Linucast.desktop"
    "/usr/share/applications/linucast.desktop"
    "/usr/share/applications/linucast-virtual-camera.desktop"
)

for file in "${DESKTOP_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Removing: $file"
        if [[ "$file" == "/usr/share/applications/"* ]]; then
            sudo rm -f "$file" 2>/dev/null || echo "    Warning: Could not remove $file (permission denied)"
        else
            rm -f "$file"
        fi
    fi
done

# Step 2: Remove launcher scripts
echo "🚀 Removing launcher scripts..."
LAUNCHER_FILES=(
    "$HOME/.local/bin/linucast"
    "/usr/local/bin/linucast"
    "/usr/bin/linucast"
)

for file in "${LAUNCHER_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Removing: $file"
        if [[ "$file" == "/usr/"* ]]; then
            sudo rm -f "$file" 2>/dev/null || echo "    Warning: Could not remove $file (permission denied)"
        else
            rm -f "$file"
        fi
    fi
done

# Step 3: Remove application directory if found
if [ -n "$PROJECT_DIR" ] && [ -d "$PROJECT_DIR" ]; then
    echo "📁 Removing application directory..."
    echo "  Directory: $PROJECT_DIR"
    
    # Stop any running processes first
    echo "  Stopping any running Linucast processes..."
    pkill -f "linucast" 2>/dev/null || true
    pkill -f "python.*linucast" 2>/dev/null || true
    
    # Ask for confirmation before removing the entire directory
    read -p "❓ Remove entire directory $PROJECT_DIR? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$PROJECT_DIR"
        echo "  ✅ Application directory removed"
    else
        echo "  ⏭️  Skipped application directory removal"
    fi
fi

# Step 4: Remove V4L2 loopback configuration (optional)
echo "📹 V4L2 loopback configuration..."
read -p "❓ Remove V4L2 loopback virtual camera configuration? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  Removing V4L2 loopback configuration..."
    
    # Remove module from autoload
    if [ -f "/etc/modules" ] && grep -q "v4l2loopback" "/etc/modules"; then
        sudo sed -i '/v4l2loopback/d' /etc/modules
        echo "  ✅ Removed v4l2loopback from /etc/modules"
    fi
    
    # Remove modprobe configuration
    if [ -f "/etc/modprobe.d/v4l2loopback.conf" ]; then
        sudo rm -f /etc/modprobe.d/v4l2loopback.conf
        echo "  ✅ Removed /etc/modprobe.d/v4l2loopback.conf"
    fi
    
    # Unload module if loaded
    if lsmod | grep -q v4l2loopback; then
        echo "  Unloading v4l2loopback module..."
        sudo modprobe -r v4l2loopback 2>/dev/null || echo "    Warning: Could not unload v4l2loopback module"
    fi
    
    echo "  ⚠️  Note: To completely remove v4l2loopback, uninstall the package:"
    echo "    - Ubuntu/Debian: sudo apt remove v4l2loopback-dkms"
    echo "    - Fedora: sudo dnf remove v4l2loopback"
    echo "    - Arch: sudo pacman -R v4l2loopback-dkms"
else
    echo "  ⏭️  Skipped V4L2 loopback configuration removal"
fi

# Step 5: Clean up PATH modifications
echo "🔧 Cleaning up PATH modifications..."
if [ -f "$HOME/.bashrc" ] && grep -q '$HOME/.local/bin.*PATH' "$HOME/.bashrc"; then
    echo "  Note: ~/.local/bin is still in your PATH (this is usually desired)"
    echo "  If you want to remove it, edit ~/.bashrc manually"
fi

# Step 6: Remove temporary files
echo "🧽 Cleaning up temporary files..."
rm -rf /tmp/linucast* 2>/dev/null || true
rm -rf /tmp/Linucast* 2>/dev/null || true

echo ""
echo "🎉 Linucast uninstallation complete!"
echo ""
echo "The following items were processed:"
echo "✅ Desktop shortcuts removed"
echo "✅ Launcher scripts removed"
if [ -n "$PROJECT_DIR" ]; then
    echo "✅ Application files handled"
fi
echo "✅ Temporary files cleaned"
echo ""
echo "If you installed Linucast using a package manager or want to remove"
echo "system dependencies, you may need to remove them manually:"
echo ""
echo "Python packages (if installed globally):"
echo "  pip uninstall mediapipe opencv-python numpy pyvirtualcam"
echo ""
echo "System packages (optional - may be used by other applications):"
echo "  - Ubuntu/Debian: sudo apt remove python3-pyqt5 libopencv-dev"
echo "  - Fedora: sudo dnf remove qt5-qtbase-devel opencv-devel"
echo "  - Arch: sudo pacman -R qt5-base opencv"
echo ""
echo "Thank you for using Linucast! 👋"
