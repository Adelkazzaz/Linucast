#!/bin/bash

# Setup v4l2loopback virtual camera device for Linucast

set -e

echo "Setting up v4l2loopback Virtual Camera"
echo "======================================"

# Check if v4l2loopback is installed
if ! lsmod | grep -q v4l2loopback; then
    echo "v4l2loopback module not loaded. Installing..."
    
    # Try to install if not present
    if ! dpkg -l | grep -q v4l2loopback-dkms; then
        echo "Installing v4l2loopback-dkms..."
        sudo apt update
        sudo apt install -y v4l2loopback-dkms
    fi
    
    # Load the module
    echo "Loading v4l2loopback module..."
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Linucast" exclusive_caps=1
else
    echo "v4l2loopback module already loaded."
fi

# Check if virtual device exists
if [ -e "/dev/video10" ]; then
    echo "Virtual camera device /dev/video10 is ready!"
    
    # Show device info
    echo ""
    echo "Device information:"
    v4l2-ctl --device=/dev/video10 --info
    
    echo ""
    echo "Supported formats:"
    v4l2-ctl --device=/dev/video10 --list-formats-ext
else
    echo "Warning: Virtual camera device /dev/video10 not found."
    echo "Trying to create it manually..."
    
    # Remove existing module and reload with specific parameters
    sudo modprobe -r v4l2loopback 2>/dev/null || true
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Linucast" exclusive_caps=1
    
    sleep 2
    
    if [ -e "/dev/video10" ]; then
        echo "Virtual camera device /dev/video10 created successfully!"
    else
        echo "Error: Failed to create virtual camera device."
        echo "Please check your kernel and v4l2loopback installation."
        exit 1
    fi
fi

# Set up automatic loading on boot
echo ""
echo "Setting up automatic module loading on boot..."

# Create modprobe configuration
sudo tee /etc/modprobe.d/v4l2loopback.conf > /dev/null << EOF
# Linucast v4l2loopback configuration
options v4l2loopback devices=1 video_nr=10 card_label="Linucast" exclusive_caps=1
EOF

# Add module to load at boot
if ! grep -q "v4l2loopback" /etc/modules; then
    echo "v4l2loopback" | sudo tee -a /etc/modules
    echo "Added v4l2loopback to /etc/modules for automatic loading."
fi

# Set permissions for video group
echo "Setting up permissions..."
sudo usermod -a -G video $USER

echo ""
echo "Setup completed successfully!"
echo ""
echo "Virtual camera device: /dev/video10"
echo "Card label: Linucast"
echo ""
echo "You can test the virtual camera with:"
echo "  ffplay /dev/video10"
echo "  or"
echo "  vlc v4l2:///dev/video10"
echo ""
echo "Note: You may need to log out and back in for group permissions to take effect."
echo "      Or run 'newgrp video' to refresh group membership."
echo ""
echo "To test the setup:"
echo "1. Run Linucast"
echo "2. Open your video conferencing app"
echo "3. Select 'Linucast' as your camera device"
