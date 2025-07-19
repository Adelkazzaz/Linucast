# Linucast Installation Guide

This guide provides detailed instructions for installing Linucast on your Linux system.

## Quick Installation

The easiest way to install Linucast is using the one-line installer:

```bash
curl -sSL https://raw.githubusercontent.com/Adelkazzaz/Linucast/main/install.sh | bash
```

This command will:
1. Download the Linucast repository
2. Install all required dependencies
3. Set up the virtual camera
4. Create desktop shortcuts

## Manual Installation

If you prefer to install manually or need more control over the installation process:

### 1. Clone the Repository

```bash
git clone https://github.com/Adelkazzaz/Linucast.git
cd Linucast
```

### 2. Run the Installation Script

```bash
./install/complete_install.sh
```

## System Dependencies

Linucast depends on the following packages:

- build-essential, cmake, pkg-config (for building C++ components)
- python3, python3-dev, python3-pip (for running Python components)
- v4l-utils, v4l2loopback-dkms (for virtual camera functionality)
- python3-opencv, ffmpeg (for video processing)

The installation script will attempt to install these automatically for you.

## Python Dependencies

Linucast requires these Python packages:

- opencv-python
- numpy
- mediapipe
- pyvirtualcam

The installation script will install these for you using pip or Poetry.

## Virtual Camera Setup

The v4l2loopback kernel module is used to create a virtual camera device. The installation script configures this module to:

1. Create a virtual video device at `/dev/video10`
2. Label it as "Linucast" for easy identification
3. Set appropriate permissions so non-root users can access it
4. Configure the module to load automatically on boot

## Manual v4l2loopback Setup (if needed)

If you need to manually set up the v4l2loopback module:

```bash
# Install the module
sudo apt install v4l2loopback-dkms  # For Debian/Ubuntu
# or
sudo dnf install v4l2loopback  # For Fedora/RHEL
# or 
sudo pacman -S v4l2loopback-dkms  # For Arch

# Load the module
sudo modprobe v4l2loopback video_nr=10 card_label="Linucast" exclusive_caps=1

# Set permissions
sudo chmod 666 /dev/video10

# Make it load on boot
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf > /dev/null
echo "options v4l2loopback video_nr=10 card_label=Linucast exclusive_caps=1" | sudo tee /etc/modprobe.d/v4l2loopback.conf > /dev/null
```

## Starting Linucast

After installation, you can start Linucast in two ways:

1. From your applications menu (look for "Linucast Virtual Camera")
2. By running the start script:
   ```bash
   ./start_virtual_camera.sh
   ```

## Verifying Installation

To verify the installation:

1. Check that the v4l2loopback module is loaded:
   ```bash
   lsmod | grep v4l2loopback
   ```

2. Check that the virtual camera device exists:
   ```bash
   ls -l /dev/video10
   v4l2-ctl --list-devices | grep -A1 Linucast
   ```

3. Run Linucast and check the terminal output for errors
