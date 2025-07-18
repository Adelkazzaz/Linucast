# Linucast Installation Guide

This guide provides instructions for installing Linucast on Linux systems.

## Prerequisites

- Linux system (tested on Ubuntu 20.04+)
- Python 3.8 or higher
- Webcam
- v4l2loopback (for virtual camera output)

## Step 1: Install Python Dependencies

```bash
# Install required Python packages
pip install mediapipe opencv-python numpy

# Optional: for virtual camera support
pip install pyvirtualcam
```

## Step 2: Set Up Virtual Camera (Optional)

To use Linucast as a virtual camera for video conferencing:

```bash
# Install v4l2loopback
sudo apt install v4l2loopback-dkms

# Load the kernel module
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="LinucastCam" exclusive_caps=1
```

To make the virtual camera persistent across reboots:

```bash
echo "v4l2loopback" | sudo tee -a /etc/modules
echo "options v4l2loopback devices=1 video_nr=10 card_label=LinucastCam exclusive_caps=1" | sudo tee -a /etc/modprobe.d/v4l2loopback.conf
```

## Troubleshooting

### Virtual Camera Issues

- Make sure the v4l2loopback module is loaded: `lsmod | grep v4l2loopback`
- Check if the virtual camera device exists: `ls -la /dev/video*`
- If necessary, try with elevated privileges: `sudo python linucast_simple.py --virtual-cam`

### MediaPipe Installation Issues

If you encounter issues with MediaPipe installation:

```bash
# Try installing a specific version
pip install mediapipe==0.9.1
```

### OpenCV Issues

If OpenCV doesn't detect your camera:

```bash
# Install additional codecs and camera support
sudo apt install libv4l-dev
```
