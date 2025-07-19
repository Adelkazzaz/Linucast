# Linucast Virtual Camera Guide

This document explains how Linucast integrates with the Linux virtual camera system using v4l2loopback and pyvirtualcam.

## How the Virtual Camera Works

Linucast creates a virtual camera that can be used in any application that supports V4L2 cameras (browsers, video conferencing apps, etc.). The system consists of:

1. **v4l2loopback**: A Linux kernel module that creates virtual video devices
2. **pyvirtualcam**: A Python library that allows applications to output video to these virtual devices
3. **Linucast**: Processes the webcam feed and outputs to the virtual device

## Setup Process

The installation script performs the following steps:

1. Installs the v4l2loopback kernel module
2. Loads the module with specific parameters:
   - `video_nr=10`: Creates device at /dev/video10
   - `card_label="Linucast"`: Names the device "Linucast"
   - `exclusive_caps=1`: Makes the device appear as a proper camera to applications

3. Sets permissions for the device file so non-root users can access it
4. Configures the module to load automatically on system boot

## Using the Virtual Camera

After starting Linucast with:
```bash
./start_virtual_camera.sh
```

The virtual camera becomes available to applications. To use it:

1. Open your application (e.g., Zoom, Google Meet, Microsoft Teams)
2. In the camera/video settings, select "Linucast" as your camera device
3. You should see your processed webcam feed with all Linucast effects applied

## Troubleshooting

### Virtual Camera Not Appearing in Applications

1. Check if the v4l2loopback module is loaded:
   ```bash
   lsmod | grep v4l2loopback
   ```

2. Verify the virtual device exists:
   ```bash
   ls -l /dev/video10
   ```

3. Check if the device has the correct label:
   ```bash
   v4l2-ctl --list-devices
   ```
   You should see "Linucast" in the output.

4. Ensure pyvirtualcam is installed:
   ```bash
   pip3 list | grep pyvirtualcam
   ```

5. Try restarting the v4l2loopback module:
   ```bash
   sudo rmmod v4l2loopback
   sudo modprobe v4l2loopback video_nr=10 card_label="Linucast" exclusive_caps=1
   sudo chmod 666 /dev/video10
   ```

### Browser Compatibility

For web browsers:

- **Firefox**: Should detect the virtual camera automatically
- **Chrome/Chromium**: May require the `--use-fake-ui-for-media-stream` flag for better virtual camera detection
- **Other Chromium-based browsers**: Similar to Chrome

### Common Issues

1. **Permission denied when accessing camera**:
   ```bash
   sudo chmod 666 /dev/video10
   ```

2. **Module not loading on boot**:
   Check if the configuration files were created:
   ```bash
   cat /etc/modules-load.d/v4l2loopback.conf
   cat /etc/modprobe.d/v4l2loopback.conf
   ```

3. **Black screen in applications**:
   Ensure Linucast is running and outputting to the virtual camera.

4. **"Device or resource busy" error**:
   Another application might be using the virtual camera. Close other applications or try:
   ```bash
   sudo rmmod v4l2loopback
   sudo modprobe v4l2loopback video_nr=10 card_label="Linucast" exclusive_caps=1
   ```

5. **Camera not detected in specific applications**:
   Some applications have more strict requirements for camera devices. Try adjusting v4l2loopback parameters:
   ```bash
   sudo rmmod v4l2loopback
   sudo modprobe v4l2loopback video_nr=10 card_label="Linucast" exclusive_caps=1 max_buffers=2
   ```

## Technical Details

Linucast uses the following components for virtual camera integration:

1. **v4l2loopback**: Creates a virtual V4L2 device that appears as a camera to the system
2. **pyvirtualcam**: Python library that provides an interface to write frames to the virtual camera
3. **Python implementation**: The `setup_virtual_camera()` method in Linucast's code initializes the virtual camera output

### Code Integration

The key part of the integration is in `linucast_simple.py`, where the virtual camera is set up if the `--virtual-cam` option is used:

```python
# Simplified version of the virtual camera setup code
import pyvirtualcam

def setup_virtual_camera(width, height, fps, device="/dev/video10"):
    try:
        cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, device=device, fmt=pyvirtualcam.PixelFormat.RGB)
        print(f"Virtual camera created: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)")
        return cam
    except Exception as e:
        print(f"Failed to create virtual camera: {e}")
        return None
```

When a frame is ready to be sent to the virtual camera:

```python
if virtual_cam is not None:
    # Convert from BGR to RGB for pyvirtualcam
    rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
    virtual_cam.send(rgb_frame)
    virtual_cam.sleep_until_next_frame()
```
