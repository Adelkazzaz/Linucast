# Linucast Troubleshooting Guide

This guide helps you diagnose and fix common issues with Linucast.

## Virtual Camera Issues

### Camera not appearing in applications

**Symptoms:** Linucast is running, but the virtual camera doesn't appear in your applications.

**Solutions:**

1. Check if the v4l2loopback module is loaded:
   ```bash
   lsmod | grep v4l2loopback
   ```
   If it's not listed, load it manually:
   ```bash
   sudo modprobe v4l2loopback video_nr=10 card_label="Linucast" exclusive_caps=1
   ```

2. Verify the virtual device exists:
   ```bash
   ls -l /dev/video10
   ```
   If it doesn't exist, there might be an issue with the v4l2loopback module installation.

3. Check permissions on the device:
   ```bash
   sudo chmod 666 /dev/video10
   ```

4. Ensure pyvirtualcam is installed:
   ```bash
   pip3 list | grep pyvirtualcam
   ```
   If not, install it:
   ```bash
   pip3 install pyvirtualcam
   ```

5. Restart Linucast with the virtual camera option explicitly:
   ```bash
   cd python_core
   python3 linucast_simple.py --virtual-cam
   ```

### Black screen in virtual camera

**Symptoms:** The virtual camera appears in applications but shows a black screen.

**Solutions:**

1. Make sure Linucast is running with the `--virtual-cam` option
2. Check terminal output for any errors related to the virtual camera
3. Try restarting the application using the camera
4. Verify the camera device number matches the v4l2loopback device:
   ```bash
   v4l2-ctl --list-devices
   ```

### Error: "Failed to create virtual camera"

**Symptoms:** Terminal shows "Failed to create virtual camera" error.

**Solutions:**

1. Check if another application is using the virtual camera
2. Try restarting the v4l2loopback module:
   ```bash
   sudo rmmod v4l2loopback
   sudo modprobe v4l2loopback video_nr=10 card_label="Linucast" exclusive_caps=1
   ```
3. Verify pyvirtualcam is installed correctly:
   ```bash
   pip3 uninstall pyvirtualcam
   pip3 install pyvirtualcam
   ```

## Performance Issues

### High CPU Usage

**Symptoms:** Linucast uses a lot of CPU and runs slowly.

**Solutions:**

1. Try a lower resolution:
   ```bash
   python3 linucast_simple.py --resolution 640x360
   ```
2. Disable features you don't need:
   - Turn off face landmarks: remove `--landmarks` option
   - Use a simpler background mode: `--mode blur` instead of `--mode replace`
3. Lower the FPS target:
   ```bash
   python3 linucast_simple.py --fps 30
   ```

### Frame drops or lag

**Symptoms:** Video is choppy or delayed.

**Solutions:**

1. Check your system's resources (CPU, memory) during operation
2. Close other resource-intensive applications
3. Reduce the resolution or disable complex features
4. If face tracking causes lag, adjust the smoothing factor:
   ```bash
   python3 linucast_simple.py --face-tracking --smoothing 0.3
   ```

## Face Tracking Issues

### Jittery or unstable tracking

**Symptoms:** Face tracking jumps around or is unstable.

**Solutions:**

1. Adjust the smoothing factor:
   ```bash
   python3 linucast_simple.py --face-tracking --smoothing 0.4
   ```
   - Higher values (0.3-0.5) = smoother but slower response
   - Lower values (0.05-0.2) = quicker response but may be jittery

2. Improve lighting conditions in your room
3. Try a lower zoom ratio if tracking is unstable:
   ```bash
   python3 linucast_simple.py --face-tracking --zoom-ratio 1.5
   ```

### Face getting cut off

**Symptoms:** Parts of your face disappear at the edges of the frame.

**Solutions:**

1. Decrease the zoom ratio:
   ```bash
   python3 linucast_simple.py --face-tracking --zoom-ratio 1.5
   ```
2. Position yourself more centrally to the camera
3. Increase your distance from the camera

## Installation Issues

### "Command not found" during installation

**Symptoms:** Installation script shows "command not found" errors.

**Solutions:**

1. Make sure you have basic development tools installed:
   ```bash
   sudo apt update
   sudo apt install build-essential cmake pkg-config git python3-dev
   ```
2. Try running the installation with sudo:
   ```bash
   sudo ./install/install.sh
   ```

### Failed to build C++ components

**Symptoms:** Errors during the C++ build process.

**Solutions:**

1. Install required development packages:
   ```bash
   sudo apt install build-essential cmake pkg-config
   ```
2. Check CMake version:
   ```bash
   cmake --version
   ```
   You need at least version 3.10
3. Try building manually:
   ```bash
   cd cpp_core
   mkdir -p build
   cd build
   cmake ..
   make
   ```

## Other Issues

### Can't find or run Linucast

**Symptoms:** Can't find Linucast in applications menu or start script doesn't work.

**Solutions:**

1. Check if the desktop entry was installed:
   ```bash
   ls -l /usr/share/applications/linucast-virtual-camera.desktop
   ```
2. Try running Linucast directly:
   ```bash
   cd /path/to/Linucast
   ./start_virtual_camera.sh
   ```
3. If start script doesn't work, check its permissions:
   ```bash
   chmod +x start_virtual_camera.sh
   ```

### Module not loading on boot

**Symptoms:** v4l2loopback module doesn't load automatically after reboot.

**Solutions:**

1. Check if configuration files were created:
   ```bash
   cat /etc/modules-load.d/v4l2loopback.conf
   cat /etc/modprobe.d/v4l2loopback.conf
   ```
2. Try loading the module manually after boot:
   ```bash
   sudo modprobe v4l2loopback video_nr=10 card_label="Linucast" exclusive_caps=1
   ```
3. Add the commands to your system startup scripts

## Reporting Issues

If you've tried the solutions above and still have problems:

1. Gather information about your system:
   ```bash
   uname -a
   lsb_release -a
   python3 --version
   pip3 list | grep -E "pyvirtualcam|opencv|numpy|mediapipe"
   ```
2. Note the exact error messages you're seeing
3. Open an issue on the Linucast GitHub repository with these details
