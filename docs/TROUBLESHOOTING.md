# Troubleshooting Linucast

This guide provides solutions to common issues encountered when using Linucast.

## Virtual Camera Setup Issues

### Virtual camera not detected

```bash
# Check if v4l2loopback is loaded
lsmod | grep v4l2loopback

# If not loaded, load the module
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="LinucastCam" exclusive_caps=1

# If already loaded but not working, try reloading
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="LinucastCam" exclusive_caps=1
```

### Permissions issues with camera devices

```bash
# Add your user to the video group
sudo usermod -a -G video $USER
# Log out and log back in for changes to take effect

# Check permissions on camera devices
ls -l /dev/video*
# Devices should be readable/writable by the video group
```

## Performance Issues

### Low FPS

If you're experiencing low frame rates:

1. Try a lower resolution: `--resolution 640x360`
2. Disable face landmarks if not needed
3. Switch to a lighter background mode (e.g., "remove" instead of "blur")
4. Check system load with `htop` or `top` to see if other processes are consuming resources

### High CPU Usage

1. Make sure you're using hardware acceleration if available
2. In the configuration file, set `ai.device` to `cuda` (NVIDIA), `rocm` (AMD) or `auto`
3. Lower the processing resolution
4. Reduce the target FPS (e.g., 30 instead of 60)

## Face Tracking Problems

### Jittery or unstable tracking

- Increase the smoothing factor: `--smoothing 0.3` (higher value = smoother but slower response)
- Ensure adequate lighting in your environment
- Position your face more centrally and avoid rapid movements

### Face not detected

- Check lighting conditions - ensure your face is well lit
- Make sure your face is clearly visible and not partially out of frame
- Try adjusting the face detection confidence threshold: `--threshold 0.4` (lower value = more sensitive)

## Python Issues

### Module import errors

If you encounter Python module import errors:

```bash
# Rebuild C++ module
cd cpp_core/build
make -j$(nproc)
cp linucast_cpp*.so ../../python_core/linucast/

# Reinstall the Python package
cd ../../python_core
poetry install
```

### Package dependencies

```bash
# Check installed packages
poetry show

# Update all dependencies
poetry update
```

## Debug Mode

For detailed troubleshooting, enable debug mode:

```bash
poetry run linucast --debug
```

Check log files in `logs/linucast.log` for error messages and diagnostics.

## Getting Help

If you're still experiencing issues:

1. Check the [GitHub Issues](https://github.com/Adelkazzaz/Linucast/issues) to see if others have reported similar problems
2. Create a new issue with detailed information about your problem
3. Join our community discussions for help
