# Linucast Installation and Maintenance

## Installation

To install Linucast, use the one-line installer:

```bash
curl -fsSL https://raw.githubusercontent.com/Adelkazzaz/Linucast/main/install/one_line_install.sh | bash
```

Or manually:

```bash
git clone https://github.com/Adelkazzaz/Linucast.git
cd Linucast
./install/setup.sh
```

## Common Issues and Fixes

### 1. Duplicate Desktop Icons

**Problem**: After installation, you see multiple Linucast icons in your applications menu.

**Solution**: Run the desktop icon fix script:

```bash
cd Linucast
./install/fix_desktop_icon.sh
```

This script will:
- Remove all duplicate desktop entries
- Create a single, clean desktop entry named "Linucast"
- Update the applications menu

### 2. Application Stops After Few Minutes

**Problem**: Linucast stops working or crashes after running for a few minutes.

**Solution**: The latest version includes several stability improvements:

- **Automatic camera recovery**: If the camera connection is lost, Linucast will attempt to reconnect
- **Memory management**: Periodic memory cleanup prevents memory leaks
- **Error handling**: Better error handling prevents crashes from temporary issues
- **Health monitoring**: Regular camera health checks ensure stable operation

If you're still experiencing issues:

1. **Update to the latest version**:
   ```bash
   cd Linucast
   git pull origin main
   ./install/build_cpp.sh
   ```

2. **Check system resources**:
   ```bash
   # Monitor CPU and memory usage
   htop
   
   # Check available cameras
   ls /dev/video*
   
   # Check virtual camera status
   v4l2-ctl --list-devices
   ```

3. **Run with debug logging**:
   ```bash
   cd python_core
   python linucast_simple.py --help
   ```

4. **Check logs**:
   ```bash
   tail -f logs/linucast.log
   ```

## Uninstallation

To completely remove Linucast from your system:

```bash
cd Linucast
./install/uninstall.sh
```

The uninstall script will:
- Remove all desktop shortcuts and launcher scripts
- Remove the application directory (with confirmation)
- Clean up V4L2 loopback configuration (optional)
- Remove temporary files

### Manual Uninstallation

If the automatic uninstaller doesn't work, you can manually remove:

1. **Desktop entries**:
   ```bash
   rm -f ~/.local/share/applications/linucast*.desktop
   rm -f ~/Desktop/linucast*.desktop
   sudo rm -f /usr/share/applications/linucast*.desktop
   ```

2. **Launcher scripts**:
   ```bash
   rm -f ~/.local/bin/linucast
   sudo rm -f /usr/local/bin/linucast
   sudo rm -f /usr/bin/linucast
   ```

3. **Application directory**:
   ```bash
   rm -rf ~/Linucast  # or wherever you installed it
   ```

4. **V4L2 loopback (optional)**:
   ```bash
   sudo modprobe -r v4l2loopback
   sudo rm -f /etc/modprobe.d/v4l2loopback.conf
   sudo sed -i '/v4l2loopback/d' /etc/modules
   ```

## Troubleshooting

### Camera Issues

- **No camera found**: Check `ls /dev/video*` and try different camera indices
- **Permission denied**: Run with `sudo` or add user to `video` group
- **Camera in use**: Close other applications using the camera

### Virtual Camera Issues

- **Virtual camera not appearing**: Ensure v4l2loopback is loaded: `lsmod | grep v4l2loopback`
- **Black screen in apps**: Check that Linucast is running and processing video

### Performance Issues

- **High CPU usage**: Lower the FPS or resolution in settings
- **Lag or stuttering**: Close other resource-intensive applications
- **Memory leaks**: Restart Linucast periodically (the app now includes automatic memory management)

### Desktop Integration Issues

- **Icon not appearing**: Run `./install/fix_desktop_icon.sh`
- **Wrong icon or name**: Remove all desktop files and recreate with the fix script
- **Multiple entries**: Use the desktop icon fix script to clean up

## Getting Help

1. Check the [troubleshooting guide](docs/TROUBLESHOOTING.md)
2. Look at [known issues](https://github.com/Adelkazzaz/Linucast/issues)
3. Create a new issue with:
   - Your Linux distribution and version
   - Error messages or logs
   - Steps to reproduce the problem
   - Output of `python tools/test_setup.py`
