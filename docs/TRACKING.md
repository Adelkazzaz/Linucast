# Linucast Simplified with Face Tracking

An enhanced version of Linucast with face tracking auto-framing feature, similar to NVIDIA Broadcast's auto-reframe or Apple Center Stage.

## Features

- **Face Tracking & Auto-framing**: Automatically centers and zooms on the detected face
  - **Dynamic Zoom Control**: Adjust zoom level while tracking is active
  - **Smooth Motion Tracking**: With adjustable smoothing factor
- **Background Effects**:
  - Blur: Apply Gaussian blur to the background
  - Remove: Replace background with black
  - Replace: Use a custom background image
- **Face Landmarks**: Optional display of facial feature points
- **FPS Control**: Toggle between 30 and 60 fps modes
- **Virtual Camera**: Output to virtual camera for use in video conferencing apps

## Requirements

- Python 3.8 or higher
- OpenCV
- MediaPipe
- NumPy
- pyvirtualcam (optional, for virtual camera output)

## Installation

```bash
# Install required packages
pip install mediapipe opencv-python numpy

# Optional: for virtual camera support
pip install pyvirtualcam
```

For virtual camera support, you'll need to set up v4l2loopback:

```bash
sudo apt install v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="LinucastCam" exclusive_caps=1
```

## Usage

Basic usage:

```bash
python linucast_simple.py
```

With face tracking enabled:

```bash
python linucast_simple.py --face-tracking
```

With custom face tracking parameters:

```bash
python linucast_simple.py --face-tracking --zoom-ratio 2.0 --smoothing 0.3
```

With high-performance settings:

```bash
python linucast_simple.py --face-tracking --landmarks --mode blur --fps 60
```

With all features:

```bash
python linucast_simple.py --face-tracking --landmarks --mode blur --virtual-cam --fps 60 --zoom-ratio 1.5
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| C | Switch camera |
| B | Blur background mode |
| R | Remove background mode |
| I | Image background mode |
| L | Toggle face landmarks display |
| T | Toggle face tracking auto-frame |
| + / - | Increase / decrease zoom (when tracking is enabled) |
| F | Toggle between 30fps and 60fps modes |
| Q or ESC | Quit |

## Command-line Options

| Option | Description |
|--------|-------------|
| --camera CAMERA | Camera index (default: 1) |
| --blur BLUR | Blur strength (default: 55) |
| --threshold THRESHOLD | Segmentation threshold (default: 0.6) |
| --landmarks | Show face landmarks |
| --resolution RESOLUTION | Output resolution (default: 640x480) |
| --virtual-cam | Output to virtual camera |
| --virtual-device DEVICE | Virtual camera device (default: /dev/video10) |
| --bg-image BG_IMAGE | Background image for replacement mode |
| --mode MODE | Background mode (blur, remove, replace) |
| --face-tracking | Enable face tracking and auto-framing |
| --zoom-ratio ZOOM_RATIO | Zoom ratio for face tracking (default: 1.8) |
| --smoothing SMOOTHING | Smoothing factor for tracking (default: 0.2) |
| --fps FPS | Target FPS (choices: 30, 60; default: 30) |

## How Face Tracking Works

The face tracking feature implements a "virtual pan-tilt-zoom" effect:

1. **Detect Face**: Uses MediaPipe Face Mesh to detect facial landmarks
2. **Calculate Center**: Finds the midpoint between the eyes
3. **Smart Tracking**: Adaptively adjusts tracking speed based on movement and FPS
4. **Face Recovery**: Maintains last known position when face detection is temporarily lost
5. **Smooth Movement**: Applies an Exponential Moving Average (EMA) filter with adaptive smoothing
6. **Crop and Zoom**: Creates a zoomed view centered on the face (with adjustable zoom level)
7. **Resize to Output**: Scales the cropped region back to full size
8. **Dynamic FPS Control**: Adjusts the processing rate between standard (30fps) and high (60fps) modes
9. **Adaptive Performance**: Balances image quality and responsiveness based on system capabilities

## Adjusting Face Tracking

- **Zoom Ratio**: Controls how much the camera zooms in (higher = closer)
  - Default: 1.8
  - Try values between 1.2 (slight zoom) and 3.0 (extreme zoom)

- **Smoothing Factor**: Controls how quickly tracking follows movement
  - Default: 0.2
  - Lower values (e.g., 0.05) result in slower, smoother tracking
  - Higher values (e.g., 0.5) result in faster, more responsive tracking

## Troubleshooting

### Face Tracking Issues

- If tracking is jittery, try decreasing the smoothing factor
- If tracking is too slow, try increasing the smoothing factor
- If the face appears cut off, try decreasing the zoom ratio
- If face tracking is lost during fast movements, the app now has improved recovery mechanisms

### Performance Issues

- Try a lower resolution: `--resolution 640x360`
- Disable face landmarks if not needed
- The app now includes adaptive frame skipping to maintain responsiveness when the CPU is heavily loaded
- For best FPS, ensure your system has sufficient resources (CPU, GPU)

### FPS Counter Shows Lower Than Expected

- The FPS counter shows the actual processing framerate, which might be lower than the target due to:
  - Complex processing operations (face detection, background segmentation)
  - System load from other applications
  - Camera hardware limitations
- The app now adaptively manages processing to maintain smooth video even when the full target FPS can't be reached

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
