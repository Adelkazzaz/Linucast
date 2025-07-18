# Linucast Usage Guide

This guide provides instructions for using Linucast's face tracking and background effects.

## Basic Usage

To start Linucast with default settings:

```bash
python linucast_simple.py
```

## Face Tracking

To enable face tracking:

```bash
python linucast_simple.py --face-tracking
```

### Face Tracking Controls

- **Toggle Tracking**: Press `T` to turn tracking on/off
- **Zoom In/Out**: Press `+`/`-` to adjust zoom level (only active in tracking mode)
- **Toggle Face Landmarks**: Press `L` to show/hide facial landmarks
- **FPS Mode**: Press `F` to toggle between 30fps and 60fps

### Adjusting Face Tracking Parameters

For smoother or more responsive tracking:

```bash
# For smoother tracking (less jittery)
python linucast_simple.py --face-tracking --smoothing 0.1

# For more responsive tracking (follows movement faster)
python linucast_simple.py --face-tracking --smoothing 0.3

# For tighter framing (zoomed in more)
python linucast_simple.py --face-tracking --zoom-ratio 2.2
```

## Background Effects

### Background Blur

```bash
python linucast_simple.py --mode blur --blur 75
```

- Higher blur value gives a stronger effect
- Press `B` to switch to blur mode during runtime

### Background Removal

```bash
python linucast_simple.py --mode remove
```

- This replaces the background with black
- Press `R` to switch to removal mode during runtime

### Background Replacement

```bash
python linucast_simple.py --mode replace --bg-image path/to/image.jpg
```

- Replace the background with a custom image
- Press `I` to switch to image replacement mode during runtime

## Virtual Camera Output

To output to a virtual camera for use in video conferencing:

```bash
python linucast_simple.py --face-tracking --virtual-cam
```

- Ensure v4l2loopback is properly set up (see Installation Guide)
- In your video conferencing app, select "LinucastCam" as your camera

## Performance Tips

- Lower resolution for better performance: `--resolution 640x360`
- Disable landmarks display for better performance: `--no-landmarks`
- If using background effects with face tracking is too slow, consider using just one feature
- For maximum performance, use 30fps mode instead of 60fps
