# Gesture Control with MediaPipe & OpenCV

This is a Python project that uses hand gestures detected via MediaPipe to trigger visual effects and zoom operations in real time.

## Features
- Detects number of fingers shown (0 to 4)
- Applies corresponding effects:
  - 0: Zoom in
  - 1: Zoom out
  - 2: Grayscale filter
  - 3: Blur
  - 4: Invert colors

## Requirements
- Python 3.7+
- OpenCV
- MediaPipe

## Run
```bash
pip install -r requirements.txt
python gesture_control.py
