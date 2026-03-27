# Camera Calibration and Augmented Reality

PRCV Spring 2026 — Project 4  
Varun Raghavendra

---

## Overview

This project contains three C++ programs that together build a complete camera calibration and augmented reality pipeline validated using my laptop webcam and a 9x6 standard checkerboard printed on an A4 sheet.

Key features of the project:

- detect checkerboard corners from a live webcam feed
- calibrate the camera and save its parameters
- estimate the checkerboard pose in real time
- place simple augmented reality graphics on the board, in my case it is the stick figures of a boy and girl
- detect Harris corner features with an adjustable threshold

---

## Programs

### `calibrate_camera`
This program is used to collect checkerboard views and calibrate the camera.

### `pose_and_augment`
This program is used to estimate the checkerboard pose and draw AR overlays on top of it.

### `detect_features`
This program is used to detect Harris corners in a live webcam stream.

---

## System Requirements

- CMake 3.10 or newer
- OpenCV 4.2 or newer
- a C++14 compiler
- a webcam
- a printed **9 x 6 internal-corner checkerboard**

For my project, I implemented everything in Ubuntu 22.04 OS

---

## Build Instructions

```bash
cd CameraCalibrationAR
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

After building, the executables will be inside the `build` folder.

---

## How to Run

### 1. Calibrate the camera

Run:

```bash
./calibrate_camera
```

What to do:

- Show the checkerboard clearly to the webcam
- When the corners are detected, press `s` to save that view
- Move the board to different angles and distances, then save more views
- Try to collect at least 5 good frames, I collected in the range of 7 to 11 for least RMS reprojection error
- Press `k` to run calibration
- Press `w` to save the result as `calibration.yml`, rerun calibration if not accessible
- Press `q` or `ESC` to quit


### 2. Run pose estimation and AR overlay

Run:

```bash
./pose_and_augment calibration.yml
```

This program detects the checkerboard, estimates its pose, and draws:

- coordinate axes
- outer checkerboard corner points
- two stick-figure characters standing on the board

As the board moves, the overlays move with it in real time.

#### Controls

- `a` — show or hide axes
- `c` — show or hide corner points
- `q` or `ESC` — quit

The terminal also prints the rotation and translation values for each frame.

---

### 3. Run Harris feature detection

Run:

```bash
./detect_features
```

This opens a live video window with a **Threshold %** slider.

- Red dots show the detected Harris corners
- Lower threshold values show more feature points
- Higher threshold values keep only the strongest corners

Press `q` to quit.

---

## Project Structure

```text
CameraCalibrationAR/
├── CMakeLists.txt
├── README.md
└── src/
    ├── calibrate_camera.cpp
    ├── pose_and_augment.cpp
    └── detect_features.cpp
```

The `calibration.yml` file is saved in the folder where we run `calibrate_camera`. Use that same file when running `pose_and_augment`.

---

## Helpful Tips

- Use good lighting and avoid reflections on the checkerboard
- Keep the full board visible when saving calibration images
- Capture the board from different angles, not just straight in front
- If calibration error is high, repeat the process with better and more varied views
- If the AR drawing looks misaligned, make sure to use the correct `calibration.yml`

---
