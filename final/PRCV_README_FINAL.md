# PRCV Hand Tracking System (MMPose + Monodepth2 + 1€ Filter)

## Overview
This project implements a real-time monocular hand tracking pipeline for robot learning applications. It combines MMPose-based keypoint detection, Monodepth2 depth estimation, and temporal filtering to produce stable and interpretable hand motion signals.

---

## Features

### Hand Tracking (MMPose)
- Uses InterNet (ResNet-50) trained on InterHand2.6M
- Detects 3D hand keypoints from webcam input
- Supports single and dual hand detection

### Stable Tracking
- Selects one dominant hand
- Tracks hand across frames using center-based matching
- ROI-based inference improves stability and performance

### 1€ Filter (Temporal Smoothing)
- Reduces jitter in keypoints
- Adaptive smoothing based on motion speed
- Maintains low latency

### Depth Estimation (Monodepth2)
- Monocular depth estimation from RGB input
- Uses ResNet18 encoder-decoder architecture
- Includes temporal smoothing for stable depth maps

### Per-Finger Depth
- Computes depth at each fingertip using local median patch
- Displays depth values in meters on video

### Gesture Recognition
- Two gesture states:
  - OPEN PALM
  - GRASP
- Based on finger extension and curl metrics
- Includes temporal smoothing to avoid flicker

### Real-Time Visualization
- Hand skeleton overlay
- Gesture label
- FPS display
- Depth map (bottom-left inset)
- Finger depth annotations

---

## Project Structure

```
robot_learning_hand_demo/
│
├── src/
│   ├── camera.py
│   ├── pose_backends.py
│   ├── one_euro_filter.py
│   ├── gesture_abstraction.py
│   ├── monodepth2_networks.py
│   ├── depth_estimator.py
│   └── pipeline.py
│
├── scripts/
│   └── run_robot_learning_gui.py
```

---

## Setup Instructions

### 1. Create workspace
```bash
cd ~
rm -rf prcv
mkdir -p prcv
cd prcv
```

### 2. Clone MMPose
```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
```

### 3. Add project
```bash
unzip prcv_mmpose_full_project_improved.zip
mv robot_learning_hand_demo projects/
```

### 4. Activate environment
```bash
source ~/venvs/mmpose-hand3d/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PWD:$PWD/projects/robot_learning_hand_demo:$PYTHONPATH
```

### 5. Install dependencies
```bash
pip install numpy==1.26.4 setuptools<82 packaging<25
pip install opencv-python scipy matplotlib cython munkres
```

### 6. Fix xtcocotools
```bash
pip uninstall -y xtcocotools
cd /tmp
git clone https://github.com/jin-s13/xtcocoapi.git
cd xtcocoapi
python setup.py build_ext --inplace
pip install . --no-build-isolation --no-deps
cd ~/prcv/mmpose
```

### 7. Download MMPose model
```bash
mkdir -p checkpoints
wget -O checkpoints/res50.pth https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth
```

### 8. Download Monodepth2 model
```bash
mkdir -p projects/robot_learning_hand_demo/checkpoints
cd projects/robot_learning_hand_demo/checkpoints

wget -O mono_640x192.zip https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip

unzip mono_640x192.zip
mkdir mono_640x192
mv encoder.pth mono_640x192/
mv depth.pth mono_640x192/
```

---

## Run

### Full pipeline
```bash
python projects/robot_learning_hand_demo/scripts/run_robot_learning_gui.py   --device cpu   --score-thr 0.08   --depth-model projects/robot_learning_hand_demo/checkpoints/mono_640x192   --infer-scale 0.55
```

---

## Notes

- Works best with:
  - good lighting
  - plain background
  - single hand in frame

- Monocular depth is approximate (not true metric depth)

- CPU performance may be limited; GPU recommended for real-time usage

---

## Summary

This system integrates MMPose, Monodepth2, and a 1€ filter to convert raw video input into stable hand tracking, gesture recognition, and depth-aware features suitable for robot learning applications.
