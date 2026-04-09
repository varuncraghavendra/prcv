# 🖐️ MMPose 3D Hand Tracking (Ubuntu 22.04 Stable Setup)

This repository provides a fully working, reproducible setup for running the MMPose 3D hand tracking demo on Ubuntu 22.04 with Python 3.10.

## 🚀 Run Demo

python demo/hand3d_internet_demo.py \
  configs/hand_3d_keypoint/internet/interhand3d/internet_res50_4xb16-20e_interhand3d-256x256.py \
  checkpoints/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth \
  --input webcam \
  --output-root vis_results/hand3d_demo \
  --show \
  --device cpu

---

## ⚙️ Setup

### 1. Clone

git clone https://github.com/open-mmlab/mmpose.git
cd mmpose

### 2. Create Environment

python3 -m venv ~/venvs/mmpose-hand3d
source ~/venvs/mmpose-hand3d/bin/activate
export PYTHONNOUSERSITE=1

### 3. Fix Packaging

pip install "pip<25.3"
pip install "setuptools<82"
pip install "packaging<25"
pip install wheel

### 4. Install Core

pip install "numpy==1.26.4"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"

### 5. Runtime Deps

pip install json-tricks opencv-python scipy "matplotlib==3.8.4" cython munkres

### 6. Install MMPose

pip install -e . --no-deps

### 7. Install MMDetection

mim install "mmdet>=3.1.0,<4.0.0"

### 8. Fix xtcocotools

pip uninstall -y xtcocotools

cd /tmp
git clone https://github.com/jin-s13/xtcocoapi.git
cd xtcocoapi

pip install cython
python setup.py build_ext --inplace
pip install . --no-build-isolation --no-deps

### 9. Download Checkpoint

mkdir -p checkpoints

wget -O checkpoints/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth \
https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth

---

## ✅ Verify

python - <<'PY'
import numpy, torch, mmcv, mmengine, mmpose, xtcocotools
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("mmcv:", mmcv.__version__)
print("mmpose:", mmpose.__file__)
print("xtcocotools OK")
PY

Expected:
numpy == 1.26.4

---

## ⚡ Notes

- DO NOT use: pip install mmcv
- Always use: mim install mmcv
- NumPy must be <2
- setuptools must be <82

---

## 🔥 Reset

rm -rf ~/venvs/mmpose-hand3d
