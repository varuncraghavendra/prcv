# OpenCV Real-Time Video Filters with Depth Anything V2

This project is a **C++ / OpenCV4 real-time video processing application** that demonstrates classic image filters, optimized convolution, face detection, and **monocular depth estimation using Depth Anything V2 (ONNX Runtime)**.  
It is designed for interactive experimentation via keyboard controls and trackbars.

---

## ✨ Features

### Core Video App
- Live camera capture using OpenCV
- Real-time keyboard-controlled filter switching
- Adjustable **brightness** and **contrast** via trackbars
- Optional face detection overlay
- Frame saving to disk

### Image Filters
Implemented **from scratch** (no OpenCV shortcuts unless noted):

- Grayscale
- Sepia
- Negative
- Brightness / contrast adjustment
- 5×5 Gaussian blur (two versions)
  - `blur5x5_1`: reference implementation
  - `blur5x5_2`: optimized separable filter
- Blur + quantization (posterization)
- Sobel X / Y gradients (signed)
- Gradient magnitude
- Emboss effect using Sobel dot-product shading

### Depth Anything V2 Integration
- ONNX Runtime–based inference
- Depth map generation (8‑bit visualization)
- Separate depth thread for performance
- Face-to-camera distance estimation (cm)
- **Depth-based fog effect** (distance-aware atmospheric fade)

---

## 🧠 Project Structure

```
.
├── vidDisplay.cpp      # Main application (camera loop, UI, threading)
├── filter.h            # Declarations for all filters + depth helpers
├── filter.cpp          # Filter implementations + DA2 integration
├── faceDetect.h/.cpp   # Face detection helpers (Haar / DNN)
├── README.md
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|----:|-------|
| `q` | Quit |
| `c` | Color (original) |
| `g` | Grayscale |
| `p` | Sepia |
| `n` | Negative |
| `b` | Blur (5×5 optimized) |
| `l` | Blur + quantize |
| `x` | Sobel X |
| `y` | Sobel Y |
| `m` | Gradient magnitude |
| `e` | Toggle emboss |
| `f` | Toggle face detection |
| `d` | Toggle depth estimation |
| `j` | Toggle depth-based fog |
| `s` | Save current frame |
| `v` | Run blur timing test |

---

## 🎚️ Trackbars

- **Brightness**: range `[-255, +255]`
- **Contrast**: range `[0.20, 3.00]`

---

## 🔧 Dependencies

### Required
- **C++17**
- **OpenCV 4.x**
- **ONNX Runtime (CPU)**  
  Used for Depth Anything V2 inference

### Optional
- Webcam (USB / integrated)
- Pretrained **Depth Anything V2 ONNX model**

---

## 🏗️ Build Instructions

### 1. Install OpenCV 4
```bash
sudo apt install libopencv-dev
```

### 2. Install ONNX Runtime (CPU)
Download prebuilt binaries from:  
https://onnxruntime.ai/

Example:
```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

### 3. Compile
```bash
g++ -std=c++17 -O2 \
    vidDisplay.cpp filter.cpp faceDetect.cpp \
    -o vidDisplay \
    `pkg-config --cflags --libs opencv4` \
    -lonnxruntime
```

---

## ▶️ Running

```bash
./vidDisplay
```

Make sure:
- The camera is accessible
- The Depth Anything V2 model path is valid (loaded once at runtime)

---

## ⚙️ Implementation Notes

- **Threaded design**: depth inference runs independently from rendering
- **Performance-aware**: separable blur and pointer-based pixel access
- **Numerical safety**: signed Sobel buffers (`CV_16SC3`) before magnitude
- **No in-place corruption**: all filters preserve source image integrity

---

## 📌 Educational Focus

This project emphasizes:
- Real-time image processing fundamentals
- Manual convolution and optimization
- Depth-from-single-image pipelines
- Clean OpenCV + ONNX Runtime integration
- UI-driven experimentation

---

## 📜 License

Educational / academic use.  
Depth Anything V2 model license applies separately.

---

## 🙌 Acknowledgements

- OpenCV
- Microsoft ONNX Runtime
- Depth Anything V2 authors

