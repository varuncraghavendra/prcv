# CBIR Project (CS 5330) by Varun Raghavendra 

This project implements a small **content-based image retrieval (CBIR)** system with:
- **`cbir_query`**: command-line search
- **`cbir_gui`**: a single-window **OpenCV GUI** for interactive querying (pick target, pick dataset, choose task/metric, view Top‑K)

---

## 1) Build (Ubuntu / Linux)

Install build tools + OpenCV, then compile:

```
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libopencv-dev
make clean
make
```

This produces:
- `./cbir_query`
- `./cbir_gui`

---

## 2) Run the GUI (recommended)

Place your image folder (e.g., `dataset/`) and (optional) embeddings CSV (e.g., `ResNet18_olym.csv`) in the project directory.

```
./cbir_gui --topk 5 --emb_csv ./ResNet18_olym.csv
```

### GUI keys
- `r` : run search (many UI changes auto-trigger search as well)
- `f` / `F` : next / previous **feature**
- `m` / `M` : next / previous **metric**
- `+` / `-` : increase / decrease **TopK**
- `t` : select a **target image** (uses `zenity` on Ubuntu)
- `d` : select a **dataset directory** (uses `zenity` on Ubuntu)
- `q` or `Esc` : quit

If you want the file/directory picker keys (`t`, `d`) to work, install `zenity`:

```
sudo apt-get install -y zenity
```

---

## 3) Run from the command line (CLI)

### Example: Task 4 (color + texture)
```
./cbir_query --db ./dataset --target ./dataset/pic.0535.jpg --feature colortexture --metric colortexture --topk 3
```

### Example: Task 5/6 (embeddings + cosine distance)
```
./cbir_query --db ./dataset --target ./dataset/pic.0893.jpg --feature embedding_resnet18 --metric cosine --topk 3 --emb_csv ./ResNet18_olym.csv
```

---

## 4) Features & metrics supported

**Features**
- `center7x7`
- `rg16`
- `rgb8`
- `rgb8_topbottom`
- `colortexture`
- `embedding_resnet18`
- `task7 - CIELAB Color Space Histogram`

**Metrics**
- `ssd`
- `histint`
- `multihist`
- `cosine`
- `colortexture`
- `Bhattacharya Distance`

---


