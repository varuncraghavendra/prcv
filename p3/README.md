# 2D Object Recognition
Name : Varun Raghavendra

Demo video link : drive.google.com/file/d/1mP8NOJslM2k3c_ghmlZD29sO3cE9YILR/view?usp=sharing

## Folder structure

```
project_3/
├── main.cpp
├── include/
│   ├── image_processing.hpp
│   ├── object_recognition.hpp
│   └── utilities.hpp
├── src/
│   ├── image_processing.cpp
│   ├── object_recognition.cpp
│   └── utilities.cpp
├── data/
│   ├── train/          ← training crops saved by pressing t
│   ├── test/           ← test crops saved by pressing x
│   ├── database_normal.csv
│   └── database_eigen.csv
├── models/
│   └── eigenspace.yml
├── output/             ← screenshots saved by pressing p
└── README.md
```

---

## Build

```bash
sudo apt update
sudo apt install libopencv-dev pkg-config

g++ -std=c++17 \
    main.cpp src/image_processing.cpp src/object_recognition.cpp src/utilities.cpp \
    $(pkg-config --cflags --libs opencv4) \
    -I./include \
    -o object_recognition

./object_recognition
```

---

## Controls

| Key | Action |
|-----|--------|
| `1` | Threshold view |
| `2` | Cleaned mask view |
| `3` | Region map view (stable colors) |
| `4` | Annotated color view (default) |
| `t` | Save current object as **training** data + rebuild eigenspace |
| `x` | Save current object as **test** data |
| `f` | Toggle Normal ↔ Eigenspace recognition mode |
| `m` | Print confusion matrix for the current mode |
| `p` | Save current view to `output/` |
| `q` | Quit |

---

### IDE 

Project built on Ubuntu 22.04, all code compiled using GNU C++ compiler (g++) through the terminal command line. 
