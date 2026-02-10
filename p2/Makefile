#
#  Varun Raghavendra
#  Spring 2026
#  CS 5330 Computer Vision
#
#  Build rules for CBIR CLI + OpenCV GUI (Makefile-only project).
#
CXX ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wno-deprecated-enum-enum-conversion

OPENCV_PKG := $(shell pkg-config --exists opencv4 && echo 1 || echo 0)
ifeq ($(OPENCV_PKG),1)
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || echo -I/usr/include/opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4 2>/dev/null || echo -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui)
else
OPENCV_CFLAGS := -I/usr/include/opencv4
OPENCV_LIBS   := -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_features2d -lopencv_videoio
endif

INCLUDES := -Iinclude $(OPENCV_CFLAGS)
LDFLAGS  := $(OPENCV_LIBS) -pthread

CORE_OBJS := main.o src/utils.o src/distances.o src/features.o src/search.o src/csv_util.o
GUI_OBJS := src/gui_main.o src/utils.o src/distances.o src/features.o src/search.o src/csv_util.o

.PHONY: all clean

all: cbir_query cbir_gui

cbir_query: $(CORE_OBJS)
	$(CXX) $(CORE_OBJS) -o $@ $(LDFLAGS)

cbir_gui: $(GUI_OBJS)
	$(CXX) $(GUI_OBJS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f main.o src/*.o cbir_query cbir_gui
