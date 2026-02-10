/*
  Varun Raghavendra
  Spring 2026
  CS 5330 Computer Vision

  "Declarations for feature extraction routines used by Tasks 1–7."

*/

#pragma once
#include "core.h"
#include <opencv2/core.hpp>
#include <string>

namespace imgsearch {

FeatureVector extractCenterPatch7x7(const cv::Mat& bgr);
FeatureVector extractRgChromHist(const cv::Mat& bgr, int binsR, int binsG);
FeatureVector extractRgbHist(const cv::Mat& bgr, int bins);
FeatureVector extractRgbTopBottom(const cv::Mat& bgr, int bins);
FeatureVector extractColorTexture(const cv::Mat& bgr, int colorBins, int textureBins);
struct Task7Params {
  int abBins = 16; 
};

size_t task7ColorLen(const Task7Params& p);

FeatureVector extractTask7LabChromWavelet(const cv::Mat& bgr, const Task7Params& p);
FeatureVector extractResNet18EmbeddingByFilename(const std::string& imagePath, const std::string& embeddingCsvPath);

}
