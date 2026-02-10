/*
  Varun Raghavendra
  Spring 2026
  CS 5330 Computer Vision

  "API and shared structs used across feature extraction, distance metrics, and search."

*/

#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace imgsearch {

struct FeatureVector {
  bool ok = true;
  std::string message;
  std::vector<float> values;
};

struct SearchMatch {
  std::string path;
  double score = 0.0;
};

std::string fileBasename(const std::string& path);
bool isImageFile(const std::string& path);
std::vector<std::string> collectImageFiles(const std::string& dir);
cv::Mat loadBgrImage(const std::string& path);

double distanceSsd(const std::vector<float>& a, const std::vector<float>& b);
double distanceHistIntersection(const std::vector<float>& a, const std::vector<float>& b);
double distanceCosine(const std::vector<float>& a, const std::vector<float>& b);
double distanceBhattacharyya(const std::vector<float>& p, const std::vector<float>& q);

double distanceTopBottomEqual(const std::vector<float>& target, const std::vector<float>& item);
double distanceColorTextureEqual(const std::vector<float>& target, const std::vector<float>& item);
double distanceTask7Equal(const std::vector<float>& target, const std::vector<float>& item, size_t colorLen);

std::vector<SearchMatch> searchClassic(
  const std::string& dbDir,
  const std::string& targetPath,
  const std::string& featureKey,
  const std::string& metricKey,
  int topk = -1
);

std::vector<SearchMatch> searchEmbeddings(
  const std::string& embeddingCsvPath,
  const std::string& dbDir,
  const std::string& targetPath,
  const std::string& metricKey,
  int topk = -1
);

std::vector<SearchMatch> searchTask7(
  const std::string& dbDir,
  const std::string& targetPath,
  int topk = -1
);


bool writeClassicFeaturesCsv(
  const std::string& dbDir,
  const std::string& featureKey,
  const std::string& outCsvPath
);

std::vector<SearchMatch> searchFromClassicCsv(
  const std::string& csvPath,
  const std::string& targetPath,
  const std::string& featureKey,
  const std::string& metricKey
);

} 
