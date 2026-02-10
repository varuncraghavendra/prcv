/*
  Varun Raghavendra
  Spring 2026
  CS 5330 Computer Vision

  "Utility helpers for paths, image discovery, and loading images with OpenCV."

*/

#include "core.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

namespace imgsearch {

static std::string lowerCopy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string fileBasename(const std::string& path) {
  const auto pos = path.find_last_of("/\\");
  return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

bool isImageFile(const std::string& path) {
  const std::string p = lowerCopy(path);
  return p.size() >= 4 && (p.ends_with(".jpg") || p.ends_with(".jpeg") || p.ends_with(".png") || p.ends_with(".bmp") || p.ends_with(".tif") || p.ends_with(".tiff"));
}

// Recursively collects image files from a directory (filters by extension).

std::vector<std::string> collectImageFiles(const std::string& dir) {
  std::vector<std::string> out;
  std::error_code ec;
  if (!fs::exists(dir, ec)) return out;
  for (const auto& entry : fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec)) {
    if (ec) break;
    if (!entry.is_regular_file(ec)) continue;
    const auto path = entry.path().string();
    if (isImageFile(path)) out.push_back(path);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// Loads an image as BGR (OpenCV default), returning empty Mat on failure.

cv::Mat loadBgrImage(const std::string& path) {
  return cv::imread(path, cv::IMREAD_COLOR);
}

}