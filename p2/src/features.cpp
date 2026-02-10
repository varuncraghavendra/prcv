/*
  Varun Raghavendra
  Spring 2026
  CS 5330 Computer Vision

  "Feature extraction implementations for Tasks 1–7"

*/

#include "project_features.h"
#include "csv_util.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <mutex>
#include <memory>

namespace imgsearch {

// L1-normalize a histogram so it sums to 1 (required for probabilistic histogram distances).

static void l1Normalize(std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += x;
  if (s <= 0.0) return;
  const float inv = float(1.0 / s);
  for (float& x : v) x *= inv;
}

FeatureVector extractCenterPatch7x7(const cv::Mat& bgr) {
  FeatureVector out;
  if (bgr.empty() || bgr.channels() != 3) { out.ok = false; out.message = "Invalid image."; return out; }
  const int rows = bgr.rows;
  const int cols = bgr.cols;
  const int cx = cols / 2;
  const int cy = rows / 2;
  const int x0 = cx - 3, y0 = cy - 3, x1 = cx + 3, y1 = cy + 3;
  if (x0 < 0 || y0 < 0 || x1 >= cols || y1 >= rows) { out.ok = false; out.message = "Image too small."; return out; }
  out.values.reserve(7 * 7 * 3);
  for (int y = y0; y <= y1; ++y) {
    const cv::Vec3b* rowPtr = bgr.ptr<cv::Vec3b>(y);
    for (int x = x0; x <= x1; ++x) {
      const cv::Vec3b px = rowPtr[x];
      out.values.push_back(float(px[0]));
      out.values.push_back(float(px[1]));
      out.values.push_back(float(px[2]));
    }
  }
  return out;
}

FeatureVector extractRgChromHist(const cv::Mat& bgr, int binsR, int binsG) {
  FeatureVector out;
  if (bgr.empty() || bgr.channels() != 3) { out.ok = false; out.message = "Invalid image."; return out; }
  if (binsR <= 0 || binsG <= 0) { out.ok = false; out.message = "Invalid bins."; return out; }
  std::vector<float> hist(size_t(binsR) * size_t(binsG), 0.0f);
  for (int y = 0; y < bgr.rows; ++y) {
    const cv::Vec3b* rowPtr = bgr.ptr<cv::Vec3b>(y);
    for (int x = 0; x < bgr.cols; ++x) {
      const cv::Vec3b px = rowPtr[x];
      const float B = float(px[0]);
      const float G = float(px[1]);
      const float R = float(px[2]);
      const float sum = R + G + B;
      if (sum <= 1e-6f) continue;
      const float rr = R / sum;
      const float gg = G / sum;
      int br = int(rr * binsR);
      int bg = int(gg * binsG);
      if (br < 0) br = 0;
      if (bg < 0) bg = 0;
      if (br >= binsR) br = binsR - 1;
      if (bg >= binsG) bg = binsG - 1;
      hist[size_t(bg) * size_t(binsR) + size_t(br)] += 1.0f;
    }
  }
  l1Normalize(hist);
  out.values = std::move(hist);
  return out;
}

// Whole-image RGB histogram feature (OpenCV BGR input, converted per bin logic).

FeatureVector extractRgbHist(const cv::Mat& bgr, int bins) {
  FeatureVector out;
  if (bgr.empty() || bgr.channels() != 3) { out.ok = false; out.message = "Invalid image."; return out; }
  if (bins <= 0) { out.ok = false; out.message = "Invalid bins."; return out; }
  std::vector<float> hist(size_t(bins) * size_t(bins) * size_t(bins), 0.0f);
  auto binOf = [bins](int v) {
    int b = int((float(v) / 256.0f) * bins);
    if (b < 0) b = 0;
    if (b >= bins) b = bins - 1;
    return b;
  };
  for (int y = 0; y < bgr.rows; ++y) {
    const cv::Vec3b* rowPtr = bgr.ptr<cv::Vec3b>(y);
    for (int x = 0; x < bgr.cols; ++x) {
      const cv::Vec3b px = rowPtr[x];
      const int bb = binOf(px[0]);
      const int bg = binOf(px[1]);
      const int br = binOf(px[2]);
      const size_t idx = (size_t(bb) * size_t(bins) + size_t(bg)) * size_t(bins) + size_t(br);
      hist[idx] += 1.0f;
    }
  }
  l1Normalize(hist);
  out.values = std::move(hist);
  return out;
}

FeatureVector extractRgbTopBottom(const cv::Mat& bgr, int bins) {
  FeatureVector out;
  if (bgr.empty() || bgr.channels() != 3) { out.ok = false; out.message = "Invalid image."; return out; }
  const int mid = bgr.rows / 2;
  cv::Mat top = bgr.rowRange(0, std::max(1, mid));
  cv::Mat bottom = bgr.rowRange(std::max(0, mid), bgr.rows);
  FeatureVector fTop = extractRgbHist(top, bins);
  if (!fTop.ok) return fTop;
  FeatureVector fBot = extractRgbHist(bottom, bins);
  if (!fBot.ok) return fBot;
  out.values.reserve(fTop.values.size() + fBot.values.size());
  out.values.insert(out.values.end(), fTop.values.begin(), fTop.values.end());
  out.values.insert(out.values.end(), fBot.values.begin(), fBot.values.end());
  return out;
}

// Texture histogram: Sobel gradient magnitude binned to capture edge/texture strength.

static std::vector<float> sobelMagnitudeHist(const cv::Mat& bgr, int bins) {
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::Mat gx, gy;
  cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
  cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
  cv::Mat mag;
  cv::magnitude(gx, gy, mag);
  double minV = 0.0, maxV = 0.0;
  cv::minMaxLoc(mag, &minV, &maxV);
  const float denom = float(std::max(maxV, 1e-6));
  std::vector<float> hist(size_t(bins), 0.0f);
  for (int y = 0; y < mag.rows; ++y) {
    const float* rowPtr = mag.ptr<float>(y);
    for (int x = 0; x < mag.cols; ++x) {
      const float m = rowPtr[x];
      int b = int((m / denom) * bins);
      if (b < 0) b = 0;
      if (b >= bins) b = bins - 1;
      hist[size_t(b)] += 1.0f;
    }
  }
  l1Normalize(hist);
  return hist;
}

// Concatenate color histogram + texture histogram (equal weights in distance).

FeatureVector extractColorTexture(const cv::Mat& bgr, int colorBins, int textureBins) {
  FeatureVector out;
  if (bgr.empty() || bgr.channels() != 3) { out.ok = false; out.message = "Invalid image."; return out; }
  FeatureVector color = extractRgbHist(bgr, colorBins);
  if (!color.ok) return color;
  std::vector<float> texture = sobelMagnitudeHist(bgr, textureBins);
  out.values.reserve(color.values.size() + texture.size());
  out.values.insert(out.values.end(), color.values.begin(), color.values.end());
  out.values.insert(out.values.end(), texture.begin(), texture.end());
  return out;
}

size_t task7ColorLen(const Task7Params& p) {
  return size_t(p.abBins) * size_t(p.abBins);
}

// Lab chroma histogram over a/b channels 

static std::vector<float> labChromaHistAB(const cv::Mat& bgr, int abBins) {
  cv::Mat lab;
  cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
  // Light blur to reduce sensor noise / JPEG blocking before histogramming.
  cv::GaussianBlur(lab, lab, cv::Size(3,3), 0.0);
  std::vector<float> hist(size_t(abBins) * size_t(abBins), 0.0f);

  for (int y = 0; y < lab.rows; ++y) {
    const cv::Vec3b* row = lab.ptr<cv::Vec3b>(y);
    for (int x = 0; x < lab.cols; ++x) {
      const int a = row[x][1];
      const int b = row[x][2];
      int ba = (a * abBins) / 256;
      int bb = (b * abBins) / 256;
      if (ba >= abBins) ba = abBins - 1;
      if (bb >= abBins) bb = abBins - 1;
      hist[size_t(bb) * size_t(abBins) + size_t(ba)] += 1.0f;
    }
  }
  l1Normalize(hist);
  return hist;
}

// Lab chroma + additional texture cues.

FeatureVector extractTask7LabChromWavelet(const cv::Mat& bgr, const Task7Params& p) {
  FeatureVector out;
  if (bgr.empty() || bgr.channels() != 3) { out.ok = false; out.message = "Invalid image."; return out; }
  const auto color = labChromaHistAB(bgr, p.abBins);
  out.values = color;
  out.ok = true;
  out.message.clear();
  return out;
}


class EmbeddingStore {
public:
  bool load(const std::string& csvPath, std::string& err) {
    std::vector<char*> names;
    std::vector<std::vector<float>> data;
    if (read_image_data_csv(const_cast<char*>(csvPath.c_str()), names, data, 0) != 0) {
      err = "Failed to read embedding CSV.";
      return false;
    }

    map_.clear();
    map_.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      const std::string rawKey = names[i] ? std::string(names[i]) : std::string();
      const std::string baseKey = fileBasename(rawKey);
      map_[baseKey] = std::move(data[i]);
    }

    for (char* p : names) delete[] p;
    path_ = csvPath;
    return true;
  }


  const std::vector<float>* findByBasename(const std::string& imgPath) const {
    const std::string key = fileBasename(imgPath);
    auto it = map_.find(key);
    if (it == map_.end()) return nullptr;
    return &it->second;
  }

  const std::string& csvPath() const { return path_; }

private:
  std::string path_;
  std::unordered_map<std::string, std::vector<float>> map_;
};

static std::shared_ptr<EmbeddingStore> getEmbeddingStore(const std::string& csvPath, std::string& err) {
  static std::mutex m;
  static std::unordered_map<std::string, std::weak_ptr<EmbeddingStore>> cache;
  std::lock_guard<std::mutex> lock(m);
  auto it = cache.find(csvPath);
  if (it != cache.end()) {
    auto sp = it->second.lock();
    if (sp) return sp;
  }
  auto store = std::make_shared<EmbeddingStore>();
  if (!store->load(csvPath, err)) return nullptr;
  cache[csvPath] = store;
  return store;
}

FeatureVector extractResNet18EmbeddingByFilename(const std::string& imagePath, const std::string& embeddingCsvPath) {
  FeatureVector out;
  std::string err;
  auto store = getEmbeddingStore(embeddingCsvPath, err);
  if (!store) { out.ok = false; out.message = err; return out; }
  const auto* v = store->findByBasename(imagePath);
  if (!v) { out.ok = false; out.message = "No embedding found for: " + fileBasename(imagePath); return out; }
  out.values = *v;
  return out;
}

}
