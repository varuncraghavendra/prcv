/*
  Varun Raghavendra
  Spring 2026
  CS 5330 Computer Vision

  "Search pipelines: compute features, load caches, compute distances, rank and return results."

*/

#include "core.h"
#include "project_features.h"
#include "csv_util.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

namespace imgsearch {

static FeatureVector computeClassic(const std::string& featureKey, const cv::Mat& img) {
  if (featureKey == "center7x7") return extractCenterPatch7x7(img);
  if (featureKey == "rg16") return extractRgChromHist(img, 16, 16);
  if (featureKey == "rgb8") return extractRgbHist(img, 8);
  if (featureKey == "rgb8_topbottom") return extractRgbTopBottom(img, 8);
  if (featureKey == "colortexture") return extractColorTexture(img, 8, 16);
  if (featureKey == "task7") {
    Task7Params p;
    return extractTask7LabChromWavelet(img, p);
  }
  FeatureVector out;
  out.ok = false;
  out.message = "Unknown featureKey: " + featureKey;
  return out;
}

static double computeMetric(const std::string& metricKey,
                            const std::vector<float>& a,
                            const std::vector<float>& b,
                            const std::string& featureKey) {
  (void)featureKey;
  if (metricKey == "ssd") return distanceSsd(a, b);
  if (metricKey == "histint") return distanceHistIntersection(a, b);
  if (metricKey == "cosine") return distanceCosine(a, b);
  if (metricKey == "multihist") return distanceTopBottomEqual(a, b);
  if (metricKey == "colortexture") return distanceColorTextureEqual(a, b);
  if (metricKey == "task7_bhatt") {
    Task7Params p;
    return distanceTask7Equal(a, b, task7ColorLen(p));
  }
  throw std::runtime_error("Unknown metricKey: " + metricKey);
}

static void sortAndClip(std::vector<SearchMatch>& v, int topk) {
  std::sort(v.begin(), v.end(), [](const SearchMatch& A, const SearchMatch& B){ return A.score < B.score; });
  if (topk > 0 && int(v.size()) > topk) v.resize(size_t(topk));
}

static std::string defaultCachePath(const std::string& dbDir, const std::string& featureKey) {
  std::filesystem::path p(dbDir);
  p /= (".cbir_cache_" + featureKey + ".csv");
  return p.string();
}

bool writeClassicFeaturesCsv(const std::string& dbDir,
                             const std::string& featureKey,
                             const std::string& outCsvPath) {
  const auto files = collectImageFiles(dbDir);
  int reset = 1;
  for (const auto& path : files) {
    cv::Mat img = loadBgrImage(path);
    FeatureVector feat = computeClassic(featureKey, img);
    if (!feat.ok) continue;

    const int rc = append_image_data_csv(
      const_cast<char*>(outCsvPath.c_str()),
      const_cast<char*>(path.c_str()),
      feat.values,
      reset
    );
    if (rc != 0) return false;
    reset = 0;
  }
  return true;
}

// Loads cached classic features from CSV for fast repeated search runs.

static bool loadFeatureCsv(const std::string& csvPath,
                           std::vector<std::string>& namesOut,
                           std::vector<std::vector<float>>& dataOut) {
  std::vector<char*> names;
  std::vector<std::vector<float>> data;
  const int rc = read_image_data_csv(const_cast<char*>(csvPath.c_str()), names, data, 0);
  if (rc != 0) return false;

  const size_t n = std::min(names.size(), data.size());
  namesOut.clear();
  dataOut.clear();
  namesOut.reserve(n);
  dataOut.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    namesOut.emplace_back(names[i] ? std::string(names[i]) : std::string());
    dataOut.emplace_back(std::move(data[i]));
  }
  for (char* p : names) delete[] p;
  return true;
}

std::vector<SearchMatch> searchClassic(const std::string& dbDir,
                                      const std::string& targetPath,
                                      const std::string& featureKey,
                                      const std::string& metricKey,
                                      int topk) {
  std::vector<SearchMatch> results;

  cv::Mat targetImg = loadBgrImage(targetPath);
  FeatureVector targetFeat = computeClassic(featureKey, targetImg);
  if (!targetFeat.ok) return results;

  const std::string cachePath = defaultCachePath(dbDir, featureKey);

  std::vector<std::string> names;
  std::vector<std::vector<float>> feats;

  if (!loadFeatureCsv(cachePath, names, feats)) {
    if (!writeClassicFeaturesCsv(dbDir, featureKey, cachePath)) {
      return results;
    }
    if (!loadFeatureCsv(cachePath, names, feats)) {
      return results;
    }
  }

  results.reserve(feats.size());
  for (size_t i = 0; i < feats.size(); ++i) {
    try {
      const double d = computeMetric(metricKey, targetFeat.values, feats[i], featureKey);
      results.push_back({names[i], d});
    } catch (...) {
    }
  }

  sortAndClip(results, topk);
  return results;
}

static std::unordered_map<std::string, std::vector<float>> loadEmbeddingMap(const std::string& csvPath) {
  std::unordered_map<std::string, std::vector<float>> out;
  std::vector<char*> names;
  std::vector<std::vector<float>> data;
  const int rc = read_image_data_csv(const_cast<char*>(csvPath.c_str()), names, data, 0);
  if (rc != 0) return out;

  const size_t n = std::min(names.size(), data.size());
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const std::string raw = names[i] ? std::string(names[i]) : std::string();
    const std::string key = fileBasename(raw);
    if (!key.empty()) out[key] = std::move(data[i]);
    if (names[i]) delete[] names[i];
  }
  return out;
}

std::vector<SearchMatch> searchEmbeddings(const std::string& embeddingCsvPath,
                                         const std::string& dbDir,
                                         const std::string& targetPath,
                                         const std::string& metricKey,
                                         int topk) {
  std::vector<SearchMatch> results;

  auto emb = loadEmbeddingMap(embeddingCsvPath);
  if (emb.empty()) return results;

  const std::string targetKey = fileBasename(targetPath);
  auto itT = emb.find(targetKey);
  if (itT == emb.end()) return results;
  const std::vector<float>& targetFeat = itT->second;

  const auto files = collectImageFiles(dbDir);
  results.reserve(files.size());

  for (const auto& path : files) {
    const std::string key = fileBasename(path);
    auto it = emb.find(key);
    if (it == emb.end()) continue;
    try {
      const double d = computeMetric(metricKey, targetFeat, it->second, "embedding");
      results.push_back({path, d});
    } catch (...) {
    }
  }

  sortAndClip(results, topk);
  return results;
}

// Runs Task 7 retrieval using the Task 7 feature extractor + Task 7 distance metric.

std::vector<SearchMatch> searchTask7(const std::string& dbDir,
                                    const std::string& targetPath,
                                    int topk) {
  return searchClassic(dbDir, targetPath, "task7", "task7_bhatt", topk);
}

std::vector<SearchMatch> searchFromClassicCsv(const std::string& csvPath,
                                            const std::string& targetPath,
                                            const std::string& featureKey,
                                            const std::string& metricKey) {
  std::vector<SearchMatch> results;

  cv::Mat targetImg = loadBgrImage(targetPath);
  FeatureVector targetFeat = computeClassic(featureKey, targetImg);
  if (!targetFeat.ok) return results;

  std::vector<std::string> names;
  std::vector<std::vector<float>> feats;
  if (!loadFeatureCsv(csvPath, names, feats)) return results;

  results.reserve(feats.size());
  for (size_t i = 0; i < feats.size(); ++i) {
    try {
      const double d = computeMetric(metricKey, targetFeat.values, feats[i], featureKey);
      results.push_back({names[i], d});
    } catch (...) {
    }
  }

  std::sort(results.begin(), results.end(), [](const SearchMatch& A, const SearchMatch& B){ return A.score < B.score; });
  return results;
}

} // namespace imgsearch