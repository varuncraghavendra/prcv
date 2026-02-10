/*
  Varun Raghavendra
  Spring 2026
  CS 5330 Computer Vision

  "Distance metrics used by Tasks 1–7 (SSD, histogram intersection, cosine, Bhattacharyya)."

*/

#include "core.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace imgsearch {

// -------------------- helpers (internal) --------------------

static inline void requireSameSize(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) throw std::runtime_error("Distance: vector size mismatch.");
  if (a.empty()) throw std::runtime_error("Distance: empty feature vector.");
}

static inline double dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) s += double(a[i]) * double(b[i]);
  return s;
}

static inline double l2Norm(const std::vector<float>& a) {
  double s = 0.0;
  for (float v : a) s += double(v) * double(v);
  return std::sqrt(s);
}

// Histogram intersection similarity: sum_i min(a_i, b_i). Larger means more similar.

static inline double histogramIntersection(const std::vector<float>& a, const std::vector<float>& b,
                                           size_t start, size_t len) {
  double s = 0.0;
  for (size_t i = 0; i < len; ++i) {
    const size_t idx = start + i;
    s += std::min(double(a[idx]), double(b[idx]));
  }
  return s;
}

static inline double distanceHistIntersectionSlice(const std::vector<float>& a, const std::vector<float>& b,
                                                   size_t start, size_t len) {
  // Hist intersection similarity -> convert to distance
  // D = 1 - sum(min(.))
  return 1.0 - histogramIntersection(a, b, start, len);
}

// Equal-weight distance across two concatenated histograms (e.g., color + texture).

static inline double distanceEqualHalfIntersection(const std::vector<float>& a, const std::vector<float>& b) {
  requireSameSize(a, b);
  const size_t half = a.size() / 2;
  if (half * 2 != a.size()) throw std::runtime_error("Expected even-length vector for 2-part feature.");
  const double d1 = distanceHistIntersectionSlice(a, b, 0, half);
  const double d2 = distanceHistIntersectionSlice(a, b, half, half);
  return 0.5 * d1 + 0.5 * d2;
}

// -------------------- public distances --------------------

// Task 1: Center-patch baseline (raw pixels). Use SSD / L2^2.
double distanceSsd(const std::vector<float>& a, const std::vector<float>& b) {
  requireSameSize(a, b);
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = double(a[i]) - double(b[i]);
    s += d * d;
  }
  return s;
}

// Task 2: Chromaticity histogram (r,g). Use histogram intersection distance.
double distanceHistIntersection(const std::vector<float>& a, const std::vector<float>& b) {
  requireSameSize(a, b);
  return distanceHistIntersectionSlice(a, b, 0, a.size());
}

// Task 3: Top/Bottom RGB hist concatenation. Use equal-weight intersection on halves.
double distanceTopBottomEqual(const std::vector<float>& target, const std::vector<float>& item) {
  return distanceEqualHalfIntersection(target, item);
}

// Task 4: Color+Texture concatenation. Use equal-weight intersection on halves.
double distanceColorTextureEqual(const std::vector<float>& target, const std::vector<float>& item) {
  return distanceEqualHalfIntersection(target, item);
}

// Task 5 & 6: DNN embeddings (CSV) and cached retrieval. Use cosine distance only.
double distanceCosine(const std::vector<float>& a, const std::vector<float>& b) {
  requireSameSize(a, b);
  const double na = l2Norm(a);
  const double nb = l2Norm(b);
  if (na <= 1e-12 || nb <= 1e-12) return 1.0; // treat degenerate vectors as maximally distant
  const double cosSim = dotProduct(a, b) / (na * nb);
  // Numerical safety: clamp to [-1, 1]
  const double c = std::max(-1.0, std::min(1.0, cosSim));
  return 1.0 - c;
}

// Bhattacharyya distance for non-negative, L1-normalized histograms.
// BC = sum_i sqrt(p_i q_i), D = sqrt(max(0, 1 - BC))
double distanceBhattacharyya(const std::vector<float>& p, const std::vector<float>& q) {
  requireSameSize(p, q);
  double bc = 0.0;
  for (size_t i = 0; i < p.size(); ++i) {
    const double a = std::max(0.0, double(p[i]));
    const double b = std::max(0.0, double(q[i]));
    bc += std::sqrt(a * b);
  }
  bc = std::min(1.0, std::max(0.0, bc));
  return 1.0 - bc;
}

// Task 7: equal weight of Lab chromaticity histogram and wavelet texture histogram.
double distanceTask7Equal(const std::vector<float>& target, const std::vector<float>& item, size_t /*colorLen*/) {
  // Task 7 is ONLY a Lab (a*, b*) chromaticity histogram. Compare full vectors.
  requireSameSize(target, item);
  return distanceBhattacharyya(target, item);
}

} // namespace imgsearch
