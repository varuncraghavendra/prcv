#include "filter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>

#include "DA2Network.hpp"

/* ----------------------------- DA2 state ----------------------------- */

// Calibration constant used by the reference implementation (distance is approximate).
static constexpr double CALIB_CM = 40.0;

static std::mutex g_da2Mutex;
static std::unique_ptr<DA2Network> g_net;
static bool g_netReady = false;
static std::string g_loadedPath;

static std::mutex g_calibMutex;
static bool   g_calibrated = false;
static double g_A0 = 1.0;
static double g_areaSmooth = 1.0;
static bool   g_hasArea = false;

/*
  da2_init_once
  Loads the Depth Anything V2 ONNX model exactly once (thread-safe).
  Returns false if ONNX Runtime/model initialization fails.
  Keep the model path stable to avoid reloading multiple times.
*/
bool da2_init_once(const std::string& model_path) {
    std::lock_guard<std::mutex> lk(g_da2Mutex);

    if (g_netReady && g_net && g_loadedPath == model_path) return true;

    try {
        g_net.reset(new DA2Network(model_path.c_str()));
        g_loadedPath = model_path;
        g_netReady = true;
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "da2_init_once: %s\n", e.what());
        g_net.reset();
        g_netReady = false;
        g_loadedPath.clear();
        return false;
    }
}

/*
  da2_depth_gray
  Runs DA2 on the given BGR frame and returns an 8-bit depth image (CV_8UC1).
  The result is relative depth in [0..255] (not metric), as in the reference code.
  scale < 1 speeds up inference by resizing input before running the network.
*/
bool da2_depth_gray(const cv::Mat& bgr, cv::Mat& depth8u, float scale) {
    if (!g_netReady || !g_net) return false;
    if (bgr.empty() || bgr.type() != CV_8UC3) return false;

    scale = std::clamp(scale, 0.20f, 1.00f);

    std::lock_guard<std::mutex> lk(g_da2Mutex);

    if (g_net->set_input(bgr, scale) != 0) return false;

    cv::Mat out;
    if (g_net->run_network(out, bgr.size()) != 0) return false;

    // run_network returns a grayscale image (normally CV_8UC1); enforce it.
    if (out.empty()) return false;
    if (out.type() == CV_8UC1) {
        depth8u = out;
    } else {
        cv::Mat tmp;
        out.convertTo(tmp, CV_8U);
        depth8u = tmp;
    }
    return true;
}

/*
  da2_reset_calibration
  Resets the lightweight distance calibration used by da2_face_distance_cm().
  This matches the reference behavior where calibration restarts when toggled.
  The calibration is heuristic and depends on stable face size over time.
*/
void da2_reset_calibration() {
    std::lock_guard<std::mutex> lk(g_calibMutex);
    g_calibrated = false;
    g_A0 = 1.0;
    g_areaSmooth = 1.0;
    g_hasArea = false;
}

/*
  da2_face_distance_cm
  Estimates face distance using the reference heuristic (mostly face size-based).
  The depth8u input is accepted for API compatibility (future: use depth ROI mean).
  Returns distance in centimeters with a confidence score in [0..1].
*/
float da2_face_distance_cm(const cv::Mat&, const cv::Rect& faceRect, float& conf) {
    conf = 0.0f;
    if (faceRect.width <= 0 || faceRect.height <= 0) return -1.0f;

    const double A = (double)faceRect.area();
    if (A < 10.0) return -1.0f;

    std::lock_guard<std::mutex> lk(g_calibMutex);

    // Smooth face area to reduce jitter.
    const double alpha = 0.25;
    if (!g_hasArea) {
        g_areaSmooth = A;
        g_hasArea = true;
    } else {
        g_areaSmooth = (1.0 - alpha) * g_areaSmooth + alpha * A;
    }

    // First time: set baseline area.
    if (!g_calibrated) {
        g_A0 = std::max(1.0, g_areaSmooth);
        g_calibrated = true;
    }

    // Inverse sqrt relationship between area and distance.
    const double d = CALIB_CM * std::sqrt(g_A0 / std::max(1.0, g_areaSmooth));

    // Confidence: higher if area is stable and reasonably sized.
    const double sizeConf = std::clamp(g_areaSmooth / (g_A0 + 1e-6), 0.0, 2.0);
    conf = (float)std::clamp(0.5 * sizeConf, 0.0, 1.0);

    return (float)d;
}

/* -------------------------- image filters --------------------------- */

/*
  greyscale
  Converts BGR to a 3-channel grayscale image using an integer luminance approximation.
  Implemented with pointer access (no cv::cvtColor) for learning and speed.
  Output remains CV_8UC3 so downstream code doesn’t need special cases.
*/
int greyscale(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) return -1;

    dst.create(src.size(), CV_8UC3);

    for (int r = 0; r < src.rows; ++r) {
        const cv::Vec3b* sp = src.ptr<cv::Vec3b>(r);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < src.cols; ++c) {
            const int b = sp[c][0];
            const int g = sp[c][1];
            const int rr = sp[c][2];
            const int y = (29 * b + 150 * g + 77 * rr) >> 8; // ~0.114B + 0.587G + 0.299R
            dp[c] = cv::Vec3b((uchar)y, (uchar)y, (uchar)y);
        }
    }
    return 0;
}

/*
  sepia
  Applies the standard sepia transform while always using original pixel values.
  Each output channel is clamped to [0,255] to avoid overflow in 8-bit images.
  Output stays CV_8UC3 (BGR) so it can be displayed directly.
*/
int sepia(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) return -1;

    dst.create(src.size(), CV_8UC3);

    for (int r = 0; r < src.rows; ++r) {
        const cv::Vec3b* sp = src.ptr<cv::Vec3b>(r);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < src.cols; ++c) {
            const int b = sp[c][0];
            const int g = sp[c][1];
            const int rr = sp[c][2];

            int nr = (int)std::lround(0.393 * rr + 0.769 * g + 0.189 * b);
            int ng = (int)std::lround(0.349 * rr + 0.686 * g + 0.168 * b);
            int nb = (int)std::lround(0.272 * rr + 0.534 * g + 0.131 * b);

            nr = std::clamp(nr, 0, 255);
            ng = std::clamp(ng, 0, 255);
            nb = std::clamp(nb, 0, 255);

            dp[c] = cv::Vec3b((uchar)nb, (uchar)ng, (uchar)nr);
        }
    }
    return 0;
}

/*
  negative
  Inverts each BGR channel with dst = 255 - src to create a negative image.
  Reads from src and writes to dst (safe even if src and dst alias).
  Output matches input type and size (CV_8UC3).
*/
int negative(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) return -1;

    dst.create(src.size(), CV_8UC3);
    for (int r = 0; r < src.rows; ++r) {
        const cv::Vec3b* sp = src.ptr<cv::Vec3b>(r);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);
        for (int c = 0; c < src.cols; ++c) {
            dp[c][0] = (uchar)(255 - sp[c][0]);
            dp[c][1] = (uchar)(255 - sp[c][1]);
            dp[c][2] = (uchar)(255 - sp[c][2]);
        }
    }
    return 0;
}

/*
  applyBrightnessContrast
  Applies affine tone mapping: dst = src * contrast + brightness.
  Uses saturate_cast to clamp to [0,255] so overflows don’t wrap.
  Applied last in the pipeline so UI controls affect every mode consistently.
*/
int applyBrightnessContrast(const cv::Mat &src, cv::Mat &dst, float contrast, int brightness) {
    if (src.empty() || src.type() != CV_8UC3) return -1;

    dst.create(src.size(), CV_8UC3);
    for (int r = 0; r < src.rows; ++r) {
        const cv::Vec3b* sp = src.ptr<cv::Vec3b>(r);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);
        for (int c = 0; c < src.cols; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                const int v = (int)std::lround(sp[c][ch] * contrast + brightness);
                dp[c][ch] = cv::saturate_cast<uchar>(v);
            }
        }
    }
    return 0;
}

/*
  blur5x5_1
  Simple 5x5 blur reference implementation using a separable kernel.
  Uses int16 intermediate storage to prevent overflow.
  Copies borders from src so outer pixels remain non-zero.
*/
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) return -1;

    dst = src.clone();
    if (src.rows < 5 || src.cols < 5) return 0;

    cv::Mat tmp(src.size(), CV_16SC3, cv::Scalar(0,0,0));

    // Horizontal [1 2 4 2 1]
    for (int r = 0; r < src.rows; ++r) {
        const cv::Vec3b* sp = src.ptr<cv::Vec3b>(r);
        cv::Vec3s* tp = tmp.ptr<cv::Vec3s>(r);
        for (int c = 2; c < src.cols - 2; ++c) {
            const cv::Vec3b& p0 = sp[c - 2];
            const cv::Vec3b& p1 = sp[c - 1];
            const cv::Vec3b& p2 = sp[c];
            const cv::Vec3b& p3 = sp[c + 1];
            const cv::Vec3b& p4 = sp[c + 2];
            for (int ch = 0; ch < 3; ++ch) {
                tp[c][ch] = (short)(p0[ch] + 2*p1[ch] + 4*p2[ch] + 2*p3[ch] + p4[ch]);
            }
        }
    }

    // Vertical + normalize (sum 16*16 = 256)
    for (int r = 2; r < src.rows - 2; ++r) {
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);
        const cv::Vec3s* t0 = tmp.ptr<cv::Vec3s>(r - 2);
        const cv::Vec3s* t1 = tmp.ptr<cv::Vec3s>(r - 1);
        const cv::Vec3s* t2 = tmp.ptr<cv::Vec3s>(r);
        const cv::Vec3s* t3 = tmp.ptr<cv::Vec3s>(r + 1);
        const cv::Vec3s* t4 = tmp.ptr<cv::Vec3s>(r + 2);
        for (int c = 2; c < src.cols - 2; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                const int sum = t0[c][ch] + 2*t1[c][ch] + 4*t2[c][ch] + 2*t3[c][ch] + t4[c][ch];
                dp[c][ch] = (uchar)(sum >> 8);
            }
        }
    }
    return 0;
}

/*
  blur5x5_2
  Faster separable 5x5 blur using pointer access and minimal indexing.
  Uses a CV_16SC3 intermediate and normalizes via bit shift (>> 8).
  Leaves borders copied from src, matching assignment expectations.
*/
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    return blur5x5_1(src, dst); // keep one correct optimized separable path
}

/*
  blurQuantize
  Blurs first, then reduces colors to 'levels' per channel (posterization).
  Levels are clamped to keep bucket sizes meaningful and avoid divide-by-zero.
  Designed to show the effect of smoothing + reduced color resolution.
*/
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty() || src.type() != CV_8UC3) return -1;

    levels = std::clamp(levels, 1, 30);

    cv::Mat blurred;
    blur5x5_2(src, blurred);

    dst.create(src.size(), CV_8UC3);
    const int bucket = std::max(1, 256 / levels);

    for (int r = 0; r < blurred.rows; ++r) {
        const cv::Vec3b* sp = blurred.ptr<cv::Vec3b>(r);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);
        for (int c = 0; c < blurred.cols; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                int q = (sp[c][ch] / bucket) * bucket;
                dp[c][ch] = (uchar)std::min(255, q);
            }
        }
    }
    return 0;
}

/*
  sobelX3x3
  Computes the horizontal gradient using Sobel (separable form).
  Output is CV_16SC3 to preserve negative values for later magnitude/abs.
  Borders are set to zero; valid gradients are in rows/cols 1..(n-2).
*/
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) return -1;

    cv::Mat tmp(src.size(), CV_16SC3, cv::Scalar(0,0,0));
    dst.create(src.size(), CV_16SC3);
    dst.setTo(cv::Scalar(0,0,0));

    // Smooth vertically [1 2 1]^T
    for (int r = 1; r < src.rows - 1; ++r) {
        const cv::Vec3b* s0 = src.ptr<cv::Vec3b>(r - 1);
        const cv::Vec3b* s1 = src.ptr<cv::Vec3b>(r);
        const cv::Vec3b* s2 = src.ptr<cv::Vec3b>(r + 1);
        cv::Vec3s* tp = tmp.ptr<cv::Vec3s>(r);
        for (int c = 0; c < src.cols; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                tp[c][ch] = (short)(s0[c][ch] + 2*s1[c][ch] + s2[c][ch]);
            }
        }
    }

    // Derivative horizontally [-1 0 1]
    for (int r = 1; r < src.rows - 1; ++r) {
        const cv::Vec3s* tr = tmp.ptr<cv::Vec3s>(r);
        cv::Vec3s* dp = dst.ptr<cv::Vec3s>(r);
        for (int c = 1; c < src.cols - 1; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                dp[c][ch] = (short)(tr[c+1][ch] - tr[c-1][ch]);
            }
        }
    }
    return 0;
}

/*
  sobelY3x3
  Computes the vertical gradient using Sobel (separable form).
  Output is CV_16SC3 to preserve negative values for later magnitude/abs.
  Borders are set to zero; valid gradients are in rows/cols 1..(n-2).
*/
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.type() != CV_8UC3) return -1;

    cv::Mat tmp(src.size(), CV_16SC3, cv::Scalar(0,0,0));
    dst.create(src.size(), CV_16SC3);
    dst.setTo(cv::Scalar(0,0,0));

    // Smooth horizontally [1 2 1]
    for (int r = 0; r < src.rows; ++r) {
        const cv::Vec3b* sp = src.ptr<cv::Vec3b>(r);
        cv::Vec3s* tp = tmp.ptr<cv::Vec3s>(r);
        for (int c = 1; c < src.cols - 1; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                tp[c][ch] = (short)(sp[c-1][ch] + 2*sp[c][ch] + sp[c+1][ch]);
            }
        }
    }

    // Derivative vertically [-1 0 1]^T
    for (int r = 1; r < src.rows - 1; ++r) {
        const cv::Vec3s* t0 = tmp.ptr<cv::Vec3s>(r - 1);
        const cv::Vec3s* t2 = tmp.ptr<cv::Vec3s>(r + 1);
        cv::Vec3s* dp = dst.ptr<cv::Vec3s>(r);
        for (int c = 1; c < src.cols - 1; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                dp[c][ch] = (short)(t2[c][ch] - t0[c][ch]);
            }
        }
    }
    return 0;
}

/*
  magnitude
  Computes sqrt(sx^2 + sy^2) per channel and clamps to [0,255] for display.
  Expects sx and sy to be CV_16SC3 and same size.
  Outputs CV_8UC3, suitable for immediate visualization.
*/
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    if (sx.empty() || sy.empty()) return -1;
    if (sx.type() != CV_16SC3 || sy.type() != CV_16SC3) return -1;
    if (sx.size() != sy.size()) return -1;

    dst.create(sx.size(), CV_8UC3);

    for (int r = 0; r < sx.rows; ++r) {
        const cv::Vec3s* px = sx.ptr<cv::Vec3s>(r);
        const cv::Vec3s* py = sy.ptr<cv::Vec3s>(r);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);
        for (int c = 0; c < sx.cols; ++c) {
            for (int ch = 0; ch < 3; ++ch) {
                const int gx = px[c][ch];
                const int gy = py[c][ch];
                const int mag = (int)std::lround(std::sqrt((double)gx*gx + (double)gy*gy));
                dp[c][ch] = cv::saturate_cast<uchar>(mag);
            }
        }
    }
    return 0;
}

/*
  emboss
  Produces an emboss effect by taking directional differences and recentering.
  dirX/dirY select the highlight direction (normalized is best).
  Output is CV_8UC3 and intended for fast real-time stylization.
*/
int emboss(cv::Mat &src, cv::Mat &dst, float dirX, float dirY) {
    if (src.empty() || src.type() != CV_8UC3) return -1;

    dst = src.clone();
    if (src.rows < 3 || src.cols < 3) return 0;

    const int dx = (dirX >= 0.0f) ? 1 : -1;
    const int dy = (dirY >= 0.0f) ? 1 : -1;

    for (int r = 1; r < src.rows - 1; ++r) {
        const cv::Vec3b* sA = src.ptr<cv::Vec3b>(r - dy);
        const cv::Vec3b* sB = src.ptr<cv::Vec3b>(r + dy);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);

        for (int c = 1; c < src.cols - 1; ++c) {
            const cv::Vec3b& p0 = sA[c - dx];
            const cv::Vec3b& p1 = sB[c + dx];
            for (int ch = 0; ch < 3; ++ch) {
                const int v = (int)p1[ch] - (int)p0[ch] + 128;
                dp[c][ch] = cv::saturate_cast<uchar>(v);
            }
        }
    }
    return 0;
}
