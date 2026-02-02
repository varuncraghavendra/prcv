#include "filter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>

#include "DA2Network.hpp"

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
 * Initializes the Depth Anything V2 network once.
 * Reloads the model only if the path changes.
 * Thread-safe and safe to call repeatedly.
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
 * Runs the depth network and produces an 8-bit depth image.
 * Internally downsamples the input for performance.
 * Safe for real-time video pipelines.
 */
bool da2_depth_gray(const cv::Mat& bgr, cv::Mat& depth8u, float /*scale_mult*/) {
    if (!g_netReady || !g_net) return false;
    if (bgr.empty() || bgr.type() != CV_8UC3) return false;

    cv::Mat src;
    cv::Mat dst;

    const float reduction = 0.5f;
    cv::Size refS(bgr.cols, bgr.rows);
    float scale_factor = 256.0f / (refS.height * reduction);

    cv::resize(bgr, src, cv::Size(), reduction, reduction);

    std::lock_guard<std::mutex> lk(g_da2Mutex);

    if (g_net->set_input(src, scale_factor) != 0) return false;
    if (g_net->run_network(dst, src.size()) != 0) return false;
    if (dst.empty()) return false;

    if (dst.type() == CV_8UC1) depth8u = dst;
    else dst.convertTo(depth8u, CV_8U);

    return true;
}

/*
 * Estimates face distance using bounding-box area.
 * Uses smoothing to reduce jitter.
 * Auto-calibrates on first valid detection.
 */
float da2_face_distance_cm(const cv::Mat&, const cv::Rect& faceRect, float& conf) {
    conf = 0.0f;
    if (faceRect.width <= 0 || faceRect.height <= 0) return -1.0f;

    const double A = (double)faceRect.area();
    if (A < 10.0) return -1.0f;

    std::lock_guard<std::mutex> lk(g_calibMutex);

    const double alpha = 0.25;
    if (!g_hasArea) {
        g_areaSmooth = A;
        g_hasArea = true;
    } else {
        g_areaSmooth = alpha * A + (1.0 - alpha) * g_areaSmooth;
    }

    if (!g_calibrated) {
        g_A0 = std::max(1.0, g_areaSmooth);
        g_calibrated = true;
    }

    const double ratio = std::max(1e-6, g_A0 / std::max(1.0, g_areaSmooth));
    const double dist_cm = CALIB_CM * std::sqrt(ratio);

    conf = (g_areaSmooth > 200.0) ? 1.0f : (float)std::clamp(g_areaSmooth / 200.0, 0.0, 1.0);
    return (float)dist_cm;
}

/*
 * Convert video to grayscale using red-channel inversion.
 * Simple pixel-wise operation without the use of OpenCV functions
 */
int greyscale(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;
    if (src.type() != CV_8UC3) return -1;

    dst.create(src.size(), src.type());

    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3b *sp = src.ptr<cv::Vec3b>(r);
        cv::Vec3b *dp = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < src.cols; c++) {
            unsigned char red = sp[c][2];
            unsigned char gray = (unsigned char)(255 - red);
            dp[c][0] = gray;
            dp[c][1] = gray;
            dp[c][2] = gray;
        }
    }
    return 0;
}

/*
 * Apply a sepia tone transformation to video feed.
 * Uses weighted RGB recombination.
 * Values are clamped between [0,255] to avoid overflow.
 */
int sepia(cv::Mat& src, cv::Mat& dst) {
    if (src.empty()) return -1;
    if (src.type() != CV_8UC3) return -1;

    dst.create(src.size(), CV_8UC3);

    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3b* sp = src.ptr<cv::Vec3b>(r);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);
        for (int c = 0; c < src.cols; c++) {
            const int b = sp[c][0];
            const int g = sp[c][1];
            const int rr = sp[c][2];

            int nb = (int)(0.272 * rr + 0.534 * g + 0.131 * b);
            int ng = (int)(0.349 * rr + 0.686 * g + 0.168 * b);
            int nr = (int)(0.393 * rr + 0.769 * g + 0.189 * b);

            dp[c][0] = (uchar)std::clamp(nb, 0, 255);
            dp[c][1] = (uchar)std::clamp(ng, 0, 255);
            dp[c][2] = (uchar)std::clamp(nr, 0, 255);
        }
    }
    return 0;
}

/*
 * Image Negative - Inverts all color channels of a video feed.
 * Constant-time per pixel.
 */
int negative(cv::Mat& src, cv::Mat& dst) {
    if (src.empty()) return -1;
    if (src.type() != CV_8UC3) return -1;

    dst.create(src.size(), CV_8UC3);

    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3b* sp = src.ptr<cv::Vec3b>(r);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);
        for (int c = 0; c < src.cols; c++) {
            dp[c][0] = (uchar)(255 - sp[c][0]);
            dp[c][1] = (uchar)(255 - sp[c][1]);
            dp[c][2] = (uchar)(255 - sp[c][2]);
        }
    }
    return 0;
}

/*
 * Adjust video brightness and contrast manually.
 * Contrast is clamped for stability.
 */
int applyBrightnessContrast(const cv::Mat& src, cv::Mat& dst, float contrast, int brightness) {
    if (src.empty()) return -1;
    if (src.type() != CV_8UC3) return -1;

    contrast = std::clamp(contrast, 0.20f, 3.00f);
    brightness = std::clamp(brightness, -255, 255);

    dst.create(src.size(), CV_8UC3);

    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3b* sp = src.ptr<cv::Vec3b>(r);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);
        for (int c = 0; c < src.cols; c++) {
            for (int k = 0; k < 3; k++) {
                int v = (int)std::lround(contrast * sp[c][k] + brightness);
                dp[c][k] = (uchar)std::clamp(v, 0, 255);
            }
        }
    }
    return 0;
}

/*
 * Blur Filter.
 * Uses full 2D kernel with normalization.
 * Slower but straightforward implementation.
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;
    if (src.type() != CV_8UC3) return -1;

    dst = src.clone();
    if (src.rows < 5 || src.cols < 5) return 0;

    static const int k[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8,16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    for (int r = 2; r < src.rows - 2; r++) {
        for (int c = 2; c < src.cols - 2; c++) {
            int sum[3] = {0, 0, 0};

            for (int dr = -2; dr <= 2; dr++) {
                for (int dc = -2; dc <= 2; dc++) {
                    const cv::Vec3b pix = src.at<cv::Vec3b>(r + dr, c + dc);
                    const int w = k[dr + 2][dc + 2];
                    sum[0] += w * pix[0];
                    sum[1] += w * pix[1];
                    sum[2] += w * pix[2];
                }
            }

            cv::Vec3b &out = dst.at<cv::Vec3b>(r, c);
            out[0] = (uchar)(sum[0] / 100);
            out[1] = (uchar)(sum[1] / 100);
            out[2] = (uchar)(sum[2] / 100);
        }
    }
    return 0;
}

/*
 * Time Optimized separable 5x5 blur.
 * Uses two 1D passes for speed.
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;
    if (src.type() != CV_8UC3) return -1;

    cv::Mat tmp(src.size(), CV_16SC3);
    dst.create(src.size(), CV_8UC3);

    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3b *sp = src.ptr<cv::Vec3b>(r);
        cv::Vec3s *tp = tmp.ptr<cv::Vec3s>(r);

        for (int c = 0; c < src.cols; c++) {
            int c2 = std::max(0, c - 2);
            int c1 = std::max(0, c - 1);
            int c4 = std::min(src.cols - 1, c + 2);
            int c3 = std::min(src.cols - 1, c + 1);

            for (int ch = 0; ch < 3; ch++) {
                int v = sp[c2][ch] + 2 * sp[c1][ch] + 4 * sp[c][ch]
                      + 2 * sp[c3][ch] + sp[c4][ch];
                tp[c][ch] = (short)(v / 10);
            }
        }
    }

    for (int r = 0; r < src.rows; r++) {
        int r2 = std::max(0, r - 2);
        int r1 = std::max(0, r - 1);
        int r4 = std::min(src.rows - 1, r + 2);
        int r3 = std::min(src.rows - 1, r + 1);

        const cv::Vec3s *t2 = tmp.ptr<cv::Vec3s>(r2);
        const cv::Vec3s *t1 = tmp.ptr<cv::Vec3s>(r1);
        const cv::Vec3s *t0 = tmp.ptr<cv::Vec3s>(r);
        const cv::Vec3s *t3 = tmp.ptr<cv::Vec3s>(r3);
        const cv::Vec3s *t4 = tmp.ptr<cv::Vec3s>(r4);

        cv::Vec3b *dp = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < src.cols; c++) {
            for (int ch = 0; ch < 3; ch++) {
                int v = t2[c][ch] + 2 * t1[c][ch] + 4 * t0[c][ch]
                      + 2 * t3[c][ch] + t4[c][ch];
                dp[c][ch] = (uchar)std::clamp(v / 10, 0, 255);
            }
        }
    }
    return 0;
}

/*
 * Combining blur and color quantization.
 * Reduces color palette for stylized effects.
 * Commonly used for cartoon rendering.
 */
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels) {
    if (levels <= 1) levels = 1;
    cv::Mat blurred;
    if (blur5x5_2(src, blurred) != 0) return -1;

    dst.create(blurred.size(), blurred.type());

    const float invLevels = (float)(levels - 1) / 255.0f;
    const float levelsTo255 = 255.0f / (float)(levels - 1);

    for (int r = 0; r < blurred.rows; ++r) {
        const cv::Vec3b* sp = blurred.ptr<cv::Vec3b>(r);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);
        for (int c = 0; c < blurred.cols; ++c) {
            for (int k = 0; k < 3; k++) {
                int q = (int)std::round(sp[c][k] * invLevels) * (int)levelsTo255;
                dp[c][k] = (uchar)std::clamp(q, 0, 255);
            }
        }
    }
    return 0;
}
