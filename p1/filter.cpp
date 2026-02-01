#include "filter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>

#include "DA2Network.hpp"

/* ----------------------------- DA2 state ----------------------------- */

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

// Implements the same core DA2 flow as da2-video.cpp:
// - downsample the frame
// - compute scale_factor = 256 / reduced_height (then multiply by scale_mult)
// - set_input + run_network
bool da2_depth_gray(const cv::Mat& bgr, cv::Mat& depth8u, float scale_mult) {
    if (!g_netReady || !g_net) return false;
    if (bgr.empty() || bgr.type() != CV_8UC3) return false;

    const float reduction = 0.5f;
    scale_mult = std::clamp(scale_mult, 0.20f, 1.00f);

    cv::Mat small;
    cv::resize(bgr, small, cv::Size(), reduction, reduction, cv::INTER_AREA);
    if (small.empty()) return false;

    const float base_scale_factor = 256.0f / std::max(1, small.rows);
    const float scale_factor = base_scale_factor * scale_mult;

    std::lock_guard<std::mutex> lk(g_da2Mutex);

    if (g_net->set_input(small, scale_factor) != 0) return false;

    cv::Mat out;
    if (g_net->run_network(out, small.size()) != 0) return false;

    if (out.empty()) return false;

    if (out.type() == CV_8UC1) depth8u = out;
    else out.convertTo(depth8u, CV_8U);

    return true;
}

void da2_reset_calibration() {
    std::lock_guard<std::mutex> lk(g_calibMutex);
    g_calibrated = false;
    g_A0 = 1.0;
    g_areaSmooth = 1.0;
    g_hasArea = false;
}

// Simple, stable heuristic based on face bounding-box area.
// Depth map is accepted (to match the header) but not required for this heuristic.
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

/* ----------------------------- Filters ----------------------------- */

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

            nb = std::clamp(nb, 0, 255);
            ng = std::clamp(ng, 0, 255);
            nr = std::clamp(nr, 0, 255);

            dp[c][0] = (uchar)nb;
            dp[c][1] = (uchar)ng;
            dp[c][2] = (uchar)nr;
        }
    }
    return 0;
}

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
            out[0] = (uchar)std::min(255, std::max(0, sum[0] / 100));
            out[1] = (uchar)std::min(255, std::max(0, sum[1] / 100));
            out[2] = (uchar)std::min(255, std::max(0, sum[2] / 100));
        }
    }
    return 0;
}

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
            int c0 = c;
            int c3 = std::min(src.cols - 1, c + 1);
            int c4 = std::min(src.cols - 1, c + 2);

            for (int ch = 0; ch < 3; ch++) {
                int v = 1 * sp[c2][ch] + 2 * sp[c1][ch] + 4 * sp[c0][ch]
                      + 2 * sp[c3][ch] + 1 * sp[c4][ch];
                tp[c][ch] = (short)(v / 10);
            }
        }
    }

    for (int r = 0; r < src.rows; r++) {
        int r2 = std::max(0, r - 2);
        int r1 = std::max(0, r - 1);
        int r0 = r;
        int r3 = std::min(src.rows - 1, r + 1);
        int r4 = std::min(src.rows - 1, r + 2);

        const cv::Vec3s *t2 = tmp.ptr<cv::Vec3s>(r2);
        const cv::Vec3s *t1 = tmp.ptr<cv::Vec3s>(r1);
        const cv::Vec3s *t0 = tmp.ptr<cv::Vec3s>(r0);
        const cv::Vec3s *t3 = tmp.ptr<cv::Vec3s>(r3);
        const cv::Vec3s *t4 = tmp.ptr<cv::Vec3s>(r4);

        cv::Vec3b *dp = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < src.cols; c++) {
            for (int ch = 0; ch < 3; ch++) {
                int v = 1 * t2[c][ch] + 2 * t1[c][ch] + 4 * t0[c][ch]
                      + 2 * t3[c][ch] + 1 * t4[c][ch];
                v /= 10;
                dp[c][ch] = (uchar)std::min(255, std::max(0, v));
            }
        }
    }
    return 0;
}

static inline uchar quantizeOne(uchar v, int levels) {
    levels = std::max(1, levels);
    const int bucket = 256 / levels;
    const int q = (bucket > 0) ? (v / bucket) * bucket : v;
    return (uchar)std::clamp(q, 0, 255);
}

int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels) {
    cv::Mat blurred;
    if (blur5x5_2(src, blurred) != 0) return -1;

    dst.create(blurred.size(), blurred.type());

    for (int r = 0; r < blurred.rows; r++) {
        const cv::Vec3b* sp = blurred.ptr<cv::Vec3b>(r);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);
        for (int c = 0; c < blurred.cols; c++) {
            dp[c][0] = quantizeOne(sp[c][0], levels);
            dp[c][1] = quantizeOne(sp[c][1], levels);
            dp[c][2] = quantizeOne(sp[c][2], levels);
        }
    }
    return 0;
}

int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
    if (src.empty()) return -1;
    if (src.type() != CV_8UC3) return -1;

    dst.create(src.size(), CV_16SC3);
    dst.setTo(cv::Scalar(0,0,0));

    for (int r = 1; r < src.rows - 1; r++) {
        const cv::Vec3b* p0 = src.ptr<cv::Vec3b>(r - 1);
        const cv::Vec3b* p1 = src.ptr<cv::Vec3b>(r);
        const cv::Vec3b* p2 = src.ptr<cv::Vec3b>(r + 1);
        cv::Vec3s* dp = dst.ptr<cv::Vec3s>(r);

        for (int c = 1; c < src.cols - 1; c++) {
            for (int ch = 0; ch < 3; ch++) {
                int gx =
                    -p0[c - 1][ch] + p0[c + 1][ch] +
                    -2 * p1[c - 1][ch] + 2 * p1[c + 1][ch] +
                    -p2[c - 1][ch] + p2[c + 1][ch];
                dp[c][ch] = (short)gx;
            }
        }
    }
    return 0;
}

int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
    if (src.empty()) return -1;
    if (src.type() != CV_8UC3) return -1;

    dst.create(src.size(), CV_16SC3);
    dst.setTo(cv::Scalar(0,0,0));

    for (int r = 1; r < src.rows - 1; r++) {
        const cv::Vec3b* p0 = src.ptr<cv::Vec3b>(r - 1);
        const cv::Vec3b* p1 = src.ptr<cv::Vec3b>(r);
        const cv::Vec3b* p2 = src.ptr<cv::Vec3b>(r + 1);
        cv::Vec3s* dp = dst.ptr<cv::Vec3s>(r);

        for (int c = 1; c < src.cols - 1; c++) {
            for (int ch = 0; ch < 3; ch++) {
                int gy =
                    -p0[c - 1][ch] - 2 * p0[c][ch] - p0[c + 1][ch] +
                     p2[c - 1][ch] + 2 * p2[c][ch] + p2[c + 1][ch];
                dp[c][ch] = (short)gy;
            }
        }
    }
    return 0;
}

int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
    if (sx.empty() || sy.empty()) return -1;
    if (sx.type() != CV_16SC3 || sy.type() != CV_16SC3) return -1;
    if (sx.size() != sy.size()) return -1;

    dst.create(sx.size(), CV_8UC3);

    for (int r = 0; r < sx.rows; r++) {
        const cv::Vec3s* px = sx.ptr<cv::Vec3s>(r);
        const cv::Vec3s* py = sy.ptr<cv::Vec3s>(r);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < sx.cols; c++) {
            for (int ch = 0; ch < 3; ch++) {
                int vx = (int)px[c][ch];
                int vy = (int)py[c][ch];
                int v = (int)std::sqrt((double)vx * vx + (double)vy * vy);
                dp[c][ch] = (uchar)std::clamp(v, 0, 255);
            }
        }
    }
    return 0;
}

int emboss(cv::Mat& src, cv::Mat& dst, float dirX, float dirY) {
    if (src.empty()) return -1;
    if (src.type() != CV_8UC3) return -1;

    cv::Mat sx, sy;
    sobelX3x3(src, sx);
    sobelY3x3(src, sy);

    dst.create(src.size(), CV_8UC3);

    for (int r = 0; r < src.rows; r++) {
        const cv::Vec3s* px = sx.ptr<cv::Vec3s>(r);
        const cv::Vec3s* py = sy.ptr<cv::Vec3s>(r);
        cv::Vec3b* dp = dst.ptr<cv::Vec3b>(r);

        for (int c = 0; c < src.cols; c++) {
            for (int ch = 0; ch < 3; ch++) {
                float v = dirX * (float)px[c][ch] + dirY * (float)py[c][ch];
                int shade = (int)std::lround(128.0f + v);
                shade = std::clamp(shade, 0, 255);
                dp[c][ch] = (uchar)shade;
            }
        }
    }
    return 0;
}
