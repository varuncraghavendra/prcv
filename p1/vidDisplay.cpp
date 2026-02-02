#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <sys/time.h>

#include "filter.h"
#include "faceDetect.h"

static const char* WIN_VIDEO = "Video";
static const char* WIN_CTRL  = "Controls";

// Trackbars
static int tbBrightness = 255; // [0..510] -> [-255..+255]
static int tbContrast   = 100; // [0..300] -> [0.20..3.00]
// Toggles / mode
static std::atomic<bool> faceOn{false};
static std::atomic<bool> depthOn{false};
static std::atomic<bool> embossOn{false};
static std::atomic<bool> negativeOn{false};
static std::atomic<char> mode{'c'}; // c g h p b x y m l
static int saveCount = 0;

// Shared frame -> depth thread
static std::mutex mtxFrame;
static std::condition_variable cvFrame;
static cv::Mat sharedFrame;
static std::atomic<bool> hasFrame{false};
static std::atomic<bool> quitFlag{false};

struct DepthCache {
    cv::Mat depth8u;
    float depthFps = 0.0f;
};
static std::mutex mtxDepth;
static DepthCache depthCache;

// Slider helpers
/// What it does: (brief) core behavior of this function.
/// Inputs/outputs: key parameters and what is produced/returned.
/// Notes: important constraints, performance, or edge-case handling.
static inline int getBrightness() { return tbBrightness - 255; }
/// What it does: (brief) core behavior of this function.
/// Inputs/outputs: key parameters and what is produced/returned.
/// Notes: important constraints, performance, or edge-case handling.
static inline float getContrast() { return std::clamp(tbContrast / 100.0f, 0.20f, 3.00f); }
/// What it does: (brief) core behavior of this function.
/// Inputs/outputs: key parameters and what is produced/returned.
/// Notes: important constraints, performance, or edge-case handling.
// Runs the same timing as timeBlur.cpp, but triggered from the video app with 'v'.
/// What it does: (brief) core behavior of this function.
/// Inputs/outputs: key parameters and what is produced/returned.
/// Notes: important constraints, performance, or edge-case handling.
static void runTimeBlurOnCathedral() {
    const char *filename = "cathedral.jpeg";
    cv::Mat src = cv::imread(filename);
    if (src.empty()) {
        std::printf("[v] Could not read %s (make sure it's in your working directory)\n", filename);
        return;
    }

    cv::Mat dst;
    const int Ntimes = 10;

    auto getTime = []() -> double {
        struct timeval cur;
        gettimeofday(&cur, NULL);
        return (cur.tv_sec + cur.tv_usec / 1000000.0);
    };

    double start = getTime();
    for (int i = 0; i < Ntimes; i++) blur5x5_1(src, dst);
    double end = getTime();
    std::printf("[v] Time per image (1): %.4lf seconds\n", (end - start) / Ntimes);

    start = getTime();
    for (int i = 0; i < Ntimes; i++) blur5x5_2(src, dst);
    end = getTime();
    std::printf("[v] Time per image (2): %.4lf seconds\n", (end - start) / Ntimes);

    // Save example outputs (one pass each) so you can include results in your report.
    cv::Mat out1, out2;
    blur5x5_1(src, out1);
    blur5x5_2(src, out2);
    cv::imwrite("cathedral_blur5x5_1.png", out1);
    cv::imwrite("cathedral_blur5x5_2.png", out2);
    std::printf("[v] Wrote cathedral_blur5x5_1.png and cathedral_blur5x5_2.png\n");
}

// ---------------------------
// Clickable buttons panel (buttons only; no extra text)
// ---------------------------
struct Button {
    std::string label;
    cv::Rect rect;
    std::atomic<bool>* state;
};
static std::vector<Button> buttons;

/// What it does: (brief) core behavior of this function.
/// Inputs/outputs: key parameters and what is produced/returned.
/// Notes: important constraints, performance, or edge-case handling.
static void rebuildButtons(int w) {
    buttons.clear();
    const int pad = 12;
    const int bw  = std::max(10, w - 2 * pad);
    const int bh  = 44;
    int y = 18;

    auto addBtn = [&](const std::string& label, std::atomic<bool>* state) {
        buttons.push_back(Button{label, cv::Rect(pad, y, bw, bh), state});
        y += bh + 10;
    };

    addBtn("Face Boxes (f)", &faceOn);
    addBtn("Depth (d)", &depthOn);
        addBtn("Emboss (t)", &embossOn);
    addBtn("Negative (n)", &negativeOn);
}

/// What it does: (brief) core behavior of this function.
/// Inputs/outputs: key parameters and what is produced/returned.
/// Notes: important constraints, performance, or edge-case handling.
static void drawControlsPanel(cv::Mat& panel) {
    panel.setTo(cv::Scalar(30, 30, 30));
    for (const auto& b : buttons) {
        const bool on = b.state && b.state->load();
        const cv::Scalar fill = on ? cv::Scalar(60, 160, 60) : cv::Scalar(70, 70, 70);
        const cv::Scalar edge = cv::Scalar(220, 220, 220);

        cv::rectangle(panel, b.rect, fill, -1);
        cv::rectangle(panel, b.rect, edge, 2);
        cv::putText(panel, b.label, cv::Point(b.rect.x + 12, b.rect.y + 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.60, cv::Scalar(255, 255, 255), 2);
    }
}

/// What it does: (brief) core behavior of this function.
/// Inputs/outputs: key parameters and what is produced/returned.
/// Notes: important constraints, performance, or edge-case handling.
static void onMouseControls(int event, int x, int y, int, void*) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    const cv::Point p(x, y);
    for (auto& b : buttons) {
        if (b.rect.contains(p) && b.state) {
            const bool newState = !b.state->load();
            b.state->store(newState);
            if (b.state == &depthOn && newState) break;
        }
    }
}

// ---------------------------
// HUD overlay (2 lines, auto-fit)
// ---------------------------
/// What it does: (brief) core behavior of this function.
/// Inputs/outputs: key parameters and what is produced/returned.
/// Notes: important constraints, performance, or edge-case handling.
static void drawHUD(cv::Mat& img, float fps, float depthFps) {
    const int br = getBrightness();
    const float ct = getContrast();
    const float s = 0.0f;
    const int hz = 10; // fixed depth update rate
    const int levels = 10; // fixed quantize levels

    const bool F = faceOn.load();
    const bool D = depthOn.load();
    const bool E = embossOn.load();
    const bool N = negativeOn.load();
    const char M = mode.load();

    char line1[256];
    char line2[256];

    std::snprintf(line1, sizeof(line1),
                  "Mode:%c Face:%d Depth:%d Emboss:%d Neg:%d | B:%d C:%.2f",
                  M, F?1:0, D?1:0, E?1:0, N?1:0, br, ct);

    std::snprintf(line2, sizeof(line2),
                  "FPS:%.1f (DA2:%.1f) | v: blur timing",
                  fps, depthFps);

    const int pad = 10;
    const int x = pad;
    const int y = pad;
    const int w = std::max(1, img.cols - 2 * pad);

    double fontScale = 0.55;
    int thickness = 2;
    int base = 0;

    auto fitScale = [&](const char* txt) {
        cv::Size ts = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &base);
        while (ts.width > w - 16 && fontScale > 0.35) {
            fontScale -= 0.02;
            ts = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &base);
        }
    };

    fitScale(line1);
    fitScale(line2);

    cv::Size t1 = cv::getTextSize(line1, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &base);
    cv::Size t2 = cv::getTextSize(line2, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &base);

    const int boxH = 10 + t1.height + 6 + t2.height + 10;
    cv::rectangle(img, cv::Rect(x, y, w, boxH), cv::Scalar(0, 0, 0), -1);

    int y1 = y + 10 + t1.height;
    int y2 = y1 + 6 + t2.height;

    cv::putText(img, line1, cv::Point(x + 8, y1),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 255), thickness);

    cv::putText(img, line2, cv::Point(x + 8, y2),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 255), thickness);
}

// ---------------------------
// Depth worker thread
// ---------------------------
/// What it does: (brief) core behavior of this function.
/// Inputs/outputs: key parameters and what is produced/returned.
/// Notes: important constraints, performance, or edge-case handling.
static void depthThreadFunc(const std::string& modelPath) {
    if (!da2_init_once(modelPath)) {
        std::fprintf(stderr, "ERROR: DA2 init failed. Depth will be unavailable.\n");
        return;
    }

    auto lastInfer = std::chrono::steady_clock::now();
    auto lastFpsTick = lastInfer;
    int frames = 0;

    while (!quitFlag.load()) {
        cv::Mat local;
        {
            std::unique_lock<std::mutex> lk(mtxFrame);
            cvFrame.wait(lk, [] { return quitFlag.load() || hasFrame.load(); });
            if (quitFlag.load()) break;
            sharedFrame.copyTo(local);
            hasFrame.store(false);
        }

        if (!depthOn.load()) continue;

        const int hz = 10; // fixed depth update rate
        const auto now = std::chrono::steady_clock::now();
        const auto minDt = std::chrono::milliseconds((int)(1000.0 / std::max(1, hz)));
        if ((now - lastInfer) < minDt) continue;
        lastInfer = now;

        cv::Mat d8;
        if (!da2_depth_gray(local, d8, 0.0f)) continue;

        {
            std::lock_guard<std::mutex> lk(mtxDepth);
            depthCache.depth8u = d8;
        }

        frames++;
        const auto t = std::chrono::steady_clock::now();
        if ((t - lastFpsTick) > std::chrono::seconds(1)) {
            float dfps = frames / std::chrono::duration<float>(t - lastFpsTick).count();
            {
                std::lock_guard<std::mutex> lk(mtxDepth);
                depthCache.depthFps = dfps;
            }
            frames = 0;
            lastFpsTick = t;
        }
    }
}

/// What it does: (brief) core behavior of this function.
/// Inputs/outputs: key parameters and what is produced/returned.
/// Notes: important constraints, performance, or edge-case handling.
int main(int argc, char** argv) {
    std::string modelPath = "model_fp16.onnx";
    if (argc >= 2 && argv[1] && std::string(argv[1]).size() > 0) modelPath = argv[1];

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "Unable to open video device\n");
        return -1;
    }

    cv::namedWindow(WIN_VIDEO, cv::WINDOW_NORMAL);
    cv::namedWindow(WIN_CTRL,  cv::WINDOW_NORMAL);
    cv::resizeWindow(WIN_VIDEO, 960, 720);
    cv::resizeWindow(WIN_CTRL,  520, 720);

    const int ctrlW = 520;
    const int ctrlH = 520;
    cv::Mat ctrlPanel(ctrlH, ctrlW, CV_8UC3);
    rebuildButtons(ctrlW);

    cv::createTrackbar("Brightness",     WIN_CTRL, &tbBrightness, 510);
    cv::createTrackbar("Contrast x100",  WIN_CTRL, &tbContrast,   300);
    cv::setMouseCallback(WIN_CTRL, onMouseControls, nullptr);

    std::thread depthWorker(depthThreadFunc, modelPath);

    cv::Mat frame, out, gray, bcFrame, embossedImg, negImg;
    cv::Mat sx16, sy16;
    std::vector<cv::Rect> faces;

    auto t0 = std::chrono::steady_clock::now();
    int frameCount = 0;
    float fps = 0.0f;

    for (;;) {
        cap >> frame;
        if (frame.empty()) break;

        // publish a frame for depth thread (thread will only run when depthOn is true)
        {
            std::lock_guard<std::mutex> lk(mtxFrame);
            frame.copyTo(sharedFrame);
            hasFrame.store(true);
        }
        cvFrame.notify_one();

        // mode pipeline
        switch (mode.load()) {
            case 'g': {
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                cv::cvtColor(gray, out, cv::COLOR_GRAY2BGR);
                break;
            }
            case 'h':
                greyscale(frame, out);
                break;
            case 'p':
                sepia(frame, out);
                break;
            case 'x':
                sobelX3x3(frame, sx16);
                cv::convertScaleAbs(sx16, out);
                break;
            case 'y':
                sobelY3x3(frame, sy16);
                cv::convertScaleAbs(sy16, out);
                break;
            case 'm':
                sobelX3x3(frame, sx16);
                sobelY3x3(frame, sy16);
                magnitude(sx16, sy16, out);
                break;
            case 'b':
                blurQuantize(frame, out, 10);
                break;
            case 'c':
            default:
                out = frame.clone();
                break;
        }

        if (embossOn.load()) {
            emboss(out, embossedImg, 0.7071f, 0.7071f);
            out = embossedImg;
        }

        if (negativeOn.load()) {
            negative(out, negImg);
            out = negImg;
        }

        // Depth cache (for thumbnail + per-face value + HUD)
        float depthFps = 0.0f;
        cv::Mat depthLocal;
        if (depthOn.load()) {
            std::lock_guard<std::mutex> lk(mtxDepth);
            depthCache.depth8u.copyTo(depthLocal);
            depthFps = depthCache.depthFps;
        }

// depth thumbnail inside video frame
        if (depthOn.load() && !depthLocal.empty()) {
            cv::Mat depthVis;
            cv::applyColorMap(depthLocal, depthVis, cv::COLORMAP_INFERNO);

            const int thumbW = std::max(140, out.cols / 4);
            const int thumbH = std::max(100, out.rows / 4);
            cv::Mat thumb;
            cv::resize(depthVis, thumb, cv::Size(thumbW, thumbH), 0, 0, cv::INTER_NEAREST);

            const int x0 = out.cols - thumbW - 12;
            const int y0 = 52;
            if (x0 >= 0 && y0 + thumbH < out.rows) {
                thumb.copyTo(out(cv::Rect(x0, y0, thumbW, thumbH)));
                cv::rectangle(out, cv::Rect(x0, y0, thumbW, thumbH), cv::Scalar(255, 255, 255), 2);
                cv::putText(out, "depth", cv::Point(x0 + 8, y0 + 22),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            }
        }

        // brightness/contrast last
        applyBrightnessContrast(out, bcFrame, getContrast(), getBrightness());
        out = bcFrame;

        // face boxes + depth value per face
        if (faceOn.load()) {
            faces.clear();
            cv::Mat faceGray;
            cv::cvtColor(frame, faceGray, cv::COLOR_BGR2GRAY);
            detectFaces(faceGray, faces);

            for (const auto& rc : faces) {
                cv::rectangle(out, rc, cv::Scalar(0, 255, 0), 2);

                if (depthOn.load() && !depthLocal.empty()) {
                    // DA2 depth output is computed on a downsampled frame in da2_depth_gray:
                    // reduction = 0.5, output size == (frame.cols*0.5, frame.rows*0.5).
                    const float reduction = 0.5f;
                    cv::Rect rcSmall((int)std::lround(rc.x * reduction),
                                     (int)std::lround(rc.y * reduction),
                                     (int)std::lround(rc.width * reduction),
                                     (int)std::lround(rc.height * reduction));
                    rcSmall &= cv::Rect(0, 0, depthLocal.cols, depthLocal.rows);

                    float conf = 0.0f;
                    float dist = da2_face_distance_cm(depthLocal, rcSmall, conf);

                    if (dist > 0.0f) {
                        char buf[96];
                        std::snprintf(buf, sizeof(buf), "dist: %.1f cm (%.2f)", dist, conf);
                        cv::putText(out, buf, cv::Point(rc.x, std::max(0, rc.y - 8)),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
                    }
                }
            }
        }

        // fps
        frameCount++;
        auto t1 = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(t1 - t0).count();
        if (dt >= 1.0f) {
            fps = frameCount / dt;
            frameCount = 0;
            t0 = t1;
        }

        drawHUD(out, fps, depthFps);

        cv::imshow(WIN_VIDEO, out);
        drawControlsPanel(ctrlPanel);
        cv::imshow(WIN_CTRL, ctrlPanel);

        char key = (char)cv::waitKey(10);
        if (key == 'q') break;

        if (key == 'c' || key == 'g' || key == 'h' || key == 'p' || key == 'b' ||
            key == 'x' || key == 'y' || key == 'm' || key == 'l') {
            mode.store(key);
        }

        if (key == 'f') faceOn.store(!faceOn.load());
        if (key == 't') embossOn.store(!embossOn.load());
        if (key == 'n') negativeOn.store(!negativeOn.load());

        if (key == 'd') {
            const bool newState = !depthOn.load();
            depthOn.store(newState);
        }

        if (key == 'v') runTimeBlurOnCathedral();

        if (key == 's') {
            char fname[128];
            std::snprintf(fname, sizeof(fname), "saved_%03d.png", saveCount++);
            cv::imwrite(fname, out);
            std::printf("Saved %s\n", fname);
        }
    }

    quitFlag.store(true);
    cvFrame.notify_all();
    if (depthWorker.joinable()) depthWorker.join();
    return 0;
}
