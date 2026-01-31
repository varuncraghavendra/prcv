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

#include "filter.h"
#include "faceDetect.h"

static const char* WIN_VIDEO = "Video";
static const char* WIN_CTRL  = "Controls";

/* Trackbars (UI parameters) */
static int tbBrightness = 255; // [0..510] -> [-255..+255]
static int tbContrast   = 100; // [20..300] -> [0.20..3.00]
static int tbLevels     = 10;  // [1..30]
static int tbDA2Scale   = 45;  // [20..100] -> [0.20..1.00]
static int tbDA2Hz      = 10;  // [1..30]

/* Toggles controlled by keyboard AND mouse buttons */
static std::atomic<bool> faceOn{false};
static std::atomic<bool> depthOn{false};
static std::atomic<bool> embossOn{false};
static std::atomic<bool> negativeOn{false};

/* Mode selects one main filter path */
static std::atomic<char> mode{'c'};
static int saveCount = 0;

/* Frame handoff for depth thread */
static std::mutex mtxFrame;
static std::condition_variable cvFrame;
static cv::Mat sharedFrame;
static std::atomic<bool> hasFrame{false};
static std::atomic<bool> quitFlag{false};

/* Depth cache produced by the DA2 thread */
struct DepthCache {
    cv::Mat depth8u;
    cv::Size size;
    float depthFps = 0.0f;
    std::chrono::steady_clock::time_point lastUpdate;
};
static std::mutex mtxDepth;
static DepthCache depthCache;

/* ----------------------------- UI helpers ----------------------------- */

static inline int getBrightness() { return tbBrightness - 255; }

static inline float getContrast() {
    return std::clamp(tbContrast / 100.0f, 0.20f, 3.00f);
}

static inline int getLevels() { return std::clamp(tbLevels, 1, 30); }

static inline float getDA2Scale() {
    return std::clamp(tbDA2Scale / 100.0f, 0.20f, 1.00f);
}

static inline int getDA2Hz() { return std::clamp(tbDA2Hz, 1, 30); }

struct Button {
    std::string label;
    cv::Rect rect;
    std::atomic<bool>* state;
};

static std::vector<Button> buttons;

/*
  rebuildButtons
  Builds a simple vertical list of clickable toggle buttons.
  Each button points to the same atomic boolean used by keyboard toggles.
  Called once on startup (or again if you decide to resize the panel).
*/
static void rebuildButtons(int panelW) {
    buttons.clear();
    const int pad = 12;
    const int bw  = std::max(10, panelW - 2 * pad);
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

/*
  drawControlsPanel
  Draws the controls UI window: toggle buttons + a compact key legend.
  Buttons are green when ON and gray when OFF for instant visual feedback.
  This panel mirrors keyboard controls, so either input method works.
*/
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

    cv::putText(panel,
                "Modes: c color | g cv-gray | h gray | p sepia | b blur | x/y sobel | m mag | l blurQ",
                cv::Point(12, panel.rows - 44), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(200, 200, 200), 1);

    cv::putText(panel,
                "Toggles: f face | d depth | t emboss | n negative | s save | q quit",
                cv::Point(12, panel.rows - 18), cv::FONT_HERSHEY_SIMPLEX, 0.50,
                cv::Scalar(200, 200, 200), 1);
}

/*
  onMouseControls
  Handles left-clicks in the Controls window to toggle feature booleans.
  If depth is turned on via mouse, calibration is reset (matches reference intent).
  Keeps interaction simple: left-click toggles exactly one button at a time.
*/
static void onMouseControls(int event, int x, int y, int, void*) {
    if (event != cv::EVENT_LBUTTONDOWN) return;

    const cv::Point p(x, y);
    for (auto& b : buttons) {
        if (b.rect.contains(p) && b.state) {
            const bool newState = !b.state->load();
            b.state->store(newState);
            if (b.state == &depthOn && newState) da2_reset_calibration();
            break;
        }
    }
}

/*
  drawHUD
  Overlays a readable status line with mode, toggles, trackbar values, and FPS.
  The HUD is informational only (it does not change the filter computations).
  Draws a black background rectangle so the text stays readable everywhere.
*/
static void drawHUD(cv::Mat& img, float fps, float depthFps) {
    const int br = getBrightness();
    const float ct = getContrast();
    const int levels = getLevels();
    const float s = getDA2Scale();
    const int hz = getDA2Hz();

    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "Mode:%c Face:%d Depth:%d Emboss:%d Neg:%d | B:%d C:%.2f Levels:%d | DA2:%.2f @%dHz | FPS:%.1f (DA2:%.1f)",
                  mode.load(),
                  faceOn.load() ? 1 : 0,
                  depthOn.load() ? 1 : 0,
                  embossOn.load() ? 1 : 0,
                  negativeOn.load() ? 1 : 0,
                  br, ct, levels, s, hz, fps, depthFps);

    cv::rectangle(img, cv::Rect(10, 8, std::max(1, img.cols - 20), 34), cv::Scalar(0, 0, 0), -1);
    cv::putText(img, buf, cv::Point(18, 32), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(0, 255, 255), 2);
}

/* ------------------------- Depth worker thread ------------------------- */

/*
  depthThreadFunc
  Runs DA2 depth inference at a user-selected rate (Hz) on the latest shared frame.
  Stores the most recent CV_8UC1 depth image in a cache protected by a mutex.
  Only runs inference when depth mode is enabled (depthOn).
*/
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

        const int hz = getDA2Hz();
        const auto now = std::chrono::steady_clock::now();
        const auto minDt = std::chrono::milliseconds((int)(1000.0 / std::max(1, hz)));
        if ((now - lastInfer) < minDt) continue;
        lastInfer = now;

        cv::Mat d8;
        if (!da2_depth_gray(local, d8, getDA2Scale())) continue;

        {
            std::lock_guard<std::mutex> lk(mtxDepth);
            depthCache.depth8u = d8;
            depthCache.size = d8.size();
            depthCache.lastUpdate = now;
        }

        frames++;
        const auto t = std::chrono::steady_clock::now();
        if ((t - lastFpsTick) > std::chrono::seconds(1)) {
            const float dfps = frames / std::chrono::duration<float>(t - lastFpsTick).count();
            {
                std::lock_guard<std::mutex> lk(mtxDepth);
                depthCache.depthFps = dfps;
            }
            frames = 0;
            lastFpsTick = t;
        }
    }
}

/* ------------------------------ main loop ------------------------------ */

/*
  main
  Captures a live webcam stream and applies custom filters based on key presses.
  Uses faceDetect.h for face boxes, and DA2Network.hpp for optional depth inference.
  The app is designed for real-time responsiveness and clean assignment mapping.
*/
int main() {
    cv::setUseOptimized(true);
    cv::setNumThreads((int)std::max(1u, std::thread::hardware_concurrency()));

    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::printf("Unable to open video device\n");
        return -1;
    }
    camera.set(cv::CAP_PROP_BUFFERSIZE, 1);

    cv::namedWindow(WIN_VIDEO, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(WIN_CTRL,  cv::WINDOW_AUTOSIZE);

    cv::createTrackbar("Brightness",      WIN_CTRL, &tbBrightness, 510);
    cv::createTrackbar("Contrast x100",   WIN_CTRL, &tbContrast,   300);
    cv::createTrackbar("Quantize Levels", WIN_CTRL, &tbLevels,      30);
    cv::createTrackbar("DA2 Scale x100",  WIN_CTRL, &tbDA2Scale,   100);
    cv::createTrackbar("DA2 Hz",          WIN_CTRL, &tbDA2Hz,       30);

    tbBrightness = 255;
    tbContrast   = 100;
    tbLevels     = 10;
    tbDA2Scale   = 45;
    tbDA2Hz      = 10;

    cv::Mat ctrlPanel(320, 560, CV_8UC3);
    rebuildButtons(ctrlPanel.cols);
    cv::setMouseCallback(WIN_CTRL, onMouseControls, nullptr);

    // DA2 model path (same default used in the reference examples)
    const std::string modelPath = "model_fp16.onnx";
    std::thread depthWorker(depthThreadFunc, modelPath);

    cv::Mat frame, gray, out, bcFrame, embossed, neg;
    cv::Mat sx16, sy16;
    std::vector<cv::Rect> faces;

    auto lastTick = std::chrono::steady_clock::now();
    float fps = 0.0f;

    for (;;) {
        camera >> frame;
        if (frame.empty()) break;

        // Push latest frame to depth thread
        {
            std::lock_guard<std::mutex> lk(mtxFrame);
            frame.copyTo(sharedFrame);
            hasFrame.store(true);
        }
        cvFrame.notify_one();

        // FPS estimate
        const auto nowTick = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(nowTick - lastTick).count();
        lastTick = nowTick;
        if (dt > 0.0f) fps = 0.9f * fps + 0.1f * (1.0f / dt);

        // Main mode filter
        switch (mode.load()) {
            case 'g':
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                cv::cvtColor(gray, out, cv::COLOR_GRAY2BGR);
                break;

            case 'h':
                greyscale(frame, out);
                break;

            case 'p':
                sepia(frame, out);
                break;

            case 'b':
                blur5x5_2(frame, out);
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

            case 'l':
                blurQuantize(frame, out, getLevels());
                break;

            case 'c':
            default:
                out = frame;
                break;
        }

        // Post-mode toggles
        if (embossOn.load()) {
            emboss(out, embossed);
            out = embossed;
        }
        if (negativeOn.load()) {
            negative(out, neg);
            out = neg;
        }

        // Depth thumbnail (if enabled and available)
        float depthFps = 0.0f;
        cv::Mat depthLocal;
        if (depthOn.load()) {
            std::lock_guard<std::mutex> lk(mtxDepth);
            depthCache.depth8u.copyTo(depthLocal);
            depthFps = depthCache.depthFps;
        }

        if (depthOn.load() && !depthLocal.empty()) {
            cv::Mat depthBgr;
            cv::cvtColor(depthLocal, depthBgr, cv::COLOR_GRAY2BGR);

            const int thumbW = std::max(120, out.cols / 4);
            const int thumbH = std::max(90,  out.rows / 4);

            cv::Mat thumb;
            cv::resize(depthBgr, thumb, cv::Size(thumbW, thumbH), 0, 0, cv::INTER_NEAREST);

            const int x0 = out.cols - thumbW - 12;
            const int y0 = 52;
            if (x0 >= 0 && y0 + thumbH < out.rows) {
                thumb.copyTo(out(cv::Rect(x0, y0, thumbW, thumbH)));
                cv::rectangle(out, cv::Rect(x0, y0, thumbW, thumbH), cv::Scalar(255, 255, 255), 2);
                cv::putText(out, "depth", cv::Point(x0 + 6, y0 + 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.60, cv::Scalar(255, 255, 255), 2);
            }
        }

        // Brightness/contrast always applied last
        applyBrightnessContrast(out, bcFrame, getContrast(), getBrightness());
        out = bcFrame;

        // Face boxes + distance overlay
        if (faceOn.load()) {
            faces.clear();
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            detectFaces(gray, faces);

            for (const auto& rc : faces) {
                cv::rectangle(out, rc, cv::Scalar(0, 255, 0), 2);

                float conf = 0.0f;
                float dist = -1.0f;

                // Use the same API as the reference code; depthLocal may be empty if depthOff.
                dist = da2_face_distance_cm(depthLocal, rc, conf);

                if (dist > 0.0f) {
                    char buf[96];
                    std::snprintf(buf, sizeof(buf), "dist: %.1f cm (%.2f)", dist, conf);
                    cv::putText(out, buf, cv::Point(rc.x, std::max(0, rc.y - 8)),
                                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        drawHUD(out, fps, depthFps);

        cv::imshow(WIN_VIDEO, out);
        drawControlsPanel(ctrlPanel);
        cv::imshow(WIN_CTRL, ctrlPanel);

        const int k = cv::waitKey(1);
        if (k < 0) continue;
        const int key = k & 0xFF;

        if (key == 'q') break;

        // Modes
        if (key == 'c' || key == 'g' || key == 'h' || key == 'p' || key == 'b' ||
            key == 'x' || key == 'y' || key == 'm' || key == 'l') {
            mode.store((char)key);
        }

        // Toggles
        if (key == 'f') faceOn.store(!faceOn.load());
        if (key == 't') embossOn.store(!embossOn.load());
        if (key == 'n') negativeOn.store(!negativeOn.load());

        if (key == 'd') {
            const bool newState = !depthOn.load();
            depthOn.store(newState);
            if (newState) da2_reset_calibration();
        }

        // Save
        if (key == 's') {
            const char* tag =
                (mode.load() == 'g') ? "cvgray" :
                (mode.load() == 'h') ? "gray" :
                (mode.load() == 'p') ? "sepia" :
                (mode.load() == 'b') ? "blur" :
                (mode.load() == 'x') ? "sobelx" :
                (mode.load() == 'y') ? "sobely" :
                (mode.load() == 'm') ? "mag" :
                (mode.load() == 'l') ? "blurq" : "color";

            char filename[256];
            std::snprintf(filename, sizeof(filename),
                          "capture_%03d_%s_face%d_depth%d_emboss%d_neg%d.png",
                          saveCount++, tag,
                          faceOn.load() ? 1 : 0,
                          depthOn.load() ? 1 : 0,
                          embossOn.load() ? 1 : 0,
                          negativeOn.load() ? 1 : 0);

            cv::imwrite(filename, out);
            std::printf("Saved %s\n", filename);
        }
    }

    // Shutdown depth thread cleanly
    quitFlag.store(true);
    cvFrame.notify_all();
    if (depthWorker.joinable()) depthWorker.join();

    return 0;
}
