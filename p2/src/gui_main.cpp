#include "core.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include <cstdio>

namespace fs = std::filesystem;

struct GuiState {
  std::string dbDir;
  std::string targetPath;
  std::string feature = "rg16";
  std::string metric = "histint";
  int topk = 3;
  std::string embCsv;  
};

static const std::vector<std::string>& featureOrder() {
  static const std::vector<std::string> k = {
      "center7x7",
      "rg16",
      "rgb8",              
      "rgb8_topbottom",
      "colortexture",        
      "embedding_resnet18",  
      "task7"        
  };
  return k;
}

static const std::vector<std::string>& metricOrder() {
  static const std::vector<std::string> k = {
      "ssd",
      "histint",
      "multihist",
      "cosine",
      "colortexture",  
      "task7_bhatt"    
  };
  return k;
}

static bool contains(const std::vector<std::string>& v, const std::string& s) {
  return std::find(v.begin(), v.end(), s) != v.end();
}

static std::string basename(const std::string& p) {
  try {
    return fs::path(p).filename().string();
  } catch (...) {
    return p;
  }
}

static std::string trim(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
  size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
  return s.substr(i);
}

static std::string runCapture(const std::string& cmd) {
  std::string out;
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return out;
  char buf[4096];
  while (fgets(buf, sizeof(buf), pipe)) out += buf;
  pclose(pipe);
  return trim(out);
}





static std::string pickFileDialog(const std::string& title) {
  
  std::string cmd = "zenity --file-selection --title=\"" + title + "\" 2>/dev/null";
  return runCapture(cmd);
}

static std::string pickDirDialog(const std::string& title) {
  std::string cmd = "zenity --file-selection --directory --title=\"" + title + "\" 2>/dev/null";
  return runCapture(cmd);
}

static void cycle(std::string& value, const std::vector<std::string>& order, int dir) {
  auto it = std::find(order.begin(), order.end(), value);
  int idx = (it == order.end()) ? 0 : int(std::distance(order.begin(), it));
  idx = (idx + dir + int(order.size())) % int(order.size());
  value = order[idx];
}

static cv::Mat loadAndFit(const std::string& path, int w, int h) {
  if (path.empty()) {
    cv::Mat blank(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::putText(blank, "(none)", {10, h / 2}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {200, 200, 200}, 2);
    return blank;
  }
  cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
  if (img.empty()) {
    cv::Mat bad(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::putText(bad, "(failed)", {10, h / 2}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 255}, 2);
    return bad;
  }
  cv::Mat out;
  cv::resize(img, out, {w, h}, 0, 0, cv::INTER_AREA);
  return out;
}

static void putHeader(cv::Mat& img, const std::string& line1, const std::string& line2) {
  const int pad = 10;
  cv::rectangle(img, {0, 0}, {img.cols, 58}, cv::Scalar(25, 25, 25), cv::FILLED);
  cv::putText(img, line1, {pad, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);
  cv::putText(img, line2, {pad, 46}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);
}

static std::string findDefaultEmbeddingsCsv(const std::string& dbDir) {
  
  const std::vector<std::string> candidates = {
      "./ResNet18_olym.csv",
      "./resnet18_olym.csv",
      "./embeddings.csv",
      "./src/ResNet18_olym.csv",
      "./src/embeddings.csv",
      (fs::path(dbDir) / "ResNet18_olym.csv").string(),
      (fs::path(dbDir) / "embeddings.csv").string(),
  };
  for (const auto& c : candidates) {
    if (fs::exists(c)) return c;
  }
  return "";
}

static std::vector<imgsearch::SearchMatch> runSearch(const GuiState& s) {
  if (s.dbDir.empty() || s.targetPath.empty()) return {};
  
  std::string feature = s.feature;
  std::string metric = s.metric;

  if (feature == "task7") {
    metric = "task7_bhatt";
  }
  if (feature == "colortexture") metric = "colortexture";
  if (feature == "embedding_resnet18" && metric == "multihist") metric = "cosine";

  if (feature == "embedding_resnet18") {
    std::string csv = s.embCsv.empty() ? findDefaultEmbeddingsCsv(s.dbDir) : s.embCsv;
    return imgsearch::searchEmbeddings(csv, s.dbDir, s.targetPath, metric, s.topk);
  }
  if (feature == "task7") {
    return imgsearch::searchTask7(s.dbDir, s.targetPath, s.topk);
  }
  return imgsearch::searchClassic(s.dbDir, s.targetPath, feature, metric, s.topk);
}


struct AsyncSearch {
  std::mutex mu;
  std::condition_variable cv;
  bool quit = false;
  bool requested = false;
  bool busy = false;

  GuiState requestState;
  std::vector<imgsearch::SearchMatch> latest;
  std::string lastError;

  std::thread worker;

  void start() {
    worker = std::thread([this]() {
      std::unique_lock<std::mutex> lk(mu);
      while (true) {
        cv.wait(lk, [&]() { return quit || requested; });
        if (quit) break;
        GuiState s = requestState;
        requested = false;
        busy = true;
        lastError.clear();
        lk.unlock();

        std::vector<imgsearch::SearchMatch> res;
        try {
          res = runSearch(s);
        } catch (const std::exception& e) {
          lastError = e.what();
        }

        lk.lock();
        latest = std::move(res);
        busy = false;
      }
    });
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lk(mu);
      quit = true;
      requested = true;
    }
    cv.notify_all();
    if (worker.joinable()) worker.join();
  }

  void request(const GuiState& s) {
    {
      std::lock_guard<std::mutex> lk(mu);
      requestState = s;
      requested = true;
    }
    cv.notify_one();
  }

  std::vector<imgsearch::SearchMatch> snapshot(bool* outBusy, std::string* outErr) {
    std::lock_guard<std::mutex> lk(mu);
    if (outBusy) *outBusy = busy;
    if (outErr) *outErr = lastError;
    return latest;
  }
};

static cv::Mat render(const GuiState& s,
                      const std::vector<imgsearch::SearchMatch>& matches,
                      bool busy,
                      const std::string& statusMsg) {
  const int cellW = 240;
  const int cellH = 180;
  const int headerH = 60;

  int k = std::max(1, s.topk);
  int cols = std::max(1, std::min(4, k));
  int rows = (k + cols - 1) / cols;

  const int canvasW = 2 * cellW + cols * cellW;
  const int canvasH = headerH + std::max(cellH, rows * cellH);

  cv::Mat canvas(canvasH, canvasW, CV_8UC3, cv::Scalar(35, 35, 35));

  
  cv::Mat target = loadAndFit(s.targetPath, 2 * cellW, cellH);
  target.copyTo(canvas(cv::Rect(0, headerH, 2 * cellW, cellH)));

  
  for (int i = 0; i < k; ++i) {
    int r = i / cols;
    int c = i % cols;
    cv::Rect roi(2 * cellW + c * cellW, headerH + r * cellH, cellW, cellH);
    cv::Mat cell = canvas(roi);

    if (i < (int)matches.size()) {
      cv::Mat m = loadAndFit(matches[i].path, cellW, cellH);
      m.copyTo(cell);

      
      cv::rectangle(cell, {0, 0}, {cellW, 22}, cv::Scalar(0, 0, 0), cv::FILLED);
      std::ostringstream ss;
      ss << "#" << (i + 1) << "  d=" << cv::format("%.4f", matches[i].score);
      cv::putText(cell, ss.str(), {8, 16}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {255, 255, 255}, 1);
    } else {
      cv::rectangle(cell, {0, 0}, {cellW, cellH}, cv::Scalar(50, 50, 50), cv::FILLED);
    }

    cv::rectangle(canvas, roi, cv::Scalar(90, 90, 90), 1);
  }

  std::string line1 = "Target: " + basename(s.targetPath) + "    DB: " + s.dbDir;
  std::string line2 = "Feature: " + s.feature + "    Metric: " + s.metric + "    TopK: " + std::to_string(s.topk);
  if (busy) line2 += "    [searching…]";
  if (!statusMsg.empty()) line2 += "    " + statusMsg;
  
  putHeader(canvas, line1, line2);
  return canvas;
}

static void usage() {
  std::cerr
      << "cbir_gui usage:\n"
      << "  ./cbir_gui [--db <dir>] [--target <img>] [--topk N] [--feature <name>] [--metric <name>] [--emb_csv <file>]\n\n"
      << "feature names: center7x7, rg16, rgb8, rgb8_topbottom, colortexture, embedding_resnet18, task7\n"
      << "metric names : ssd, histint, multihist, cosine, colortexture, task7_bhatt\n\n"
      << "In the GUI:\n"
      << "  d = choose dataset directory (zenity if available)\n"
      << "  t = choose target image (zenity if available)\n"
      << "  f/F = cycle feature\n"
      << "  m/M = cycle metric\n"
      << "  +/- = change TopK\n"
      << "  r = run search\n"
      << "  q or ESC = quit\n";
}

static bool parseArgs(int argc, char** argv, GuiState& s) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const std::string& flag) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        return false;
      }
      return true;
    };

    if (a == "--help" || a == "-h") {
      usage();
      return false;
    } else if (a == "--db") {
      if (!need(a)) return false;
      s.dbDir = argv[++i];
    } else if (a == "--target") {
      if (!need(a)) return false;
      s.targetPath = argv[++i];
    } else if (a == "--topk") {
      if (!need(a)) return false;
      s.topk = std::max(1, std::stoi(argv[++i]));
    } else if (a == "--feature") {
      if (!need(a)) return false;
      s.feature = argv[++i];
    } else if (a == "--metric") {
      if (!need(a)) return false;
      s.metric = argv[++i];
    } else if (a == "--emb_csv" || a == "--embeddings") {
      if (!need(a)) return false;
      s.embCsv = argv[++i];
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      return false;
    }
  }
  

  if (!contains(featureOrder(), s.feature)) {
    std::cerr << "Unknown feature: " << s.feature << "\n";
    return false;
  }
  if (!contains(metricOrder(), s.metric)) {
    std::cerr << "Unknown metric: " << s.metric << "\n";
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  GuiState s;
  if (!parseArgs(argc, argv, s)) return 1;

  AsyncSearch async;
  async.start();
  async.request(s);  

  std::vector<imgsearch::SearchMatch> matches;
  bool busy = true;
  std::string statusMsg;

  cv::namedWindow("CBIR", cv::WINDOW_AUTOSIZE);

  while (true) {
    
    std::string err;
    matches = async.snapshot(&busy, &err);
    if (!err.empty()) {
      statusMsg = std::string("Error: ") + err;
    } else {
      statusMsg.clear();
    }
    cv::Mat view = render(s, matches, busy, statusMsg);
    if (busy) {
      cv::putText(view, "Searching...", {20, 95}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 0}, 3, cv::LINE_AA);
      cv::putText(view, "Searching...", {20, 95}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 1, cv::LINE_AA);
    }
    cv::imshow("CBIR", view);
    int key = cv::waitKey(30);
    if (key < 0) continue;
    key &= 0xff;

    if (key == 'q' || key == 27) break;
    if (key == 'f') {
      cycle(s.feature, featureOrder(), +1);
      async.request(s);
    } else if (key == 'F') {
      cycle(s.feature, featureOrder(), -1);
      async.request(s);
    } else if (key == 'm') {
      cycle(s.metric, metricOrder(), +1);
      async.request(s);
    } else if (key == 'M') {
      cycle(s.metric, metricOrder(), -1);
      async.request(s);
    } else if (key == '+' || key == '=') {
      s.topk = std::min(25, s.topk + 1);
      async.request(s);
    } else if (key == '-' || key == '_') {
      s.topk = std::max(1, s.topk - 1);
      async.request(s);
    } else if (key == 'r') {
      async.request(s);
    } else if (key == 't') {
      
      std::string p = pickFileDialog("Select target image");
      if (!p.empty()) {
        s.targetPath = p;
        async.request(s);
      }
    } else if (key == 'd') {
      
      std::string p = pickDirDialog("Select dataset directory");
      if (!p.empty()) {
        s.dbDir = p;
        
        if (s.targetPath.empty()) {
          auto imgs = imgsearch::collectImageFiles(s.dbDir);
          if (!imgs.empty()) s.targetPath = imgs.front();
        }
        async.request(s);
      }
    }
  }
  async.stop();
  return 0;
}
