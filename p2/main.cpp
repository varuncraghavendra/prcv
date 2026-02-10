/*
  Varun Raghavendra
  Spring 2026
  CS 5330 Computer Vision

  "Program entrypoint: parses CLI args and launches CLI search or OpenCV GUI."

*/

#include "core.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

struct Args {
  std::string dbDir;
  std::string target;
  std::string feature;
  std::string metric;
  int topk = 3;
  std::string embCsv;    
  std::string computeCsv; 
  std::string searchCsv;  
  bool gui = false;       
  bool task6 = false;     
  std::string task6Out = "task6_report.txt";
};

// Prints help/usage for both CLI and GUI invocations.

static void usage() {
  std::cout <<
  "Usage:\n"
  "  cbir_query --db <dir> --target <img>\n"
  "            --feature <center7x7|rg16|rgb8|rgb8_topbottom|colortexture|task7|embedding_resnet18>\n"
  "            --metric  <ssd|histint|multihist|cosine|colortexture|task7_bhatt>\n"
  "            --topk <N>\n"
  "            [--emb_csv <ResNet18_olym.csv>]\n"
  "            [--compute_csv <out.csv>]    (precompute features for all images)\n"
  "            [--search_csv <in.csv>]      (search using precomputed features)\n"
  "            [--gui]                      (display query and matches)\n"
  "            [--task6 [--task6_out <file>]] (generate Task 6 comparison report)\n";
}

// Parses CLI flags for mode (CLI vs GUI), task selection, and paths.

static bool parseArgs(int argc, char** argv, Args& a) {
  for (int i=1; i<argc; ++i) {
    std::string k = argv[i];
    auto need = [&](std::string* out)->bool{
      if (i+1 >= argc) return false;
      *out = argv[++i];
      return true;
    };

    if (k=="--db") { if(!need(&a.dbDir)) return false; }
    else if (k=="--target") { if(!need(&a.target)) return false; }
    else if (k=="--feature") { if(!need(&a.feature)) return false; }
    else if (k=="--metric") { if(!need(&a.metric)) return false; }
    else if (k=="--topk") {
      std::string v; if(!need(&v)) return false;
      a.topk = std::max(1, std::stoi(v));
    }
    else if (k=="--emb_csv") { if(!need(&a.embCsv)) return false; }
    else if (k=="--compute_csv") { if(!need(&a.computeCsv)) return false; }
    else if (k=="--search_csv") { if(!need(&a.searchCsv)) return false; }
    else if (k=="--gui") { a.gui = true; }
    else if (k=="--task6") { a.task6 = true; }
    else if (k=="--task6_out") { if(!need(&a.task6Out)) return false; }
    else if (k=="--help" || k=="-h") { usage(); std::exit(0); }
    else { std::cerr << "Unknown arg: " << k << "\n"; return false; }
  }

  
  
  if (a.task6) {
    return !a.dbDir.empty() && !a.embCsv.empty();
  }

  
  
  
  
  
  if (!a.computeCsv.empty()) {
    return !a.dbDir.empty() && !a.feature.empty();
  }
  if (!a.searchCsv.empty()) {
    return !a.target.empty() && !a.feature.empty() && !a.metric.empty();
  }
  
  if (a.dbDir.empty() || a.target.empty() || a.feature.empty() || a.metric.empty())
    return false;
  
  
  if ((a.feature == "resnet18" || a.feature == "embedding_resnet18") && a.embCsv.empty())
    return false;
  return true;
}

int main(int argc, char** argv) {
  Args args;
  if (!parseArgs(argc, argv, args)) {
    usage();
    return 1;
  }

  
  if (args.task6) {
    if (!fs::exists(args.dbDir) || !fs::is_directory(args.dbDir)) {
      std::cerr << "ERROR: --db path does not exist or is not a directory: " << args.dbDir << "\n";
      return 2;
    }
    
    auto dbImgs = imgsearch::collectImageFiles(args.dbDir);
    if (dbImgs.empty()) {
      std::cerr << "ERROR: No images found under --db.\n";
      return 3;
    }

    auto findById = [&](const std::string& id) -> std::string {
      
      for (const auto& p : dbImgs) {
        std::string b = imgsearch::fileBasename(p);
        if (b.find(id) != std::string::npos) return p;
      }
      return "";
    };

    const std::vector<std::string> ids = {"1072", "948", "734"};
    std::ofstream out(args.task6Out);
    if (!out.is_open()) {
      std::cerr << "ERROR: Could not write report to: " << args.task6Out << "\n";
      return 4;
    }
    out << "Task 6: DNN Embeddings vs Classic Features\n";
    out << "DB: " << args.dbDir << "\n";
    out << "Embeddings CSV: " << args.embCsv << "\n\n";

    for (const auto& id : ids) {
      std::string target = findById(id);
      if (target.empty()) {
        out << "--- Target id " << id << ": NOT FOUND in DB ---\n\n";
        continue;
      }
      out << "--- Target: " << imgsearch::fileBasename(target) << " ---\n";

      
      auto emb = imgsearch::searchEmbeddings(args.embCsv, args.dbDir, target, "cosine", 10);
      out << "DNN (ResNet18 embeddings, cosine distance)\n";
      for (size_t i = 0; i < emb.size(); ++i) {
        out << "  #" << (i + 1) << "  " << imgsearch::fileBasename(emb[i].path) << "  d=" << emb[i].score << "\n";
      }
      out << "\n";

      
      auto cls = imgsearch::searchClassic(args.dbDir, target, "colortexture", "colortexture", 10);
      out << "Classic (RGB hist + Sobel magnitude hist, equal-weight distance)\n";
      for (size_t i = 0; i < cls.size(); ++i) {
        out << "  #" << (i + 1) << "  " << imgsearch::fileBasename(cls[i].path) << "  d=" << cls[i].score << "\n";
      }
      out << "\n";

      
      out << "Notes: Compare whether the embedding returns semantically similar scenes/objects,\n"
          << "while the classic feature may over-focus on colour/texture statistics. Use the\n"
          << "two ranked lists above to support your discussion in the write-up.\n\n";
    }

    std::cout << "Wrote Task 6 report to " << args.task6Out << "\n";
    return 0;
  }

  
  if (!fs::exists(args.dbDir) || !fs::is_directory(args.dbDir)) {
    std::cerr << "ERROR: --db path does not exist or is not a directory: "
              << args.dbDir << "\n";
    return 2;
  }
  if (!args.task6 && (!fs::exists(args.target) || fs::is_directory(args.target))) {
    std::cerr << "ERROR: --target file does not exist: " << args.target
              << "\n";
    return 3;
  }

  
  if (!args.task6) {
    cv::Mat t = cv::imread(args.target, cv::IMREAD_COLOR);
    if (t.empty()) {
      std::cerr << "ERROR: OpenCV failed to read --target image: "
                << args.target << "\n"
                << "Tip: verify the path with: ls -l <path> and file <path>\n";
      return 4;
    }
  }

  
  auto dbImgs = imgsearch::collectImageFiles(args.dbDir);
  std::cerr << "DB scan: found " << dbImgs.size() << " image(s) under "
            << args.dbDir << "\n";
  if (dbImgs.empty()) {
    std::cerr << "ERROR: No images found under --db.\n"
              << "Tip: check the directory contains image files (.jpg/.png/.jpeg/...) and you have permissions.\n";
    return 5;
  }

  std::string featureKey = args.feature;
  std::string metricKey = args.metric;

  // Backwards compatibility: older builds used the name "custom_hybrid"/"custom".
  // Task 7 is now Lab chromaticity histogram +  texture, compared with Bhattacharyya distance.
  if (featureKey == "custom_hybrid" || featureKey == "custom") featureKey = "task7";
  if (featureKey == "resnet18") featureKey = "embedding_resnet18";
  if (featureKey == "colortexture") metricKey = "colortexture";
  if (metricKey == "custom" || metricKey == "task7") metricKey = "task7_bhatt";
  if (featureKey == "task7") metricKey = "task7_bhatt";

  if (!args.computeCsv.empty()) {
    if (!imgsearch::writeClassicFeaturesCsv(args.dbDir, featureKey, args.computeCsv)) {
      std::cerr << "Failed to write features to CSV file " << args.computeCsv << "\n";
      return 2;
    }
    std::cout << "Feature CSV written to " << args.computeCsv << "\n";
    return 0;
  }

  std::vector<imgsearch::SearchMatch> matches;
  
  if (!args.searchCsv.empty()) {
    matches = imgsearch::searchFromClassicCsv(args.searchCsv, args.target, featureKey, metricKey);
  } else if (featureKey == "task7") {
    matches = imgsearch::searchTask7(args.dbDir, args.target, args.topk);
  } else if (featureKey == "embedding_resnet18") {
    matches = imgsearch::searchEmbeddings(args.embCsv, args.dbDir, args.target, metricKey, args.topk);
  } else {
    matches = imgsearch::searchClassic(args.dbDir, args.target, featureKey, metricKey, args.topk);
  }

  if (matches.empty()) {
    std::cerr << "No matches found (check paths / feature / metric / embeddings).\n";
    return 2;
  }
  int K = std::min(args.topk, (int)matches.size());
  for (int i=0; i<K; ++i) {
    std::cout << imgsearch::fileBasename(matches[i].path)
              << "  dist=" << matches[i].score << "\n";
  }

  
  if (args.gui) {
    cv::Mat T = imgsearch::loadBgrImage(args.target);
    if (!T.empty()) {
      cv::imshow("Query", T);
    }
    for (int i=0; i<K; ++i) {
      cv::Mat I = imgsearch::loadBgrImage(matches[i].path);
      if (!I.empty()) {
        std::string winName = std::string("Match ") + std::to_string(i) + " (" + imgsearch::fileBasename(matches[i].path) + ")";
        cv::imshow(winName, I);
      }
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
  return 0;
}