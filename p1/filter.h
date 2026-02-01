#pragma once
#include <opencv2/opencv.hpp>
#include <string>

/*
  filter_rewrite.h
  Declarations for all custom image filters used by vidDisplay.
  Filters operate on BGR images (CV_8UC3) unless otherwise noted.
  Depth Anything V2 helpers are exposed here so vidDisplay can call them.
*/

// Color / tone
int greyscale(cv::Mat &src, cv::Mat &dst);
int sepia(cv::Mat &src, cv::Mat &dst);
int negative(cv::Mat &src, cv::Mat &dst);
int applyBrightnessContrast(const cv::Mat &src, cv::Mat &dst, float contrast, int brightness);

// Blur + quantize
int blur5x5_1(cv::Mat &src, cv::Mat &dst);   // reference (simple)
int blur5x5_2(cv::Mat &src, cv::Mat &dst);   // faster separable version
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

// Gradients
int sobelX3x3(cv::Mat &src, cv::Mat &dst);   // dst is CV_16SC3
int sobelY3x3(cv::Mat &src, cv::Mat &dst);   // dst is CV_16SC3
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst); // dst is CV_8UC3

// Stylized
int emboss(cv::Mat &src, cv::Mat &dst, float dirX = 0.7071f, float dirY = 0.7071f);

// Depth Anything V2 (ONNX Runtime) helper API
bool  da2_init_once(const std::string& model_path);
bool  da2_depth_gray(const cv::Mat& bgr, cv::Mat& depth8u, float scale);
void  da2_reset_calibration();
float da2_face_distance_cm(const cv::Mat& depth8u, const cv::Rect& face, float& conf);
