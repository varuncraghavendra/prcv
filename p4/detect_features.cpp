/*
  Varun Raghavendra
  Spring 2026
  CS 5330 Computer Vision
  Harris Corner Detection for Robust Real-Time Feature Extraction
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

static int g_thresh = 2;
static const int g_maxThresh = 100;

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open webcamera.\n";
        return 1;
    }

    cv::namedWindow("Harris Feature Detection", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Threshold %", "Harris Feature Detection",
                       &g_thresh, g_maxThresh, nullptr);

    std::cout << "Harris Corner Detection\n"
              << "  Move the slider to adjust the detection threshold.\n"
              << "  Press q to quit.\n";

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        /* Harris operates on grayscale image gradients rather than color values.
           The float conversion preserves the response precision for corner scoring. */
        cv::Mat gray, grayF;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(grayF, CV_32F);

        /* Compute the Harris corner response map over the full image.
           High positive responses indicate pixels with strong intensity variation in two directions. */
        cv::Mat harrisResp;
        cv::cornerHarris(grayF, harrisResp, 3, 3, 0.04);

        /* Normalize the raw response so thresholding becomes easier to tune interactively.
           This also keeps the displayed detector behavior consistent across frames. */
        cv::Mat harrisNorm;
        cv::normalize(harrisResp, harrisNorm, 0, 255, cv::NORM_MINMAX, CV_32F);

        double thresh = std::max(1, g_thresh);
        cv::Mat mask = (harrisNorm > float(thresh));

        /* Dilating the mask makes the selected feature pixels easier to visualize. */
        cv::Mat maskD;
        cv::dilate(mask, maskD, cv::Mat());

        int count = 0;
        for (int y = 0; y < maskD.rows; ++y) {
            for (int x = 0; x < maskD.cols; ++x) {
                if (maskD.at<uchar>(y, x)) {
                    cv::circle(frame, cv::Point(x, y), 3,
                               cv::Scalar(0, 0, 255), -1);
                    ++count;
                }
            }
        }

        std::string infoCorners = "Harris corners: " + std::to_string(count);
        cv::putText(frame, infoCorners, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Harris Feature Detection", frame);
        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27 || key == 'q') break;
    }

    return 0;
}
