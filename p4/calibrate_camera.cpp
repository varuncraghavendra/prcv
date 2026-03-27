/*
  Varun Raghavendra
  Spring 2026
  CS 5330 Computer Vision
  Camera Calibration using Checkerboard Corner Detection and Intrinsic Estimation
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

static void printUsage() {
    std::cout << "\n=== Camera Calibration Tool ===\n"
              << "  s       : Save current detected corners (need target visible)\n"
              << "  k       : Run calibration  (need >= 5 saved frames)\n"
              << "  w       : Write calibration.yml\n"
              << "  q       : Quit\n\n";
}

int main() {
    const cv::Size boardSize(9, 6);

    std::vector<std::vector<cv::Point2f>> corner_list;
    std::vector<std::vector<cv::Vec3f>>   point_list;
    cv::Size imageSize;

    cv::Mat lastCameraMatrix, lastDistCoeffs;
    bool haveCalibration = false;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open webcamera.\n";
        return 1;
    }

    std::cout << "Camera Calibration  (9x6 checkerboard)\n";
    printUsage();

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Empty frame received.\n";
            break;
        }
        imageSize = frame.size();

        /* Convert the frames to grayscale before checkerboard detection.
           This keeps the corner extraction pipeline consistent and efficient. */
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(
            gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            /* Refine each detected corner to sub-pixel precision for better calibration.
               Accurate image points directly improve the quality of the estimated intrinsics. */
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                             30, 0.1));
            cv::drawChessboardCorners(frame, boardSize, corners, found);

            std::ostringstream oss;
            oss << "Corners: " << corners.size()
                << "  First: (" << std::fixed << std::setprecision(1)
                << corners[0].x << ", " << corners[0].y << ")";
            cv::putText(frame, oss.str(), cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(frame, "No checkerboard found", cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        std::ostringstream saved;
        saved << "Saved frames: " << corner_list.size();
        cv::putText(frame, saved.str(), cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

        if (haveCalibration) {
            cv::putText(frame, "Calibration ready - press w to save",
                        cv::Point(10, imageSize.height - 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        }

        cv::imshow("Calibration", frame);
        char key = static_cast<char>(cv::waitKey(1));

        if (key == 27 || key == 'q') {
            break;

        } else if (key == 's' || key == 'S') {
            /* Save the current 2D image corners together with their matching 3D board points. */
            if (found && static_cast<int>(corners.size()) == boardSize.area()) {
                corner_list.push_back(corners);

                std::vector<cv::Vec3f> point_set;
                point_set.reserve(boardSize.area());
                for (int r = 0; r < boardSize.height; ++r)
                    for (int c = 0; c < boardSize.width; ++c)
                        point_set.emplace_back(float(c), float(-r), 0.0f);
                point_list.push_back(point_set);

                std::cout << "[s] Saved frame " << point_list.size()
                          << "  (" << corners.size() << " corners)\n";
            } else {
                std::cout << "[s] Checkerbord not clearly seen in the video frame.\n";
            }

        } else if (key == 'k' || key == 'K') {
            /* Run full camera calibration after collecting atleast 5 checkerboard views.
               The below code estimates the intrinsic matrix and lens distortion parameters. Reprojection error will never be exactly 0px due to image noise and lens blur.
               The reprojection error should be less than 1px, as mentioned by a person in a Stack Overflow thread I have cited in the main report */
            if (corner_list.size() < 5) {
                std::cout << "[k] Need at least 5 saved frames (have "
                          << corner_list.size() << ").\n";
            } else {
                cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
                cameraMatrix.at<double>(0, 2) = imageSize.width  * 0.5;
                cameraMatrix.at<double>(1, 2) = imageSize.height * 0.5;
                cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);

                std::cout << "\n--- Before calibration ---\n"
                          << "Camera matrix:\n" << cameraMatrix << "\n"
                          << "Distortion coefficients:\n" << distCoeffs << "\n";

                std::vector<cv::Mat> rvecs, tvecs;
                double rms = cv::calibrateCamera(
                    point_list, corner_list, imageSize,
                    cameraMatrix, distCoeffs, rvecs, tvecs,
                    cv::CALIB_FIX_ASPECT_RATIO,
                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                     100, DBL_EPSILON));

                std::cout << "\n--- After calibration ---\n"
                          << "RMS reprojection error: " << rms << " px\n"
                          << "Camera matrix:\n" << cameraMatrix << "\n"
                          << "Distortion coefficients:\n" << distCoeffs << "\n";

                lastCameraMatrix = cameraMatrix.clone();
                lastDistCoeffs   = distCoeffs.clone();
                haveCalibration  = true;
            }

        } else if (key == 'w' || key == 'W') {
            /* Save the latest calibration to a YAML file for later pose estimation.
               This separates one-time camera calibration from downstream AR tasks. */
            if (haveCalibration) {
                cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE);
                if (fs.isOpened()) {
                    fs << "camera_matrix"           << lastCameraMatrix;
                    fs << "distortion_coefficients" << lastDistCoeffs;
                    fs.release();
                    std::cout << "[w] calibration.yml written.\n";
                } else {
                    std::cerr << "[w] Could not open calibration.yml for writing.\n";
                }
            } else {
                std::cout << "[w] No calibration yet\n";
            }
        }
    }

    return 0;
}
