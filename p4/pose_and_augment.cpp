/*
  Varun Raghavendra
  Spring 2026
  CS 5330 Computer Vision
  Pose Estimation and Augmented Reality Overlay using Checkerboard Tracking
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

static void drawCircle3D(cv::Mat &img,
                         float cx, float cy, float cz,
                         float r,
                         const cv::Mat &rvec, const cv::Mat &tvec,
                         const cv::Mat &K, const cv::Mat &D,
                         const cv::Scalar &color,
                         int thickness = 2,
                         int nSeg = 16)
{
    /* Approximate a 3D circle by sampling points on its boundary in world space.
       After projection, consecutive points are connected to form a smooth overlay. */
    std::vector<cv::Point3f> pts3d;
    pts3d.reserve(nSeg);
    for (int i = 0; i < nSeg; ++i) {
        float angle = float(i) * 2.0f * float(CV_PI) / float(nSeg);
        pts3d.emplace_back(cx + r * std::cos(angle),
                           cy + r * std::sin(angle),
                           cz);
    }
    std::vector<cv::Point2f> pts2d;
    cv::projectPoints(pts3d, rvec, tvec, K, D, pts2d);
    for (int i = 0; i < nSeg; ++i) {
        cv::line(img, pts2d[i], pts2d[(i + 1) % nSeg], color, thickness);
    }
}

static cv::Point2f project1(cv::Point3f p3,
                             const cv::Mat &rvec, const cv::Mat &tvec,
                             const cv::Mat &K, const cv::Mat &D)
{
    /* This helper function keeps repeated single-point projection calls concise.
       It is used throughout the virtual object drawing logic. */
    std::vector<cv::Point3f> in  = {p3};
    std::vector<cv::Point2f> out;
    cv::projectPoints(in, rvec, tvec, K, D, out);
    return out[0];
}

static void drawStickFigure(cv::Mat &img,
                             float ox, float oy,
                             float scale,
                             const cv::Mat &rvec, const cv::Mat &tvec,
                             const cv::Mat &K, const cv::Mat &D,
                             const cv::Scalar &bodyClr,
                             const cv::Scalar &legClr,
                             const cv::Scalar &accentClr,
                             bool isGirl)
{
    /* Building the boy and girl stick character in checkerboard world coordinates.
       This makes the augmentation respond correctly to camera motion and board orientation. */
    const float s  = scale;

    float zFeet    = 0.0f;
    float zKnee    = s * 0.28f;
    float zHip     = s * 0.48f;
    float zWaist   = s * 0.52f;
    float zShoulder= s * 0.72f;
    float zNeck    = s * 0.78f;
    float zChin    = s * 0.82f;
    float zHead    = s * 0.93f;
    float rHead    = s * 0.11f;

    float armSpread = s * 0.22f;
    float legSpread = s * 0.14f;

    float shoulderW = s * 0.18f;

    auto P = [&](float x, float y, float z) -> cv::Point2f {
        return project1({x, y, z}, rvec, tvec, K, D);
    };

    int thick = 2;

    drawCircle3D(img, ox, oy, zHead, rHead, rvec, tvec, K, D, bodyClr, thick);

    cv::line(img, P(ox, oy, zNeck), P(ox, oy, zChin), bodyClr, thick);

    cv::line(img, P(ox, oy, zNeck), P(ox, oy, zWaist), bodyClr, thick);

    float elbowZ = zShoulder - s * 0.14f;
    float wristZ = zShoulder - s * 0.28f;
    cv::line(img, P(ox, oy, zShoulder),
                  P(ox - shoulderW, oy, elbowZ), bodyClr, thick);
    cv::line(img, P(ox - shoulderW, oy, elbowZ),
                  P(ox - shoulderW * 1.3f, oy, wristZ), accentClr, thick);
    cv::line(img, P(ox, oy, zShoulder),
                  P(ox + shoulderW, oy, elbowZ), bodyClr, thick);
    cv::line(img, P(ox + shoulderW, oy, elbowZ),
                  P(ox + shoulderW * 1.3f, oy, wristZ), accentClr, thick);

    if (!isGirl) {
        cv::line(img, P(ox, oy, zHip),
                      P(ox - legSpread, oy, zKnee), legClr, thick);
        cv::line(img, P(ox - legSpread, oy, zKnee),
                      P(ox - legSpread, oy, zFeet), legClr, thick);
        cv::line(img, P(ox, oy, zHip),
                      P(ox + legSpread, oy, zKnee), legClr, thick);
        cv::line(img, P(ox + legSpread, oy, zKnee),
                      P(ox + legSpread, oy, zFeet), legClr, thick);
        cv::line(img, P(ox - legSpread, oy, zHip),
                      P(ox + legSpread, oy, zHip), legClr, thick);
    } else {
        float skirtW  = s * 0.30f;
        float skirtBot= zKnee;
        cv::line(img, P(ox - shoulderW * 0.5f, oy, zWaist),
                      P(ox - skirtW, oy, skirtBot), legClr, thick);
        cv::line(img, P(ox + shoulderW * 0.5f, oy, zWaist),
                      P(ox + skirtW, oy, skirtBot), legClr, thick);
        cv::line(img, P(ox - skirtW, oy, skirtBot),
                      P(ox + skirtW, oy, skirtBot), legClr, thick);
        cv::line(img, P(ox - legSpread * 0.7f, oy, skirtBot),
                      P(ox - legSpread * 0.7f, oy, zFeet), accentClr, thick);
        cv::line(img, P(ox + legSpread * 0.7f, oy, skirtBot),
                      P(ox + legSpread * 0.7f, oy, zFeet), accentClr, thick);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <calibration.yml>\n";
        return 1;
    }

    /* Load the previously estimated camera intrinsics and distortion values.
       These parameters are required to project 3D board geometry back into the image. */
    cv::FileStorage fs(argv[1], cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Could not open: " << argv[1] << "\n";
        return 1;
    }
    cv::Mat cameraMatrix, distCoeffs;
    fs["camera_matrix"]           >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    if (cameraMatrix.empty() || distCoeffs.empty()) {
        std::cerr << "Invalid calibration data.\n";
        return 1;
    }
    std::cout << "Loaded calibration.\nCamera matrix:\n" << cameraMatrix
              << "\nDistortion:\n" << distCoeffs << "\n";

    const cv::Size boardSize(9, 6);
    std::vector<cv::Vec3f> boardObjPts;
    boardObjPts.reserve(boardSize.area());
    for (int r = 0; r < boardSize.height; ++r)
        for (int c = 0; c < boardSize.width; ++c)
            boardObjPts.emplace_back(float(c), float(-r), 0.0f);

    std::vector<cv::Point3f> axisPoints = {
        {0, 0, 0}, {2, 0, 0}, {0, -2, 0}, {0, 0, 2}
    };

    std::vector<cv::Point3f> outerCorners = {
        {0,0,0}, {8,0,0}, {8,-5,0}, {0,-5,0}
    };

    bool showAxes    = true;
    bool showCorners = true;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open default camera.\n";
        return 1;
    }

    std::cout << "\nAR running.  Controls: q=quit  a=toggle axes  c=toggle corners\n";

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        /* Detect the checkerboard and refine the corner coordinates in every frame.
           These image measurements are the basis for recovering the board pose. */
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(
            gray, boardSize, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                             30, 0.1));

            /* solvePnP estimates the rigid transform between board coordinates and the camera.
               Once rvec and tvec (rotation and transformation vectors) are known, every virtual 3D object can be rendered consistently. */
            cv::Mat rvec, tvec;
            bool ok = cv::solvePnP(boardObjPts, corners,
                                   cameraMatrix, distCoeffs,
                                   rvec, tvec);
            if (ok) {
                double angle = cv::norm(rvec) * 180.0 / CV_PI;
                std::cout << std::fixed << std::setprecision(2)
                          << "rvec=[" << rvec.at<double>(0) << ", "
                          << rvec.at<double>(1) << ", " << rvec.at<double>(2)
                          << "]  angle=" << angle << "deg  "
                          << "tvec=[" << tvec.at<double>(0) << ", "
                          << tvec.at<double>(1) << ", " << tvec.at<double>(2) << "]\n";

                /* Reproject the board outline points to visually verify pose quality.
                   If the pose is accurate, these projected corners should align with the checkerboard. */
                if (showCorners) {
                    std::vector<cv::Point2f> cp;
                    cv::projectPoints(outerCorners, rvec, tvec,
                                      cameraMatrix, distCoeffs, cp);
                    for (auto &pt : cp)
                        cv::circle(frame, pt, 8, cv::Scalar(0, 255, 255), -1);
                    for (int i = 0; i < 4; ++i)
                        cv::line(frame, cp[i], cp[(i+1)%4],
                                 cv::Scalar(0, 255, 255), 1);
                }

                /* Draw 3D axes to show the board coordinate frame.
                   This is the main reference coordinate system for the renders  */
                if (showAxes) {
                    std::vector<cv::Point2f> ap;
                    cv::projectPoints(axisPoints, rvec, tvec,
                                      cameraMatrix, distCoeffs, ap);
                    cv::arrowedLine(frame, ap[0], ap[1], cv::Scalar(0,   0, 255), 3);
                    cv::arrowedLine(frame, ap[0], ap[2], cv::Scalar(0, 255,   0), 3);
                    cv::arrowedLine(frame, ap[0], ap[3], cv::Scalar(255, 0,   0), 3);
                }

                /* Render two virtual characters in board coordinates so they move with the target.
                   Using the same recovered pose for all projected landmarks keeps the overlay stable. */
                drawStickFigure(frame,
                                1.0f, -2.5f,
                                3.5f,
                                rvec, tvec, cameraMatrix, distCoeffs,
                                cv::Scalar(255, 200,   0),
                                cv::Scalar(  0, 200, 255),
                                cv::Scalar(  0, 180, 255),
                                false);

                drawStickFigure(frame,
                                5.5f, -2.5f,
                                3.5f,
                                rvec, tvec, cameraMatrix, distCoeffs,
                                cv::Scalar(255,   0, 200),
                                cv::Scalar(180,   0, 255),
                                cv::Scalar(255, 180, 255),
                                true);
            }
        } else {
            cv::putText(frame, "No checkerboard detected",
                        cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                        0.7, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Augmented Reality", frame);
        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27 || key == 'q') break;
        if (key == 'a') { showAxes    = !showAxes;    std::cout << "Axes: "    << (showAxes    ? "on" : "off") << "\n"; }
        if (key == 'c') { showCorners = !showCorners; std::cout << "Corners: " << (showCorners ? "on" : "off") << "\n"; }
    }

    return 0;
}
