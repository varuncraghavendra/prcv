#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat image = imread(argv[1]);
    if (image.empty()) return 1;

    imshow("image", image);

    while (waitKey(30) != 'q') {}

    return 0;
}
