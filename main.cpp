#include<stdio.h>
#include<string>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(){
    std::string image_path = "sample.png";
    Mat src = imread(image_path, IMREAD_COLOR);
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 5);

    vector<Vec3f> circles;
    HoughCircles(src, circles, HOUGH_GRADIENT, 1, 0);
    imshow("Display window", circles);
    int k = waitKey(0);
    return 0;
}