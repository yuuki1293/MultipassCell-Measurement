#include<stdio.h>
#include<string>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(){
    std::string image_path = "many_circle_bin.png";
    Mat src = imread(image_path, IMREAD_COLOR);
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 5);

    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 50, 100, 10, 5, 30);

    for( size_t i = 0; i < circles.size(); i++ )
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle( src, center, 1, Scalar(0,100,100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle( src, center, radius, Scalar(255,0,255), 3, LINE_AA);
    }

    imshow("Display window", src);
    int k = waitKey(0);
    return 0;
}