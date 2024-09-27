#include<stdio.h>
#include<string>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace traits;

int main(){
    std::string image_path = "many_circle_bin.png";
    Mat src = imread(image_path, IMREAD_COLOR);
    Mat gray, labelImg, stats, centroids, Dst;
    src.copyTo(Dst);
    cvtColor(src, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 5);

    vector<Vec3f> circles;
    int n = connectedComponentsWithStats(gray, labelImg, stats, centroids);
    
    /* 重心 */
    for (size_t i = 1; i < n; i++){
        double *param = centroids.ptr<double>(i);
        int x = static_cast<int>(param[0]);
        int y = static_cast<int>(param[1]);

        circle(Dst, Point(x, y), 3, Scalar(0, 0, 255), -1);
    }

    imshow("Display window", Dst);
    int k = waitKey(0);
    return 0;
}