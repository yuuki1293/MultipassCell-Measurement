#include<stdio.h>
#include<string>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace traits;

Mat remove_area(const Mat image, int min, int max);
Mat crop(const Mat image, int left, int top, int right, int bottom);

int main(){
    std::string image_path = "many_circle_bin.png";
    Mat src = imread(image_path, IMREAD_COLOR);
    Mat gray, labelImg, stats, centroids, Dst;
    src.copyTo(Dst);
    cvtColor(src, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 5);

    gray = remove_area(gray, 200, 2000);
    gray = crop(gray, 400, 400, 400, 400);
    vector<Vec3f> circles;
    int n = connectedComponentsWithStats(gray, labelImg, stats, centroids);
    
    // 重心
    for (size_t i = 1; i < n; i++){
        double *param = centroids.ptr<double>(i);
        int x = static_cast<int>(param[0]);
        int y = static_cast<int>(param[1]);

        circle(Dst, Point(x, y), 3, Scalar(0, 0, 255), -1);
    }

    // 面積値の出力
    for (size_t i = 1; i < n; i++) {
        int *param = stats.ptr<int>(i);
        std::print("area {} = {} ({}, {})\n", i, param[ConnectedComponentsTypes::CC_STAT_AREA], param[ConnectedComponentsTypes::CC_STAT_LEFT], param[ConnectedComponentsTypes::CC_STAT_TOP]);

        int x = param[ConnectedComponentsTypes::CC_STAT_LEFT];
        int y = param[ConnectedComponentsTypes::CC_STAT_TOP];
        stringstream num;
        num << i;
        putText(Dst, num.str(), cv::Point(x + 5, y + 10), FONT_HERSHEY_COMPLEX, 0.7, Scalar(255, 0, 0), 2);
    }

    imshow("Display window", Dst);
    int k = waitKey(0);
    return 0;
}

// 特定の大きさのオブジェクト以外を削除
Mat remove_area(const Mat image, int min, int max){
    Mat hierarchy, result;
    vector<vector<Point>> contours;
    image.copyTo(result);

    findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if(!(min <= area && area <= max)){
            drawContours(result, contours, (int)i, Scalar(0), -1);
        }
    }
    
    return result;
}

// 端を黒塗りする
Mat crop(const Mat image, int left, int top, int right, int bottom){
    Mat result;
    image.copyTo(result);

    rectangle(result, Point(0, 0), Point(left, result.rows), Vec3b(0, 0, 0), FILLED);
    rectangle(result, Point(0, 0), Point(result.cols, top), Vec3b(0, 0, 0), FILLED);
    rectangle(result, Point(result.cols - right, 0), Point(result.cols, result.rows), Vec3b(0, 0, 0), FILLED);
    rectangle(result, Point(0, result.rows - bottom), Point(result.cols, result.rows), Vec3b(0, 0, 0), FILLED);
    
    return result;
}