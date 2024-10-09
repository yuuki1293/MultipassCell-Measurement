#include <stdio.h>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

#define REAL_LEN 88900 // μm

using namespace cv;
using namespace std;
using namespace traits;

Mat remove_area(const Mat image, int min, int max);
Mat crop(const Mat image, int left, int top, int right, int bottom);
string toSVG(const Point *centers, int n);

int main()
{
    std::string image_path = "many_circle_bin.png";
    Mat src = imread(image_path, IMREAD_COLOR);
    Mat gray, labelImg, stats, centroids, Dst;
    bitwise_not(src, Dst);
    
    vector<Mat> bgr_channels;
    split(src, bgr_channels);
    Mat green_channel = bgr_channels[1];
    inRange(green_channel, Scalar(180), Scalar(255), gray); // 2値化

    gray = remove_area(gray, 100, 2000);   // 100~200ピクセル以外の図形を削除
    gray = crop(gray, 400, 400, 400, 400); // 縁から400ピクセルを削除
    int n = connectedComponentsWithStats(gray, labelImg, stats, centroids);

    Point centers[n];
    // 重心
    for (size_t i = 1; i < n; i++)
    {
        double *param = centroids.ptr<double>(i);
        int x = param[0];
        int y = param[1];

        centers[i] = Point(x, y);
        circle(Dst, Point(x, y), 3, Scalar(0, 255, 0), -1);
    }

    // 面積値の出力
    for (size_t i = 1; i < n; i++)
    {
        int *param = stats.ptr<int>(i);
        int x = param[ConnectedComponentsTypes::CC_STAT_LEFT];
        int y = param[ConnectedComponentsTypes::CC_STAT_TOP];

        std::print("area {} = {} ({}, {})\n", i, param[ConnectedComponentsTypes::CC_STAT_AREA], centers[i].x, centers[i].y);

        stringstream num;
        num << i;
        putText(Dst, num.str(), cv::Point(x, y), FONT_HERSHEY_COMPLEX, 1.0, Scalar(0, 0, 255), 2);
    }

    // 単位をピクセルからμｍに変換
    for (size_t i = 1; i < n; i++)
    {
        int um_per_pix = REAL_LEN / src.cols;

        centers[i] = centers[i] * um_per_pix;
    }

    // Save svg
    std::ofstream out("out.svg");
    out << toSVG(centers, n);
    out.close();

    for (size_t i = 1; i < n; i++)
    {
        print("{} = ({}, {})\n", i, centers[i].x, centers[i].y);
    }

    imwrite("label.png", Dst);
    imshow("Display window", Dst);
    int k = waitKey(0);
    return 0;
}

// 特定の大きさのオブジェクト以外を削除
Mat remove_area(const Mat image, int min, int max)
{
    Mat hierarchy, result;
    vector<vector<Point>> contours;
    image.copyTo(result);

    findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (!(min <= area && area <= max))
        {
            drawContours(result, contours, (int)i, Scalar(0), -1);
        }
    }

    return result;
}

// 端を黒塗りする
Mat crop(const Mat image, int left, int top, int right, int bottom)
{
    Mat result;
    image.copyTo(result);

    rectangle(result, Point(0, 0), Point(left, result.rows), Vec3b(0, 0, 0), FILLED);
    rectangle(result, Point(0, 0), Point(result.cols, top), Vec3b(0, 0, 0), FILLED);
    rectangle(result, Point(result.cols - right, 0), Point(result.cols, result.rows), Vec3b(0, 0, 0), FILLED);
    rectangle(result, Point(0, result.rows - bottom), Point(result.cols, result.rows), Vec3b(0, 0, 0), FILLED);

    return result;
}

string toSVG(const Point *centers, int n)
{
    string s = "<?xml version=\"1.0\"?>";
    s += "<svg width=\"210mm\" height=\"297mm\" xmlns=\"http://www.w3.org/2000/svg\">";

    for (size_t i = 1; i < n; i++)
    {
        s += std::format("<circle cx=\"{}mm\" cy=\"{}mm\" r=\"0.5mm\" stroke=\"black\" fill=\"#fff\" stroke-width=\"1\" />", centers[i].x / 1000., centers[i].y / 1000.);
    }

    s += "</svg>";

    return s;
}