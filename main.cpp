#include <stdio.h>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

#define REAL_LEN 88900 // μm
#define THRESHOLD 180 // しきい値

using namespace cv;
using namespace std;
using namespace traits;

Mat projectiveT(Mat src);
Mat binaryT(Mat src);
Mat remove_area(const Mat image, int min, int max);
Mat crop(const Mat image, int left, int top, int right, int bottom);
string toSVG(const Point *centers, int n);

int main()
{
    std::string image_path = "sample.JPG";
    Mat src = imread(image_path, IMREAD_COLOR);
    Mat process, labelImg, stats, centroids, labelingBin, labeling;

    process = projectiveT(src);
    process = binaryT(process);
    process = crop(process, 400, 400, 400, 400); // 縁から400pxを削除
    process = remove_area(process, 100, 3000); // 100~200ピクセル以外の図形を削除

    bitwise_not(process, labelingBin); // ラベリング用画像
    cvtColor(labelingBin, labeling, COLOR_GRAY2RGB);
    int n = connectedComponentsWithStats(process, labelImg, stats, centroids);
    Point centers[n];
    // 重心
    for (size_t i = 1; i < n; i++)
    {
        double *param = centroids.ptr<double>(i);
        int x = param[0];
        int y = param[1];

        centers[i] = Point(x, y);
        circle(labeling, Point(x, y), 10, Scalar(255, 0, 0), -1);
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
        putText(labeling, num.str(), cv::Point(x, y), FONT_HERSHEY_COMPLEX, 1.5, Scalar(0, 0, 255), 2);
    }

    // 単位をピクセルからμｍに変換
    for (size_t i = 1; i < n; i++)
    {
        int um_per_pix = REAL_LEN / process.cols;

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

    imwrite("label.png", labeling);
    imshow("Display window", labeling);
    int k = waitKey(0);
    return 0;
}

// 射影変換
Mat projectiveT(Mat src){
    vector<Point2f> src_pts;
    src_pts.push_back(Point2f(1468, 528)); // ↖
    src_pts.push_back(Point2f(3322, 577)); // ↗
    src_pts.push_back(Point2f(1474, 2634)); // ↙
    src_pts.push_back(Point2f(3316, 2516)); // ↘

    vector<Point2f> dst_pts;
    dst_pts.push_back(Point2f(0, 0)); // ↖
    dst_pts.push_back(Point2f(2000, 0)); // ↗
    dst_pts.push_back(Point2f(0, 2000)); // ↙
    dst_pts.push_back(Point2f(2000, 2000)); // ↘

    Mat M = getPerspectiveTransform(src_pts, dst_pts);
    Mat result;
    warpPerspective(src, result, M, Size(2000, 2000));

    return result;
}

// 2値化
Mat binaryT(Mat src){
    vector<Mat> bgr_channels;
    split(src, bgr_channels);
    Mat green_channel = bgr_channels[1];
    Mat result;
    inRange(green_channel, Scalar(THRESHOLD), Scalar(255), result); // 2値化
    
    return result;
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

// SVG形式に変換
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