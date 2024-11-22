#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 1. 画像を読み込む
    Mat src = imread("sample.JPG", IMREAD_GRAYSCALE);  // 画像パスを指定
    if (src.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // 2. グレースケール画像に変換
    Mat gray;
    adaptiveThreshold(src, gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2);

    Mat med;
    medianBlur(gray, med, 5);

    Mat dil, ero;
    Mat element = cv::Mat::ones(3, 3, CV_8UC1);
    erode(med, ero, element, Point(-1, -1), 2);
    dilate(ero, dil, element, Point(-1, -1), 2);

    // 3. エッジ検出（Canny）
    Mat edges;
    Canny(med, edges, 50, 50, 3);

    vector<Vec2f> lines;
    HoughLines(ero, lines, 1, CV_PI / 180, 500);

    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 10000 * (-b));
        pt1.y = cvRound(y0 + 10000 * (a));
        pt2.x = cvRound(x0 - 10000 * (-b));
        pt2.y = cvRound(y0 - 10000 * (a));
        line(src, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }
    
    // 6. 結果を表示
    imshow("med", ero);
    imshow("dilero", dil);
    imshow("Detected Lines", src);
    waitKey(0);

    return 0;
}
