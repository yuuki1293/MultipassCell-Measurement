#include <stdio.h>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

#define REAL_LEN 88900 // μm
#define THRESHOLD 50  // しきい値

cv::Mat projectiveT(cv::Mat src);
cv::Mat binaryT(cv::Mat src);
cv::Mat remove_area(const cv::Mat image, int min, int max);
cv::Mat crop(const cv::Mat image, int left, int top, int right, int bottom);
std::string toSVG(const cv::Point *centers, int n);
std::vector<int> sort(cv::Mat src, int n, cv::Mat labels);
void checkSaturation(cv::Mat src);

int main()
{
    std::string image_path = "sample.JPG";
    cv::Mat src = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat process, labelImg, stats, centroids, labelingBin, labeling;

    cv::Mat projective = projectiveT(src);
    imwrite("projectiveT.png", projective);
    checkSaturation(projective);
    process = binaryT(projective);
    imwrite("binaryT.png", process);
    process = crop(process, 400, 400, 400, 400); // 縁から400pxを削除
    imwrite("crop.png", process);
    process = remove_area(process, 100, 3000); // 100~200ピクセル以外の図形を削除
    imwrite("remove_area.png", process);

    bitwise_not(process, labelingBin); // ラベリング用画像
    cvtColor(labelingBin, labeling, cv::COLOR_GRAY2RGB);
    int n = connectedComponentsWithStats(process, labelImg, stats, centroids);
    std::vector<int> sortmap = sort(projective, n, labelImg);

    int *statsArray[n];
    double *centroidsArray[n];
    for (size_t i = 0; i < n; i++)
    {
        statsArray[i] = stats.ptr<int>(sortmap[i]);
        centroidsArray[i] = centroids.ptr<double>(sortmap[i]);
    }

    double diff[n];
    for (size_t i = 1; i < n - 1; i++)
    {
        int *parama = statsArray[i];
        int *paramb = statsArray[i + 1];

        diff[i] = paramb[cv::CC_STAT_AREA] - parama[cv::CC_STAT_AREA];
    }
    {
        int *parama = statsArray[n - 1];
        int *paramb = statsArray[1];

        diff[n - 1] = paramb[cv::CC_STAT_AREA] - parama[cv::CC_STAT_AREA];
    }

    cv::Point centers[n];
    // 重心
    for (size_t i = 1; i < n; i++)
    {
        double *param = centroidsArray[i];
        int x = param[0];
        int y = param[1];

        centers[i] = cv::Point(x, y);
        circle(labeling, cv::Point(x, y), 10, cv::Scalar(255, 0, 0), -1);
    }

    // 面積値の出力
    for (size_t i = 1; i < n; i++)
    {
        int *param = statsArray[i];
        int x = param[cv::CC_STAT_LEFT];
        int y = param[cv::CC_STAT_TOP];

        std::print("area {} = {} ({}, {})\n", i, param[cv::CC_STAT_AREA], centers[i].x, centers[i].y);

        std::stringstream num;
        num << i;
        putText(labeling, num.str(), cv::Point(x, y), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 2);
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
        std::print("{} = ({}, {})\n", i, centers[i].x, centers[i].y);
    }

    imwrite("label.png", labeling);
    imshow("Display window", labeling);
    int k = cv::waitKey(0);
    return 0;
}

// 射影変換
cv::Mat projectiveT(cv::Mat src)
{
    std::vector<cv::Point2f> src_pts;
    src_pts.push_back(cv::Point2f(1314, 578));  // ↖
    src_pts.push_back(cv::Point2f(3270, 632));  // ↗
    src_pts.push_back(cv::Point2f(1266, 2698)); // ↙
    src_pts.push_back(cv::Point2f(3358, 2566)); // ↘

    std::vector<cv::Point2f> dst_pts;
    dst_pts.push_back(cv::Point2f(0, 0));       // ↖
    dst_pts.push_back(cv::Point2f(2000, 0));    // ↗
    dst_pts.push_back(cv::Point2f(0, 2000));    // ↙
    dst_pts.push_back(cv::Point2f(2000, 2000)); // ↘

    cv::Mat M = getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat result;
    warpPerspective(src, result, M, cv::Size(2000, 2000));

    return result;
}

// 2値化
cv::Mat binaryT(cv::Mat src)
{
    std::vector<cv::Mat> bgr_channels;
    split(src, bgr_channels);
    cv::Mat green_channel = bgr_channels[1];
    cv::Mat result;
    inRange(green_channel, cv::Scalar(THRESHOLD), cv::Scalar(255), result); // 2値化

    return result;
}

// 特定の大きさのオブジェクト以外を削除
cv::Mat remove_area(const cv::Mat image, int min, int max)
{
    cv::Mat hierarchy, result;
    std::vector<std::vector<cv::Point>> contours;
    image.copyTo(result);

    findContours(image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (!(min <= area && area <= max))
        {
            drawContours(result, contours, (int)i, cv::Scalar(0), -1);
        }
    }

    return result;
}

// 端を黒塗りする
cv::Mat crop(const cv::Mat image, int left, int top, int right, int bottom)
{
    cv::Mat result;
    image.copyTo(result);

    cv::rectangle(result, cv::Point(0, 0), cv::Point(left, result.rows), cv::Vec3b(0, 0, 0), cv::FILLED);
    cv::rectangle(result, cv::Point(0, 0), cv::Point(result.cols, top), cv::Vec3b(0, 0, 0), cv::FILLED);
    cv::rectangle(result, cv::Point(result.cols - right, 0), cv::Point(result.cols, result.rows), cv::Vec3b(0, 0, 0), cv::FILLED);
    cv::rectangle(result, cv::Point(0, result.rows - bottom), cv::Point(result.cols, result.rows), cv::Vec3b(0, 0, 0), cv::FILLED);

    return result;
}

// SVG形式に変換
std::string toSVG(const cv::Point *centers, int n)
{
    std::string s = "<?xml version=\"1.0\"?>";
    s += "<svg width=\"210mm\" height=\"297mm\" xmlns=\"http://www.w3.org/2000/svg\">";

    for (size_t i = 1; i < n; i++)
    {
        s += std::format("<circle cx=\"{}mm\" cy=\"{}mm\" r=\"0.5mm\" stroke=\"black\" fill=\"#fff\" stroke-width=\"1\" />", centers[i].x / 1000., centers[i].y / 1000.);
    }

    s += "</svg>";

    return s;
}

// 反射順にソート
std::vector<int> sort(cv::Mat src, int n, cv::Mat labels)
{
    // G値を取り出す
    cv::Mat bgr_channels[3];
    split(src, bgr_channels);

    // インデックスを保持するベクトルを作成
    std::vector<int> indices(n), concentrations(n);
    for (int i = 1; i < n; ++i)
    {
        indices[i] = i;
        concentrations[i] = 0;
    }

    // ラベルごとに濃度測定
    for (int y = 0; y < labels.rows; y++)
    {
        for (int x = 0; x < labels.cols; x++)
        {
            int label = labels.at<int>(y, x);
            uchar concentration = bgr_channels[1].at<uchar>(y, x);
            concentrations[label] += concentration;
            if(label != 0)
                std::print("{}\n", concentration);
        }
    }

#ifdef DEBUG
    std::puts("concentrations");
    for (size_t i = 1; i < n; i++)
    {
        std::print(" {} {}\n", i, concentrations[i]);
    }
#endif // DEBUG

    // 面積を基準にインデックスをソート
    std::sort(indices.begin() + 1, indices.end(), [concentrations](int a, int b)
              { return concentrations[a] > concentrations[b]; });

    return indices;
}

void checkSaturation(cv::Mat src) {
    cv::Mat bgr_channels[3], result;
    split(src, bgr_channels);
    cv:: Mat g_channel = bgr_channels[1];

    inRange(g_channel, cv::Scalar(0), cv::Scalar(254), result);
    imwrite("saturation.png", result);
}