#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE    CV_8UC3

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

Mat kmeans(Mat input, int clusterCount, int attempts);
Mat kmeansPosition(Mat input, int clusterCount, int attempts, double sigma);


int main() {

    Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);

    Mat input_gray;
    cvtColor(input, input_gray, CV_RGB2GRAY); // Converting image to gray

    if (!input.data)
    {
        std::cout << "Could not open" << std::endl;
        return -1;
    }

    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", input_gray);

    Mat output = kmeans(input_gray, 10, 5);
    Mat output_position = kmeansPosition(input_gray, 10, 5, 7);

    imshow("clustered image", output);
    imshow("clustered image (with position)", output_position);

    waitKey(0);

    return 0;
}

Mat kmeans(Mat input, int clusterCount, int attempts) {
    Mat output = Mat::zeros(input.rows, input.cols, input.type());

    Mat samples(input.rows * input.cols, 1, CV_32F);
    for (int y = 0; y < input.rows; y++)
        for (int x = 0; x < input.cols; x++)
            samples.at<float>(y * input.cols + x, 0) = input.at<G>(y, x) / (float)255;

    Mat labels;
    Mat centers;
    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

    for (int y = 0; y < input.rows; y++)
        for (int x = 0; x < input.cols; x++)
        {
            int cluster_idx = labels.at<int>(y * input.cols + x, 0);
            output.at<G>(y, x) = (G)(int)(centers.at<float>(cluster_idx, 0) * 255);
        }

    return output;
}

Mat kmeansPosition(Mat input, int clusterCount, int attempts, double sigma) {
    Mat output = Mat::zeros(input.rows, input.cols, input.type());

    Mat samples(input.rows * input.cols, 3, CV_32F);
    for (int y = 0; y < input.rows; y++)
        for (int x = 0; x < input.cols; x++) {
            samples.at<float>(y * input.cols + x, 0) = input.at<G>(y, x) / (float)255;
            samples.at<float>(y * input.cols + x, 1) = (y / sigma) / input.rows;
            samples.at<float>(y * input.cols + x, 2) = (x / sigma) / input.cols;
        }

    Mat labels;
    Mat centers;
    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

    for (int y = 0; y < input.rows; y++)
        for (int x = 0; x < input.cols; x++)
        {
            int cluster_idx = labels.at<int>(y * input.cols + x, 0);
            output.at<G>(y, x) = (G)(int)(centers.at<float>(cluster_idx, 0) * 255);
        }

    return output;
}