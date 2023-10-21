#include <iostream>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

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

Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

int main() {

	clock_t start, finish;
	double duration;

	start = clock();

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat output;


	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);
	output = gaussianfilter(input, 1, 1, 1, "zero-paddle"); //Boundary process: zero-paddle, mirroring, adjustkernel

	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("%fÃÊ", duration);

	namedWindow("Gaussian Filter_sep", WINDOW_AUTOSIZE);
	imshow("Gaussian Filter_sep", output);


	waitKey(0);

	return 0;
}


Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {

	Mat kernel1;
	Mat kernel2;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom1;
	float denom2;
	float kernelvalue1;
	float kernelvalue2;

	// Initialiazing Kernel Matrix 
	kernel1 = Mat::zeros(kernel_size, 1, CV_32F);
	kernel2 = Mat::zeros(1, kernel_size, CV_32F);


	denom1 = 0.0;
	for (int a = -n; a <= n; a++) {
		float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))));
		kernel1.at<float>(a + n, 0) = value1;
		denom1 += value1;
	}

	for (int a = -n; a <= n; a++) {
		kernel1.at<float>(a + n, 0) /= denom1;
	}

	denom2 = 0.0;
	for (int b = -n; b <= n; b++) {
		float value2 = exp(-(pow(b, 2) / (2 * pow(sigmaT, 2))));
		kernel2.at<float>(0, b + n) = value2;
		denom2 += value2;
	}

	for (int b = -n; b <= n; b++) {
		kernel2.at<float>(0, b + n) /= denom2;
	}

	Mat output = Mat::zeros(row, col, input.type());
	Mat temp = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1_r = 0.0;
			float sum1_g = 0.0;
			float sum1_b = 0.0;
			float sum2_r = 0.0;
			float sum2_g = 0.0;
			float sum2_b = 0.0;

			if (!strcmp(opt, "zero-paddle")) {

				for (int b = -n; b <= n; b++) {
					kernelvalue2 = kernel2.at<float>(0, b + n);
					if ((j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
						sum2_r += kernelvalue2 * (float)(input.at<C>(i, j + b)[0]);
						sum2_g += kernelvalue2 * (float)(input.at<C>(i, j + b)[1]);
						sum2_b += kernelvalue2 * (float)(input.at<C>(i, j + b)[2]);
					}
				}
				temp.at<C>(i, j)[0] = (G)sum2_r;
				temp.at<C>(i, j)[1] = (G)sum2_g;
				temp.at<C>(i, j)[2] = (G)sum2_b;

				for (int a = -n; a <= n; a++) {
					kernelvalue1 = kernel1.at<float>(a + n, 0);
					if ((i + a <= row - 1) && (i + a >= 0)) { //if the pixel is not a border pixel
						sum1_r += kernelvalue1 * (float)(temp.at<C>(i + a, j)[0]);
						sum1_g += kernelvalue1 * (float)(temp.at<C>(i + a, j)[1]);
						sum1_b += kernelvalue1 * (float)(temp.at<C>(i + a, j)[2]);
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;
			}

			else if (!strcmp(opt, "mirroring")) {

				for (int b = -n; b <= n; b++) {
					kernelvalue2 = kernel2.at<float>(0, b + n);
					if (j + b > col - 1) {
						tempb = j - b;
					}
					else if (j + b < 0) {
						tempb = -(j + b);
					}
					else {
						tempb = j + b;
					}
					sum2_r += kernelvalue2 * (float)(input.at<C>(i, tempb)[0]);
					sum2_g += kernelvalue2 * (float)(input.at<C>(i, tempb)[1]);
					sum2_b += kernelvalue2 * (float)(input.at<C>(i, tempb)[2]);
				}
				temp.at<C>(i, j)[0] = (G)sum2_r;
				temp.at<C>(i, j)[1] = (G)sum2_g;
				temp.at<C>(i, j)[2] = (G)sum2_b;

				for (int a = -n; a <= n; a++) {
					if (i + a > row - 1) {
						tempa = i - a;
					}
					else if (i + a < 0) {
						tempa = -(i + a);
					}
					else {
						tempa = i + a;
					}
					sum1_r += kernelvalue1 * (float)(temp.at<C>(tempa, j)[0]);
					sum1_g += kernelvalue1 * (float)(temp.at<C>(tempa, j)[1]);
					sum1_b += kernelvalue1 * (float)(temp.at<C>(tempa, j)[2]);
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;
			}


			else if (!strcmp(opt, "adjustkernel")) {

				float sum_nol = 0.0;

				for (int b = -n; b <= n; b++) {
					kernelvalue2 = kernel2.at<float>(0, b + n);
					if ((j + b <= col - 1) && (j + b >= 0)) {
						sum2_r += kernelvalue2 * (float)(input.at<C>(i, j + b)[0]);
						sum2_g += kernelvalue2 * (float)(input.at<C>(i, j + b)[1]);
						sum2_b += kernelvalue2 * (float)(input.at<C>(i, j + b)[2]);
						sum_nol += kernelvalue2;
					}
				}
				temp.at<C>(i, j)[0] = (G)(sum2_r / sum_nol);
				temp.at<C>(i, j)[1] = (G)(sum2_g / sum_nol);
				temp.at<C>(i, j)[2] = (G)(sum2_b / sum_nol);

				sum_nol = 0.0;

				for (int a = -n; a <= n; a++) {
					kernelvalue1 = kernel1.at<float>(a + n, 0);
					if ((i + a <= row - 1) && (i + a >= 0)) {
						sum1_r += kernelvalue1 * (float)(temp.at<C>(a + n, j)[0]);
						sum1_g += kernelvalue1 * (float)(temp.at<C>(a + n, j)[1]);
						sum1_b += kernelvalue1 * (float)(temp.at<C>(a + n, j)[2]);
						sum_nol += kernelvalue1;
					}
				}
				output.at<C>(i, j)[0] = (G)(sum1_r / sum_nol);
				output.at<C>(i, j)[1] = (G)(sum1_g / sum_nol);
				output.at<C>(i, j)[2] = (G)(sum1_b / sum_nol);
			}
		}
	}
	return output;
}