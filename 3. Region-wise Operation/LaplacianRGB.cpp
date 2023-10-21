#include <iostream>
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

Mat Laplacianfilter(const Mat input);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat output;


	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);
	output = Laplacianfilter(input); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Laplacian Filter", WINDOW_AUTOSIZE);
	imshow("Laplacian Filter", output);


	waitKey(0);

	return 0;
}


Mat Laplacianfilter(const Mat input) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Laplacian Filter Kernel N
	int tempa = 0;
	int tempb = 0;
	int kernelvalue = 0;
	float out_r = 0.0;
	float out_g = 0.0;
	float out_b = 0.0;

	// Initialiazing Kernel Matrix with 3x3 size
	// Fill code to initialize Sobel filter kernel matrix (Given in the lecture notes)
	kernel = (Mat_<int>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);

	Mat output = Mat::zeros(row, col, input.type());

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			float sum1_r = 0.0;
			float sum1_g = 0.0;
			float sum1_b = 0.0;

			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 

					kernelvalue = kernel.at<int>(a + n, b + n);

					if (i + a > row - 1) {  //mirroring for the border pixels
						tempa = i - a;
					}
					else if (i + a < 0) {
						tempa = -(i + a);
					}
					else {
						tempa = i + a;
					}
					if (j + b > col - 1) {
						tempb = j - b;
					}
					else if (j + b < 0) {
						tempb = -(j + b);
					}
					else {
						tempb = j + b;
					}
					sum1_r += kernelvalue * (float)(input.at<C>(tempa, tempb)[0]);
					sum1_g += kernelvalue * (float)(input.at<C>(tempa, tempb)[1]);
					sum1_b += kernelvalue * (float)(input.at<C>(tempa, tempb)[2]);
				}

			}
			out_r = 8 * abs(sum1_r);
			out_g = 8 * abs(sum1_g);
			out_b = 8 * abs(sum1_b);

			if (out_r > 255)
				out_r = 255;
			else if (out_r < 0)
				out_r = 0;

			if (out_g > 255)
				out_g = 255;
			else if (out_g < 0)
				out_g = 0;

			if (out_b > 255)
				out_b = 255;
			else if (out_b < 0)
				out_b = 0;

			output.at<C>(i, j)[0] = (G)out_r;
			output.at<C>(i, j)[1] = (G)out_g;
			output.at<C>(i, j)[2] = (G)out_b;
		}

	}
	return output;
}