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

Mat UnsharpMask(const Mat input, int n, float sigmaT, float sigmaS, const char* boundary_proc, float k);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;


	cvtColor(input, input_gray, CV_RGB2GRAY);


	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	output = UnsharpMask(input_gray, 1, 1, 1, "zero-paddle", 0.5); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Unsharp mask result", WINDOW_AUTOSIZE);
	imshow("Unsharp mask result", output);


	waitKey(0);

	return 0;
}


Mat UnsharpMask(const Mat input, int n, float sigmaT, float sigmaS, const char* boundary_proc, float k) {

	Mat kernel;
	Mat L;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;
	float kernelvalue;

	// Initialiazing Kernel Matrix 
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
	L = Mat::zeros(row, col, input.type());


	denom = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))) - (pow(b, 2) / (2 * pow(sigmaT, 2))));
			kernel.at<float>(a + n, b + n) = value1;
			denom += value1;
		}
	}

	//normalized w
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom;
		}
	}


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {


			if (!strcmp(boundary_proc, "zero-paddle")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						kernelvalue = kernel.at<float>(a + n, b + n);
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum1 += kernelvalue * (float)(input.at<G>(i + a, j + b));
						}
					}
				}
				L.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(boundary_proc, "mirroring")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) {      // for each kernel window
					for (int b = -n; b <= n; b++) {

						kernelvalue = kernel.at<float>(a + n, b + n);
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
						sum1 += kernelvalue * (float)(input.at<G>(tempa, tempb));
					}
				}
				L.at<G>(i, j) = (G)sum1;
			}


			else if (!strcmp(boundary_proc, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) {   // for each kernel window
					for (int b = -n; b <= n; b++) {

						kernelvalue = kernel.at<float>(a + n, b + n);
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1 += kernelvalue * (float)(kernel.at<G>(i + a, j + b));
							sum2 += kernelvalue;
						}
					}
				}
				L.at<G>(i, j) = (G)(sum1 / sum2);
			}
		}
	}

	L = (input - k * L) / (1 - k);

	return L;
		
}