#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IM_TYPE	CV_64FC3

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

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char* opt);
Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char* opt);
Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);
Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale

	// 8-bit unsigned char -> 64-bit floating point
	input.convertTo(input, CV_64FC3, 1.0 / 255);
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);

	// Add noise to original image
	Mat noise_Gray = Add_Gaussian_noise(input_gray, 0, 0.1);
	Mat noise_RGB = Add_Gaussian_noise(input, 0, 0.1);

	// Denoise, using gaussian filter
	Mat Denoised_Gray = Gaussianfilter_Gray(noise_Gray, 3, 10, 10, "zero-padding");
	Mat Denoised_RGB = Gaussianfilter_RGB(noise_RGB, 3, 10, 10, "adjustkernel");

	// Denoise, using gaussian filter
	Mat Denoised_Gray_B = Bilateralfilter_Gray(noise_Gray, 3, 10, 10, 0.1, "zero-padding");
	Mat Denoised_RGB_B = Bilateralfilter_RGB(noise_RGB, 3, 10, 10, 0.1, "adjustkernel");

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Gaussian Noise (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (Grayscale)", noise_Gray);

	namedWindow("Gaussian Noise (RGB)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (RGB)", noise_RGB);

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Denoised (Grayscale)", Denoised_Gray);

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE);
	imshow("Denoised (RGB)", Denoised_RGB);

	namedWindow("Denoised_Bilateral (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Denoised_Bilateral (Grayscale)", Denoised_Gray_B);

	namedWindow("Denoised_Bilateral (RGB)", WINDOW_AUTOSIZE);
	imshow("Denoised_Bilateral (RGB)", Denoised_RGB_B);

	waitKey(0);

	return 0;
}

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {

	Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());
	RNG rng;
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);

	add(input, NoiseArr, NoiseArr);

	return NoiseArr;
}

Mat Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char* opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempx;
	int tempy;
	float denom;
	float kernelvalue;

	// Initialiazing Gaussian Kernel Matrix
	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);


	denom = 0.0;
	for (int x = -n; x <= n; x++) {  // Denominator in m(s,t)
		for (int y = -n; y <= n; y++) {
			float value1 = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
			kernel.at<float>(x + n, y + n) = value1;
			denom += value1;
		}
	}

	//normalized w
	for (int x = -n; x <= n; x++) {  // Denominator in m(s,t)
		for (int y = -n; y <= n; y++) {
			kernel.at<float>(x + n, y + n) /= denom;
		}
	}

	Mat output = Mat::zeros(row, col, input.type());

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1 = 0.0;
			float sum2 = 0.0;
			if (!strcmp(opt, "zero-padding")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "zero-padding" boundary process:

						kernelvalue = kernel.at<float>(x + n, y + n);
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
							sum1 += kernelvalue * (float)(input.at<G>(i + x, j + y));
						}
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "mirroring")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "mirroring" boundary process:
						kernelvalue = kernel.at<float>(x + n, y + n);
						if (i + x > row - 1) {  //mirroring for the border pixels
							tempx = i - x;
						}
						else if (i + x < 0) {
							tempx = -(i + x);
						}
						else {
							tempx = i + x;
						}
						if (j + y > col - 1) {
							tempy = j - y;
						}
						else if (j + y < 0) {
							tempy = -(j + y);
						}
						else {
							tempy = j + y;
						}
						sum1 += kernelvalue * (float)(input.at<G>(tempx, tempy));
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "adjustkernel")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "adjustkernel" boundary process:
						kernelvalue = kernel.at<float>(x + n, y + n);
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							sum1 += kernelvalue * (float)(kernel.at<G>(i + x, j + y));
							sum2 += kernelvalue;
						}
					}
				}
				output.at<G>(i, j) = (G)(sum1 / sum2);
			}

		}
	}

	return output;
}

Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char* opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempx;
	int tempy;
	float denom;
	float kernelvalue;

	// Initialiazing Gaussian Kernel Matrix
	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);


	denom = 0.0;
	for (int x = -n; x <= n; x++) {  // Denominator in m(s,t)
		for (int y = -n; y <= n; y++) {
			float value1 = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
			kernel.at<float>(x + n, y + n) = value1;
			denom += value1;
		}
	}

	//normalized w
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom;
		}
	}

	Mat output = Mat::zeros(row, col, input.type());

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1_r = 0.0;
			float sum1_g = 0.0;
			float sum1_b = 0.0;
			float sum2 = 0.0;
			if (!strcmp(opt, "zero-padding")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "zero-padding" boundary process (3-channel input):
						kernelvalue = kernel.at<float>(x + n, y + n);
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
							sum1_r += kernelvalue * (float)(input.at<C>(i + x, j + y)[0]);
							sum1_g += kernelvalue * (float)(input.at<C>(i + x, j + y)[1]);
							sum1_b += kernelvalue * (float)(input.at<C>(i + x, j + y)[2]);
						}
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;
			}

			else if (!strcmp(opt, "mirroring")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "mirroring" boundary process (3-channel input):
						kernelvalue = kernel.at<float>(x + n, y + n);
						if (i + x > row - 1) {  //mirroring for the border pixels
							tempx = i - x;
						}
						else if (i + x < 0) {
							tempx = -(i + x);
						}
						else {
							tempx = i + x;
						}
						if (j + y > col - 1) {
							tempy = j - y;
						}
						else if (j + y < 0) {
							tempy = -(j + y);
						}
						else {
							tempy = j + y;
						}
						sum1_r += kernelvalue * (float)(input.at<C>(tempx, tempy)[0]);
						sum1_g += kernelvalue * (float)(input.at<C>(tempx, tempy)[1]);
						sum1_b += kernelvalue * (float)(input.at<C>(tempx, tempy)[2]);
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;
			}

			else if (!strcmp(opt, "adjustkernel")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "adjustkernel" boundary process (3-channel input):
						kernelvalue = kernel.at<float>(x + n, y + n);
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
							sum1_r += kernelvalue * (float)(input.at<C>(i + x, j + y)[0]);
							sum1_g += kernelvalue * (float)(input.at<C>(i + x, j + y)[1]);
							sum1_b += kernelvalue * (float)(input.at<C>(i + x, j + y)[2]);
							sum2 += kernelvalue;
						}
					}
				}
				output.at<C>(i, j)[0] = (G)(sum1_r / sum2);
				output.at<C>(i, j)[1] = (G)(sum1_g / sum2);
				output.at<C>(i, j)[2] = (G)(sum1_b / sum2);
			}

		}
	}

	return output;
}

Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt) {
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempx;
	int tempy;
	float denom, kernelvalue, weight1, weight2, sum1, sum2;

	// Initialiazing Gaussian Kernel Matrix
	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	for (int x = -n; x <= n; x++) {
		for (int y = -n; y <= n; y++) {
			kernel.at<float>(x + n, y + n) = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
		}
	}

	Mat output = Mat::zeros(row, col, input.type());

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			sum1 = sum2 = 0.0;

			denom = 0.0;
			weight2 = 0.0;
			if (!strcmp(opt, "zero-padding")) {

				//W(i, j)
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "zero-padding" boundary process:

						weight1 = kernel.at<float>(x + n, y + n);
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							weight2 = exp(-(pow(input.at<G>(i, j) - input.at<G>(i + x, j + y), 2)) / (2 * (pow(sigma_r, 2))));
						}
						else {
							weight2 = exp(-(pow(input.at<G>(i, j), 2)) / (2 * (pow(sigma_r, 2))));
						}
						denom += weight1 * weight2;
					}
				}

				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							weight2 = exp(-(pow(input.at<G>(i, j) - input.at<G>(i + x, j + y), 2)) / (2 * (pow(sigma_r, 2))));
							kernelvalue = weight2 * kernel.at<float>(x + n, y + n) / denom;
							sum1 += kernelvalue * (float)(input.at<G>(i + x, j + y));
						}
					}
				}

				output.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "mirroring")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "mirroring" boundary process:

						weight1 = kernel.at<float>(x + n, y + n);
						if (i + x > row - 1) {  //mirroring for the border pixels
							tempx = i - x;
						}
						else if (i + x < 0) {
							tempx = -(i + x);
						}
						else {
							tempx = i + x;
						}
						if (j + y > col - 1) {
							tempy = j - y;
						}
						else if (j + y < 0) {
							tempy = -(j + y);
						}
						else {
							tempy = j + y;
						}
						weight2 = exp(-(pow(input.at<G>(i, j) - input.at<G>(tempx, tempy), 2)) / (2 * (pow(sigma_r, 2))));
						denom += weight1 * weight2;
					}
				}

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "mirroring" boundary process:

						weight1 = kernel.at<float>(x + n, y + n);
						if (i + x > row - 1) {  //mirroring for the border pixels
							tempx = i - x;
						}
						else if (i + x < 0) {
							tempx = -(i + x);
						}
						else {
							tempx = i + x;
						}
						if (j + y > col - 1) {
							tempy = j - y;
						}
						else if (j + y < 0) {
							tempy = -(j + y);
						}
						else {
							tempy = j + y;
						}
						weight2 = exp(-(pow(input.at<G>(i, j) - input.at<G>(tempx, tempy), 2)) / (2 * (pow(sigma_r, 2))));
						kernelvalue = weight2 * kernel.at<float>(x + n, y + n) / denom;
						sum1 += kernelvalue * (float)(input.at<G>(tempx, tempy));
					}
				}

				output.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "adjustkernel")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "adjustkernel" boundary process:
						weight1 = kernel.at<float>(x + n, y + n);
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
							weight2 = exp(-(pow(input.at<G>(i, j) - input.at<G>(i + x, j + y), 2)) / (2 * (pow(sigma_r, 2))));
							denom += weight1 * weight2;
						}
					}
				}

				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
							weight2 = exp(-(pow(input.at<G>(i, j) - input.at<G>(i + x, j + y), 2)) / (2 * (pow(sigma_r, 2))));
							kernelvalue = weight2 * kernel.at<float>(x + n, y + n) / denom;
							sum1 += kernelvalue * (float)(input.at<G>(i + x, j + y));
							sum2 += kernelvalue;
						}
					}
				}

				output.at<G>(i, j) = (G)(sum1 / sum2);
			}

		}
	}

	return output;
}

Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt) {
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempx, tempy;
	float denom[3], kernelvalue[3], weight1, weight2[3], sum1[3], sum2[3];

	// Initialiazing Gaussian Kernel Matrix
	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	for (int x = -n; x <= n; x++) {
		for (int y = -n; y <= n; y++) {
			kernel.at<float>(x + n, y + n) = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
		}
	}

	Mat output = Mat::zeros(row, col, input.type());

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			sum1[0] = sum1[1] = sum1[2] = 0.0;
			sum2[0] = sum2[1] = sum2[2] = 0.0;

			denom[0] = denom[1] = denom[2] = 0.0;
			weight2[0] = weight2[1] = weight2[2] = 0.0;
			if (!strcmp(opt, "zero-padding")) {

				//W(i, j)
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "zero-padding" boundary process:

						weight1 = kernel.at<float>(x + n, y + n);
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							for (int k = 0; k < 3; k++) {
								weight2[k] = exp(-(pow(input.at<C>(i, j)[k] - input.at<C>(i + x, j + y)[k], 2)) / (2 * (pow(sigma_r, 2))));
							}
						}
						else {
							for (int k = 0; k < 3; k++) {
								weight2[k] = exp(-(pow(input.at<C>(i, j)[k], 2)) / (2 * (pow(sigma_r, 2))));
							}
						}
						denom[0] += weight1 * weight2[0];
						denom[1] += weight1 * weight2[1];
						denom[2] += weight1 * weight2[2];
					}
				}

				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							for (int k = 0; k < 3; k++) {
								weight2[k] = exp(-(pow(input.at<C>(i, j)[k] - input.at<C>(i + x, j + y)[k], 2)) / (2 * (pow(sigma_r, 2))));
								kernelvalue[k] = weight2[k] * kernel.at<float>(x + n, y + n) / denom[k];
								sum1[k] += kernelvalue[k] * (float)(input.at<C>(i + x, j + y)[k]);
							}
						}
					}
				}

				output.at<C>(i, j)[0] = (G)sum1[0];
				output.at<C>(i, j)[1] = (G)sum1[1];
				output.at<C>(i, j)[2] = (G)sum1[2];
			}

			else if (!strcmp(opt, "mirroring")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "mirroring" boundary process:

						weight1 = kernel.at<float>(x + n, y + n);
						if (i + x > row - 1) {  //mirroring for the border pixels
							tempx = i - x;
						}
						else if (i + x < 0) {
							tempx = -(i + x);
						}
						else {
							tempx = i + x;
						}
						if (j + y > col - 1) {
							tempy = j - y;
						}
						else if (j + y < 0) {
							tempy = -(j + y);
						}
						else {
							tempy = j + y;
						}
						for (int k = 0; k < 3; k++) {
							weight2[k] = exp(-(pow(input.at<C>(i, j)[k] - input.at<C>(tempx, tempy)[k], 2)) / (2 * (pow(sigma_r, 2))));
						}
						denom[0] += weight1 * weight2[0];
						denom[1] += weight1 * weight2[1];
						denom[2] += weight1 * weight2[2];
					}
				}

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "mirroring" boundary process:

						weight1 = kernel.at<float>(x + n, y + n);
						if (i + x > row - 1) {  //mirroring for the border pixels
							tempx = i - x;
						}
						else if (i + x < 0) {
							tempx = -(i + x);
						}
						else {
							tempx = i + x;
						}
						if (j + y > col - 1) {
							tempy = j - y;
						}
						else if (j + y < 0) {
							tempy = -(j + y);
						}
						else {
							tempy = j + y;
						}
						for (int k = 0; k < 3; k++) {
							weight2[k] = exp(-(pow(input.at<C>(i, j)[k] - input.at<C>(tempx, tempy)[k], 2)) / (2 * (pow(sigma_r, 2))));
							kernelvalue[k] = weight2[k] * kernel.at<float>(x + n, y + n) / denom[k];
							sum1[k] += kernelvalue[k] * (float)(input.at<C>(tempx, tempy)[k]);
						}
					}
				}

				for (int k = 0; k < 3; k++) {
					output.at<C>(i, j)[k] = (G)sum1[k];
				}
			}

			else if (!strcmp(opt, "adjustkernel")) {

				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						// Gaussian filter with "adjustkernel" boundary process:
						weight1 = kernel.at<float>(x + n, y + n);
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
							for (int k = 0; k < 3; k++) {
								weight2[k] = exp(-(pow(input.at<C>(i, j)[k] - input.at<C>(i + x, j + y)[k], 2)) / (2 * (pow(sigma_r, 2))));
								denom[k] += weight1 * weight2[k];
							}
						}
					}
				}

				for (int x = -n; x <= n; x++) {
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
							for (int k = 0; k < 3; k++) {
								weight2[k] = exp(-(pow(input.at<C>(i, j)[k] - input.at<C>(i + x, j + y)[k], 2)) / (2 * (pow(sigma_r, 2))));
								kernelvalue[k] = weight2[k] * kernel.at<float>(x + n, y + n) / denom[k];
								sum1[k] += kernelvalue[k] * (float)(input.at<C>(i + x, j + y)[k]);
								sum2[k] += kernelvalue[k];
							}
						}
					}
				}

				for (int k = 0; k < 3; k++) {
					output.at<C>(i, j)[k] = (G)(sum1[k] / sum2[k]);
				}
			}

		}
	}

	return output;
}