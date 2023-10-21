1. salt_and_pepper.cpp
purpose: salt and pepper noise를 만들고 median filter를 이용해 noise를 제거
input: lena.jpg
output: Grayscale, RGB, Impulse Noise (Grayscale), Impulse Noise (RGB), Denoised (Grayscale), Denoised (RGB)
function:
	- Mat Add_salt_pepper_Noise(const Mat input, float ps, float pp);
		: salt and pepper noise를 더하는 함수
	- Mat Salt_pepper_noise_removal_Gray(const Mat input, int n, const char* opt);
		: mean filter를 이용해 gray scale image의 salt and pepper noise를 제거하는 함수
	- Mat Salt_pepper_noise_removal_RGB(const Mat input, int n, const char* opt);
		: mean filter를 이용해 RGB scale image의 salt and pepper noise를 제거하는 함수

2. Gaussian.cpp
purpose: input image에 gaussian noise를 더하고 gaussian filter와 bilateral filter를 이용해 noise를 제거

input: lena.jpg
output: Grayscale, RGB, Gaussian Noise (Grayscale), Gaussian Noise (RGB), Denoised (Grayscale), Denoised (RGB), Denoised_Bilateral (Grayscale), Denoised_Bilateral (RGB)

function:
	- Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
		: Gaussian noise를 더하는 함수
	- Mat Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char* opt);
		: Gaussian filter를 사용해 gray scale image의 noise를 제거하는 함수
	- Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char* opt);
		: Gaussian filter를 사용해 RGB scale image의 noise를 제거하는 함수
	- Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);
		: Bilateral filter를 사용해 gray scale image의 noise를 제거하는 함수
	- Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);
		: Bilateral filter를 사용해 RGB scale image의 noise를 제거하는 함수