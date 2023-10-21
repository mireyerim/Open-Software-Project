// stitching.cpp : Defines the entry point for the console application.
//

#include <opencv2/opencv.hpp>                //include header files
#include <iostream>                          //include header files

using namespace cv;                          //cv 네임스페이스 지정 --> 키워드 앞에 cv::를 붙이지 않아도 됨

template <typename T>                        //이름을 T로 해서 아래 선언된 함수를 템플릿으로 지정
Mat cal_affine(int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[], int number_of_points);         //affine transform을 estimate하는 함수를 선언

void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int diff_x, int diff_y, float alpha);  //two image를 merge하는 함수를 선언

int main() {
	Mat I1, I2;                              //2개의 image에 대한 클래스 생성

	// Read each image
	I1 = imread("stitchingL.jpg");           //file로부터 왼쪽 이미지 l1에 load하기
	I2 = imread("stitchingR.jpg");           //file로부터 오른쪽 이미지 I2에 load하기

	I1.convertTo(I1, CV_32FC3, 1.0 / 255);   //image I1을 변환, 0~255데이터를 255로 나누어 0.0~1.0사이의 데이터들로 변경
	I2.convertTo(I2, CV_32FC3, 1.0 / 255);   //image I2을 변환, 0~255데이터를 255로 나누어 0.0~1.0사이의 데이터들로 변경

	// corresponding pixels	
	int ptl_x[28] = { 509, 558, 605, 649, 680, 689, 705, 730, 734, 768, 795, 802, 818, 837, 877, 889, 894, 902, 917, 924, 930, 948, 964, 969, 980, 988, 994, 998 };  //왼쪽 이미지에서 오른쪽 이미지와 상응하는 점들의 x좌표
	int ptl_y[28] = { 528, 597, 581, 520, 526, 581, 587, 496, 506, 500, 342, 558, 499, 642, 474, 456, 451, 475, 530, 381, 472, 475, 426, 539, 329, 341, 492, 511 };  //왼쪽 이미지에서 오른쪽 이미지와 상응하는 점들의 y좌표
	int ptr_x[28] = { 45, 89, 142, 194, 226, 230, 246, 279, 281, 314, 352, 345, 365, 372, 421, 434, 439, 446, 456, 472, 471, 488, 506, 503, 527, 532, 528, 531 };    //오른쪽 이미지에서 왼쪽 이미지와 상응하는 점들의 x좌표
	int ptr_y[28] = { 488, 561, 544, 482, 490, 546, 552, 462, 471, 467, 313, 526, 468, 607, 445, 429, 424, 447, 500, 358, 446, 449, 403, 510, 312, 324, 466, 484 };  //오른쪽 이미지에서 왼쪽 이미지와 상응하는 점들의 y좌표

	if (!I1.data || !I2.data) {              //Check for invalid input
		std::cout << "Could not open or find the image" << std::endl;   //input image가 invalid이면 다음 message 출력
		return -1;                           //프로그램 종료
	}

	// height(row), width(col) of each image
	const float I1_row = I1.rows;            //image I1의 row을 상수로 저장
	const float I1_col = I1.cols;            //image I1의 col을 상수로 저장
	const float I2_row = I2.rows;            //image I2의 row을 상수로 저장
	const float I2_col = I2.cols;            //image I2의 col을 상수로 저장

	// calculate affine Matrix A12, A21
	Mat A12 = cal_affine<float>(ptl_x, ptl_y, ptr_x, ptr_y, 28);   //inverse warping에서 affine matrix calculate
	Mat A21 = cal_affine<float>(ptr_x, ptr_y, ptl_x, ptl_y, 28);   //forward warping에서 affine matrix calculate 

	//먼저 forward warping을 통해 최종 merged image의 size를 구하고 merged image의 array를 구하기 위함
	// compute corners (p1, p2, p3, p4)
	// p1: (0,0)
	// p2: (row, 0)
	// p3: (row, col)
	// p4: (0, col)
	Point2f p1(A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5));     //I2의 p1을 forwarding warping한 좌표
	Point2f p2(A21.at<float>(0) * 0 + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * I2_row + A21.at<float>(5));    //I2의 p2를 forwarding warping한 좌표
	Point2f p3(A21.at<float>(0) * I2_col + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * I2_row + A21.at<float>(5));    //I2의 p3를 forwarding warping한 좌표
	Point2f p4(A21.at<float>(0) * I2_col + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * 0 + A21.at<float>(5));       //I2의 p4를 forwarding warping한 좌표

	// compute boundary for merged image(I_f)
	// bound_u <= 0
	// bound_b >= I1_row-1
	// bound_l <= 0
	// bound_b >= I1_col-1
	int bound_u = (int)round(min(0.0f, min(p1.y, p4.y)));         //I1과 I2의 upper bound중 최솟값을 구해서 merged image의 upper bound를 구함
	int bound_b = (int)round(max(I1_row - 1, max(p2.y, p3.y)));   //I1과 I2의 bottom bound중 최댓값을 구해서 merged image의 upper bound를 구함
	int bound_l = (int)round(min(0.0f, min(p1.x, p2.x)));         //I1과 I2의 left bound중 최솟값을 구해서 merged image의 left bound를 구함
	int bound_r = (int)round(max(I1_col - 1, max(p3.x, p4.x)));   //I1과 I2의 right bound중 최댓값을 구해서 merged image의 right bound를 구함

	Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_32FC3, Scalar(0));   // initialize merged image

	for (int i = bound_u; i <= bound_b; i++) {                    //inverse warping with bilinear interplolation
		for (int j = bound_l; j <= bound_r; j++) {                //I_f rectangle 범위 내에서
			float x = A12.at<float>(0) * j + A12.at<float>(1) * i + A12.at<float>(2) - bound_l;  //I_f rectangle내의 x좌표에 inverse matrix를 곱한 실수 값
			float y = A12.at<float>(3) * j + A12.at<float>(4) * i + A12.at<float>(5) - bound_u;  //I_f rectangle내의 y좌표에 inverse matrix를 곱한 실수 값

			float y1 = floor(y);      //위에서 구한 실수값에서 내림한 y좌표
			float y2 = ceil(y);       //위에서 구한 실수값에서 올림한 y좌표
			float x1 = floor(x);      //위에서 구한 실수값에서 내림한 x좌표
			float x2 = ceil(x);       //위에서 구한 실수값에서 올림한 x좌표
			 
			float mu = y - y1;        //실수 y의 소수부
			float lambda = x - x1;    //실수 x의 소수부

			if (x1 >= 0 && x2 < I2_col && y1 >= 0 && y2 < I2_row)     //x1, x2, y1, y2값이 I_f rectangle 범위 내일 때
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = lambda * (mu * I2.at<Vec3f>(y2, x2) + (1 - mu) * I2.at<Vec3f>(y1, x2)) +
				(1 - lambda) * (mu * I2.at<Vec3f>(y2, x1) + (1 - mu) * I2.at<Vec3f>(y1, x1));  //bilinear interpolation 방식으로 (i - bound_u, j - bound_l)좌표에 있는 픽셀의 B,G,R데이터 반환
		}
	}

	blend_stitching(I1, I2, I_f, bound_l, bound_u, 0.5);   // image stitching with blend

	namedWindow("Left Image");                             //Left Image 이름의 새로운 윈도우 창 띄우기, autosize
	imshow("Left Image", I1);                              //I1 Image를 'Left Image' window에 표시하기

	namedWindow("Right Image");                            //Right Image 이름의 새로운 윈도우 창 띄우기, autosize
	imshow("Right Image", I2);                             //I2 Image를 'Right Image' window에 표시하기

	namedWindow("result");                                 //result 이름의 새로운 윈도우 창 띄우기, autosize
	imshow("result", I_f);                                 //I_f Image를 'result' window에 표시하기

	I_f.convertTo(I_f, CV_8UC3, 255.0);                    //CV_8U is unsigned 8 bit/pixel, data 값을 0-255값으로 변경하기
	imwrite("result.png", I_f);                            //Image를 result.png 파일에 저장

	waitKey(0);                                            //키 입력이 있을 때까지 기다림

	return 0;                                              //프로그램 종료
}

template <typename T>                        //이름을 T로 해서 아래 선언된 함수를 템플릿으로 지정
Mat cal_affine(int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[], int number_of_points) {     //affine transform을 estimate하는 함수를 선언

	//For pairs of corresponding pixels
	//estimate the affine transform을 위해 unknown값 6개를 구해야 함
	Mat M(2 * number_of_points, 6, CV_32F, Scalar(0));   //affine 되기 전 image: row=2*number_of_points, col=6이고 원소가 0인 array class를 만들기
	Mat b(2 * number_of_points, 1, CV_32F);              //affine된 image: row=2*number_of_points, col=1인 array class 만들기

	Mat M_trans, temp, affineM;             //M_trans: transpose 된 M array를 위한 클래스, temp: transpose된 M과 M을 곱한 array를 저장하기 위한 클래스, affineM: affine matrix를 저장하는 array 클래스

	for (int i = 0; i < number_of_points; i++) {             // initialize matrix
		M.at<T>(2 * i, 0) = ptl_x[i];		M.at<T>(2 * i, 1) = ptl_y[i];		M.at<T>(2 * i, 2) = 1;     //affine matrix를 구하기 위한 Mx=b 식에서 M의 홀수번째 row를 initializing하기 위함
		M.at<T>(2 * i + 1, 3) = ptl_x[i];		M.at<T>(2 * i + 1, 4) = ptl_y[i];		M.at<T>(2 * i + 1, 5) = 1;   //affine matrix를 구하기 위한 Mx=b 식에서 M의 짝수번째 row를 initializing하기 위함
		b.at<T>(2 * i) = ptr_x[i];		b.at<T>(2 * i + 1) = ptr_y[i];   //affine matrix를 구하기 위한 Mx=b 식에서 b를 initialinzing 하기 위함
	}

	// (M^T * M)^(−1) * M^T * b ( * : Matrix multiplication)
	transpose(M, M_trans);          //M의 transpose matrix 구하기
	invert(M_trans * M, temp);      //M의 transpose와 M을 곱하기
	affineM = temp * M_trans * b;   //(M^T * M) * b로 affine matrix 구하기

	return affineM;                 //affine matrix 리턴
}

void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int bound_l, int bound_u, float alpha) {   //Blend two images I1 and I2

	int col = I_f.cols;     //merged image class의 col 받아오기
	int row = I_f.rows;     //merged image class의 row 받아오기

	// I2 is already in I_f by inverse warping
	for (int i = 0; i < I1.rows; i++) {         //I_f의 I1 image row 범위내에서
		for (int j = 0; j < I1.cols; j++) {     //I_f의 I1 image col 범위내에서
			bool cond_I2 = I_f.at<Vec3f>(i - bound_u, j - bound_l) != Vec3f(0, 0, 0) ? true : false;    //I_f의 I1 image 범위 내에서 RGB 값이 이미 정해져있다면 true, 아니면 false

			if (cond_I2)            //I1 and I2 are valid
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = alpha * I1.at<Vec3f>(i, j) + (1 - alpha) * I_f.at<Vec3f>(i - bound_u, j - bound_l);   //I_f의 RGB값을 blending한 값으로 업데이트
			else                    //only I1 is valid
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = I1.at<Vec3f>(i, j);     //I1값으로 RGB값 업데이트

		}
	}
}
