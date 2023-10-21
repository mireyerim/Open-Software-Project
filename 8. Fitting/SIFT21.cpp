#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#define RATIO_THR 0.4

using namespace std;
using namespace cv;

template <typename T>                        //이름을 T로 해서 아래 선언된 함수를 템플릿으로 지정
Mat cal_affine(int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[], int number_of_points);         //affine transform을 estimate하는 함수를 선언
void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int diff_x, int diff_y, float alpha);  //two image를 merge하는 함수를 선언

double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
double nearestNeighborT(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);

int main() {

	Mat input1 = imread("input2.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input2 = imread("input1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input1_gray, input2_gray;

	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	resize(input1, input1, Size(input1.cols / 2, input1.rows / 2));
	resize(input2, input2, Size(input2.cols / 2, input2.rows / 2));

	cvtColor(input1, input1_gray, CV_RGB2GRAY);
	cvtColor(input2, input2_gray, CV_RGB2GRAY);

	FeatureDetector* detector = new SiftFeatureDetector(
		0,		// nFeatures
		4,		// nOctaveLayers
		0.04,	// contrastThreshold
		10,		// edgeThreshold
		1.6		// sigma
	);

	DescriptorExtractor* extractor = new SiftDescriptorExtractor();

	// Create a image for displaying mathing keypoints
	Size size = input2.size();

	// Compute keypoints and descriptor from the source image in advance
	vector<KeyPoint> keypoints1;
	Mat descriptors1;

	detector->detect(input1_gray, keypoints1);
	extractor->compute(input1_gray, keypoints1, descriptors1);
	printf("input1 : %d keypoints are found.\n", (int)keypoints1.size());

	vector<KeyPoint> keypoints2;
	Mat descriptors2;

	// Detect keypoints
	detector->detect(input2_gray, keypoints2);
	extractor->compute(input2_gray, keypoints2, descriptors2);

	printf("input2 : %zd keypoints are found.\n", keypoints2.size());

	// Find nearest neighbor pairs
	vector<Point2f> srcPoints;
	vector<Point2f> dstPoints;
	bool crossCheck = true;
	bool ratio_threshold = true;
	findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints, dstPoints, crossCheck, ratio_threshold);
	printf("%zd keypoints are matched.\n", srcPoints.size());


	//stitching & affine transform

	Mat I1, I2;                              //2개의 image에 대한 클래스 생성

	// Read each image
	I1 = imread("input1.jpg");           //file로부터 왼쪽 이미지 l1에 load하기
	I2 = imread("input2.jpg");           //file로부터 오른쪽 이미지 I2에 load하기

	I1.convertTo(I1, CV_32FC3, 1.0 / 255);   //image I1을 변환, 0~255데이터를 255로 나누어 0.0~1.0사이의 데이터들로 변경
	I2.convertTo(I2, CV_32FC3, 1.0 / 255);   //image I2을 변환, 0~255데이터를 255로 나누어 0.0~1.0사이의 데이터들로 변경

	int n = srcPoints.size();

	int* ptl_x = new int[n];
	int* ptl_y = new int[n];
	int* ptr_x = new int[n];
	int* ptr_y = new int[n];

	for (int i = 0; i < n; i++) {
		ptl_x[i] = dstPoints[i].y;
		ptl_y[i] = dstPoints[i].x;
		ptr_x[i] = srcPoints[i].y;
		ptr_y[i] = srcPoints[i].x;
	}

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
	Mat A12 = cal_affine<float>(ptl_x, ptl_y, ptr_x, ptr_y, n);   //inverse warping에서 affine matrix calculate
	Mat A21 = cal_affine<float>(ptr_x, ptr_y, ptl_x, ptl_y, n);   //forward warping에서 affine matrix calculate 

	//먼저 forward warping을 통해 최종 merged image의 size를 구하고 merged image의 array를 구하기 위함
	// compute corners (p1, p2, p3, p4)
	// p1: (0,0)
	// p2: (row, 0)
	// p3: (row, col)
	// p4: (0, col)
	Point2f p1(A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5));     //I2의 p1을 forwarding warping한 좌표
	Point2f p2(A21.at<float>(0) * 0 + A21.at<float>(1) * I2_col + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * I2_col + A21.at<float>(5));    //I2의 p2를 forwarding warping한 좌표
	Point2f p3(A21.at<float>(0) * I2_row + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * I2_row + A21.at<float>(4) * 0 + A21.at<float>(5));    //I2의 p3를 forwarding warping한 좌표
	Point2f p4(A21.at<float>(0) * I2_row + A21.at<float>(1) * I2_col + A21.at<float>(2), A21.at<float>(3) * I2_row + A21.at<float>(4) * I2_col + A21.at<float>(5));       //I2의 p4를 forwarding warping한 좌표

	Point2f p1_i(A12.at<float>(0) * 0 + A12.at<float>(1) * 0 + A12.at<float>(2), A12.at<float>(3) * 0 + A12.at<float>(4) * 0 + A12.at<float>(5));     //I2의 p1을 forwarding warping한 좌표
	Point2f p2_i(A12.at<float>(0) * 0 + A12.at<float>(1) * I1_col + A12.at<float>(2), A12.at<float>(3) * 0 + A12.at<float>(4) * I1_col + A12.at<float>(5));    //I2의 p2를 forwarding warping한 좌표
	Point2f p3_i(A12.at<float>(0) * I1_row + A12.at<float>(1) * 0 + A12.at<float>(2), A12.at<float>(3) * I1_row + A12.at<float>(4) * 0 + A12.at<float>(5));    //I2의 p3를 forwarding warping한 좌표
	Point2f p4_i(A12.at<float>(0) * I1_row + A12.at<float>(1) * I1_col + A12.at<float>(2), A12.at<float>(3) * I1_row + A12.at<float>(4) * I1_col + A12.at<float>(5));       //I2의 p4를 forwarding warping한 좌표

	// compute boundary for merged image(I_f)
	// bound_u <= 0
	// bound_b >= I1_row-1
	// bound_l <= 0
	// bound_b >= I1_col-1
	int bound_u = (int)round(min(0.0f, min(p1.x, p2.x)));
	int bound_b = (int)round(max(I1_row, max(p3.x, p4.x)));
	int bound_l = (int)round(min(0.0f, min(p1.y, p3.y)));
	int bound_r = (int)round(max(I1_col, max(p2.x, p4.x)));

	int bound_ui = (int)round(min(0.0f, min(p1_i.x, p2_i.x)));
	int bound_bi = (int)round(max(I1_row, max(p3_i.x, p4_i.x)));
	int bound_li = (int)round(min(0.0f, min(p1_i.y, p3_i.y)));
	int bound_ri = (int)round(max(I1_col, max(p2_i.x, p4_i.x)));

	int diff_x = abs(bound_u);
	int diff_y = abs(bound_l);

	int diff_xi = abs(bound_ui);
	int diff_yi = abs(bound_li);

	Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_32FC3, Scalar(0));   // initialize merged image

	for (int i = -diff_xi; i < I_f.rows - diff_xi; i++) {                    //inverse warping with bilinear interplolation
		for (int j = -diff_yi; j < I_f.cols - diff_yi; j++) {                //I_f rectangle 범위 내에서
			float x = A12.at<float>(0) * i + A12.at<float>(1) * j + A12.at<float>(2) + diff_xi;  //I_f rectangle내의 x좌표에 inverse matrix를 곱한 실수 값
			float y = A12.at<float>(3) * i + A12.at<float>(4) * j + A12.at<float>(5) + diff_yi;  //I_f rectangle내의 y좌표에 inverse matrix를 곱한 실수 값

			float y1 = floor(y);      //위에서 구한 실수값에서 내림한 y좌표
			float y2 = ceil(y);       //위에서 구한 실수값에서 올림한 y좌표
			float x1 = floor(x);      //위에서 구한 실수값에서 내림한 x좌표
			float x2 = ceil(x);       //위에서 구한 실수값에서 올림한 x좌표

			float mu = y - y1;        //실수 y의 소수부
			float lambda = x - x1;    //실수 x의 소수부

			if (x1 >= 0 && x2 < I2_row && y1 >= 0 && y2 < I2_col)     //x1, x2, y1, y2값이 I_f rectangle 범위 내일 때
				I_f.at<Vec3f>(i + diff_xi, j + diff_yi) = lambda * I2.at<Vec3f>(x2, y) + (1 - lambda) * I2.at<Vec3f>(x1, y);  //bilinear interpolation 방식으로 (i - bound_u, j - bound_l)좌표에 있는 픽셀의 B,G,R데이터 반환
		}
	}

	blend_stitching(I1, I2, I_f, diff_x, diff_y, 0.5);   // image stitching with blend

	namedWindow("Left Image");                             //Left Image 이름의 새로운 윈도우 창 띄우기, autosize
	imshow("Left Image", I1);                              //I1 Image를 'Left Image' window에 표시하기

	namedWindow("Right Image");                            //Right Image 이름의 새로운 윈도우 창 띄우기, autosize
	imshow("Right Image", I2);                             //I2 Image를 'Right Image' window에 표시하기

	namedWindow("result");                                 //result 이름의 새로운 윈도우 창 띄우기, autosize
	imshow("result", I_f);                                 //I_f Image를 'result' window에 표시하기

	waitKey(0);

	return 0;
}

/**
* Calculate euclid distance
*/
double euclidDistance(Mat& vec1, Mat& vec2) {
	double sum = 0.0;
	int dim = vec1.cols;
	for (int i = 0; i < dim; i++) {
		sum += (vec1.at<float>(0, i) - vec2.at<float>(0, i)) * (vec1.at<float>(0, i) - vec2.at<float>(0, i));
	}

	return sqrt(sum);
}

/**
* Find the index of nearest neighbor point from keypoints.
*/
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	double minDist = 1e6;

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		// each row of descriptor

		double dist = euclidDistance(vec, v);
		if (dist < minDist) {
			minDist = dist;
			neighbor = i;
		}

	}

	return neighbor;
}

double nearestNeighborT(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	double first = 1e6;
	double second = 1e6;

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);

		double dist = euclidDistance(vec, v);
		if (dist < first) {
			second = first;
			first = dist;
		}
		else if ((second > dist) && (dist > first)) {
			second = dist;
		}
	}

	double ratio = first / second;
	return ratio;
}


/**
* Find pairs of points with the smallest distace between them
*/
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold) {
	for (int i = 0; i < descriptors1.rows; i++) {
		KeyPoint pt1 = keypoints1[i];
		Mat desc1 = descriptors1.row(i);

		int nn = nearestNeighbor(desc1, keypoints2, descriptors2);

		// Refine matching points using ratio_based thresholding
		if (ratio_threshold) {

			double ratio = nearestNeighborT(desc1, keypoints2, descriptors2);
			if (ratio < RATIO_THR) {}
			else
				continue;
		}

		// Refine matching points using cross-checking
		if (crossCheck) {

			Mat desc2 = descriptors2.row(nn);
			int l = nearestNeighbor(desc2, keypoints1, descriptors1);
			if (i == l) {}
			else
				continue;
		}

		KeyPoint pt2 = keypoints2[nn];
		srcPoints.push_back(pt1.pt);
		dstPoints.push_back(pt2.pt);
	}
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

void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int diff_x, int diff_y, float alpha) {   //Blend two images I1 and I2

	int bound_x = I1.rows + diff_x;
	int bound_y = I1.cols + diff_y;

	int col = I_f.cols;     //merged image class의 col 받아오기
	int row = I_f.rows;     //merged image class의 row 받아오기

	// I2 is already in I_f by inverse warping
	for (int i = 0; i < row; i++) {         //I_f의 I1 image row 범위내에서
		for (int j = 0; j < col; j++) {     //I_f의 I1 image col 범위내에서

			bool cond1 = (i < bound_x && i > diff_x) && (j < bound_y && j > diff_y) ? true : false;  //I_f의 I1 image 범위 내에서 RGB 값이 이미 정해져있다면 true, 아니면 false
			bool cond2 = I_f.at<Vec3f>(i, j) != Vec3f(0, 0, 0) ? true : false;

			if (cond1 && cond2)            //I1 and I2 are valid
				I_f.at<Vec3f>(i, j) = alpha * I1.at<Vec3f>(i - diff_x, j - diff_y) + (1 - alpha) * I_f.at<Vec3f>(i, j);   //I_f의 RGB값을 blending한 값으로 업데이트
			else if (cond1)                 //only I1 is valid
				I_f.at<Vec3f>(i, j) = I1.at<Vec3f>(i - diff_x, j - diff_y);     //I1값으로 RGB값 업데이트

		}
	}
}