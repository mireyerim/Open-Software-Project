#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

#define RATIO_THR 0.4

using namespace std;
using namespace cv;

template <typename T>
Mat cal_ransac_affine(vector<Point2f> srcPoints, vector<Point2f> dstPoints, int number_of_points, vector<int> k);
void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int diff_x, int diff_y, float alpha);

double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
double nearestNeighborDistance(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
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

	//resize(input1, input1, Size(input1.cols / 2, input1.rows / 2));
	//resize(input2, input2, Size(input2.cols / 2, input2.rows / 2));

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
	Size size1 = input1.size();

	Size sz1 = Size(size1.width + input2_gray.size().width, max(size1.height, input2_gray.size().height));
	Mat matchingImage1 = Mat::zeros(sz1, CV_8UC3);

	input1.copyTo(matchingImage1(Rect(0, 0, size1.width, size1.height)));
	input2.copyTo(matchingImage1(Rect(size1.width, 0, input2_gray.size().width, input2_gray.size().height)));

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
	vector<Point2f> srcPoints1;
	vector<Point2f> dstPoints1;
	bool crossCheck = true;
	bool ratio_threshold = true;
	findPairs(keypoints1, descriptors1, keypoints2, descriptors2, srcPoints1, dstPoints1, crossCheck, ratio_threshold);
	printf("%zd keypoints are matched.\n", srcPoints1.size());


	int n = 3;
	vector<int> k;
	srand((unsigned int)time(NULL));
	for (int i = 0; i < n; i++) {
		k.push_back(rand() % (int)srcPoints1.size());
	}

	input1.convertTo(input1, CV_32FC3, 1.0 / 255);
	input2.convertTo(input2, CV_32FC3, 1.0 / 255);

	const float I1_row = input1.rows;
	const float I2_row = input2.rows;
	const float I1_col = input1.cols;
	const float I2_col = input2.cols;
	int s = 8;
	int maxscore = 0;
	int sigma = 400;
	vector<int> ll;
	for (int l = 0; l < s; l++) {
		int score = 0;
		int u = 0;
		vector<int> li;
		Mat A12 = cal_ransac_affine<float>(srcPoints1, dstPoints1, n, k);

		for (int j = 0; j < srcPoints1.size(); j++) {
			Point2f p12(A12.at<float>(0) * srcPoints1[j].y + A12.at<float>(1) * srcPoints1[j].x + A12.at<float>(2), A12.at<float>(3) * srcPoints1[j].y + A12.at<float>(4) * srcPoints1[j].x + A12.at<float>(5));
			if ((p12.x - dstPoints1[j].x) * (p12.x - dstPoints1[j].x) + (p12.y - dstPoints1[j].y) * (p12.y - dstPoints1[j].y) < sigma * sigma) {
				score = score + 1;
				li.push_back(j);
				u++;
			}
		}
		if (score > maxscore) {
			maxscore = score;
			for (int j = u - 1; j > 0; j--) {
				ll.push_back(li.at(j));
			}
		}
	}

	//ll.size()
	Mat A12 = cal_ransac_affine<float>(srcPoints1, dstPoints1, ll.size(), ll);
	Mat A21 = cal_ransac_affine<float>(dstPoints1, srcPoints1, ll.size(), ll);

	// compute corners (p1, p2, p3, p4)
	Point2f p1(A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5));
	Point2f p2(A21.at<float>(0) * 0 + A21.at<float>(1) * I2_col + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * I2_col + A21.at<float>(5));
	Point2f p3(A21.at<float>(0) * I2_row + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * I2_row + A21.at<float>(4) * 0 + A21.at<float>(5));
	Point2f p4(A21.at<float>(0) * I2_row + A21.at<float>(1) * I2_col + A21.at<float>(2), A21.at<float>(3) * I2_row + A21.at<float>(4) * I2_col + A21.at<float>(5));

	Point2f p1_i(A12.at<float>(0) * 0 + A12.at<float>(1) * 0 + A12.at<float>(2), A12.at<float>(3) * 0 + A12.at<float>(4) * 0 + A12.at<float>(5));
	Point2f p2_i(A12.at<float>(0) * 0 + A12.at<float>(1) * I1_col + A12.at<float>(2), A12.at<float>(3) * 0 + A12.at<float>(4) * I1_col + A12.at<float>(5));
	Point2f p3_i(A12.at<float>(0) * I1_row + A12.at<float>(1) * 0 + A12.at<float>(2), A12.at<float>(3) * I1_row + A12.at<float>(4) * 0 + A12.at<float>(5));
	Point2f p4_i(A12.at<float>(0) * I1_row + A12.at<float>(1) * I1_col + A12.at<float>(2), A12.at<float>(3) * I1_row + A12.at<float>(4) * I1_col + A12.at<float>(5));


	// compute boundary for merged image(I_f)
	int bound_u = (int)round(min(0.0f, min(p1.x, p2.x)));
	int bound_b = (int)round(max(I1_row, max(p3.x, p4.x)));
	int bound_l = (int)round(min(0.0f, min(p1.y, p3.y)));
	int bound_r = (int)round(max(I1_col, max(p2.y, p4.y)));

	int bound_u_ = (int)round(min(0.0f, min(p1_i.x, p2_i.x)));
	int bound_b_ = (int)round(max(I2_row, max(p3_i.x, p4_i.x)));
	int bound_l_ = (int)round(min(0.0f, min(p1_i.y, p3_i.y)));
	int bound_r_ = (int)round(max(I2_col, max(p2_i.y, p4_i.y)));

	int diff_x = abs(bound_u);
	int diff_y = abs(bound_l);

	int diff_xi = abs(bound_u_);
	int diff_yi = abs(bound_l_);

	// initialize merged image
	Mat I_f(bound_b + -bound_u + 1, bound_r - bound_l + 1, CV_32FC3, Scalar(0));
	// inverse warping with bilinear interplolation
	for (int i = -diff_xi; i < I_f.rows - diff_xi; i++) {
		for (int j = -diff_yi; j < I_f.cols - diff_yi; j++) {

			float x = A12.at<float>(0) * i + A12.at<float>(1) * j + A12.at<float>(2) + diff_xi;
			float y = A12.at<float>(3) * i + A12.at<float>(4) * j + A12.at<float>(5) + diff_yi;

			float y1 = floor(y);
			float y2 = floor(y) + 1;
			float x1 = floor(x);
			float x2 = floor(x) + 1;

			float mu = y - y1;
			float lambda = x - x1;

			if (x1 >= 0 && x2 < I2_row && y1 >= 0 && y2 < I2_col)
				I_f.at<Vec3f>(i + diff_xi, j + diff_yi) = (1 - mu) * ((1 - lambda) * input2.at<Vec3f>(x1, y1) + (lambda)*input2.at<Vec3f>(x2, y1)) + (mu) * ((1 - lambda) * input2.at<Vec3f>(x1, y2) + (lambda)*input2.at<Vec3f>(x2, y2));

		}
	}

	// image stitching with blend
	blend_stitching(input1, input2, I_f, diff_x, diff_y, 0.5);
	namedWindow("Left Image");
	imshow("Left Image", input1);

	namedWindow("Right Image");
	imshow("Right Image", input2);


	namedWindow("result");
	imshow("result", I_f);

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
double nearestNeighborDistance(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	int secondneighbor = -1;
	double minDist = 1e6;
	double secondMinDist = 1e6;
	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);
		double distance = euclidDistance(vec, v);
		if (minDist > distance) {
			minDist = distance;
			neighbor = i;
		}
	}
	for (int i = 0; i < descriptors.rows; i++) {
		if (i == neighbor) {
			continue;
		}
		Mat v = descriptors.row(i);
		double distance = euclidDistance(vec, v);
		if (secondMinDist > distance) {
			secondMinDist = distance;
			secondneighbor = i;
		}
	}
	return minDist / secondMinDist;
}

int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	double minDist = 1e6;
	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);
		if (minDist > euclidDistance(vec, v)) {
			minDist = euclidDistance(vec, v);
			neighbor = i;
		}
	}
	return neighbor;
}


void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold) {
	int count = 0;
	for (int i = 0; i < descriptors1.rows; i++) {
		KeyPoint pt1 = keypoints1[i];
		Mat desc1 = descriptors1.row(i);

		int nn1 = nearestNeighbor(desc1, keypoints2, descriptors2);
		double ratio = nearestNeighborDistance(desc1, keypoints2, descriptors2);
		// Refine matching points using ratio_based thresholding
		if (ratio_threshold && ratio >= RATIO_THR) {
			continue;
		}

		// Refine matching points using cross-checking
		if (crossCheck) {
			Mat desc2 = descriptors2.row(nn1);
			int nn2 = nearestNeighbor(desc2, keypoints1, descriptors1);
			if (nn2 != i) {
				continue;
			}

		}
		KeyPoint pt2 = keypoints2[nn1];
		srcPoints.push_back(pt1.pt);
		dstPoints.push_back(pt2.pt);
		count += 1;
	}
}

template <typename T>
Mat cal_ransac_affine(vector<Point2f> srcPoints, vector<Point2f> dstPoints, int number_of_points, vector<int> k) {

	Mat M(2 * number_of_points, 6, CV_32F, Scalar(0));
	Mat b(2 * number_of_points, 1, CV_32F, Scalar(0));

	Mat M_trans, temp, affineM;

	// initialize matrix
	for (int i = 0; i < number_of_points; i++) {

		Point2f pt1 = srcPoints[k.at(i)];
		Point2f pt2 = dstPoints[k.at(i)];
		M.at<float>(2 * i, 0) = pt1.y;
		M.at<float>(2 * i, 1) = pt1.x;
		M.at<float>(2 * i, 2) = 1;
		M.at<float>(2 * i + 1, 3) = pt1.y;
		M.at<float>(2 * i + 1, 4) = pt1.x;
		M.at<float>(2 * i + 1, 5) = 1;

		b.at<float>(2 * i) = pt2.y;
		b.at<float>(2 * i + 1) = pt2.x;

	}

	M_trans = M.t();
	temp = (M_trans * M).inv();
	affineM = temp * M_trans * b;

	return affineM;
}

void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int diff_x, int diff_y, float alpha) {

	int bound_x = I1.rows + diff_x;
	int bound_y = I1.cols + diff_y;

	int col = I_f.cols;
	int row = I_f.rows;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			// for check validation of I1 & I2
			bool cond1 = (i < bound_x && i > diff_x) && (j < bound_y && j > diff_y) ? true : false;
			bool cond2 = I_f.at<Vec3f>(i, j) != Vec3f(0, 0, 0) ? true : false;

			if (cond1 && cond2) {
				I_f.at<Vec3f>(i, j) = alpha * I1.at<Vec3f>(i - diff_x, j - diff_y) + (1 - alpha) * I_f.at<Vec3f>(i, j);
			}

			else if (cond1) {
				I_f.at<Vec3f>(i, j) = I1.at<Vec3f>(i - diff_x, j - diff_y);
			}

		}
	}
}
