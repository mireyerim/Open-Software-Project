#include "hist_func.h"

void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF);
void hist_matching_color(Mat& input, Mat& matched, G* trans_func_eq, G* trans_func_ref, G* trans_func_matching);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat ref = imread("ref.jpg", CV_LOAD_IMAGE_COLOR);
	Mat equalized_YUV;
	Mat equalized_YUV_ref;

	float** PDF_RGB = cal_PDF_RGB(input);
	float** PDF_ref = cal_PDF_RGB(ref);

	cvtColor(input, equalized_YUV, CV_RGB2YUV);	// RGB -> YUV
	cvtColor(ref, equalized_YUV_ref, CV_RGB2YUV);	// RGB -> YUV

	// split each channel(Y, U, V)
	Mat channels[3];
	split(equalized_YUV, channels);
	Mat Y = channels[0];
	Mat channels_ref[3];
	split(equalized_YUV_ref, channels_ref);
	Mat Y_ref = channels_ref[0];

	// PDF or transfer function txt files1
	FILE* f_hist_matching_PDF_RGB;
	FILE* f_PDF_RGB;
	FILE* f_PDF_ref_RGB;
	FILE* f_trans_func_matching_YUV;

	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+");
	fopen_s(&f_PDF_ref_RGB, "PDF_ref_RGB.txt", "w+");
	fopen_s(&f_hist_matching_PDF_RGB, "matched_PDF_RGB.txt", "w+");
	fopen_s(&f_trans_func_matching_YUV, "trans_func_matching_YUV.txt", "w+");
			
	float* CDF_YUV = cal_CDF(Y);
	float* CDF_ref = cal_CDF(Y_ref);

	// transfer function
	G trans_func_eq_YUV[L] = { 0 };
	G trans_func_eq_ref[L] = { 0 };
	G trans_func_matching[L] = { 0 };

	hist_eq(Y, Y, trans_func_eq_YUV, CDF_YUV);
	hist_eq(Y_ref, Y_ref, trans_func_eq_ref, CDF_ref);	// histogram equalization on Y_ref channel

	Mat matched_YUV = Y.clone();
	hist_matching_color(Y, Y, trans_func_eq_YUV, trans_func_eq_ref, trans_func_matching);

	// merge Y, U, V channels
	merge(channels, 3, matched_YUV);

	// YUV -> RGB (use "CV_YUV2RGB" flag)
	cvtColor(matched_YUV, matched_YUV, CV_YUV2RGB);

	// matched PDF (YUV)
	float** matched_PDF_RGB = cal_PDF_RGB(matched_YUV);

	for (int i = 0; i < L; i++) {
		// write PDF
		fprintf(f_PDF_RGB, "%d\t%f\t%f\t%f\n", i, PDF_RGB[i][0], PDF_RGB[i][1], PDF_RGB[i][2]);
		fprintf(f_PDF_ref_RGB, "%d\t%f\t%f\t%f\n", i, PDF_ref[i][0], PDF_ref[i][1], PDF_ref[i][2]);
		fprintf(f_hist_matching_PDF_RGB, "%d\t%f\t%f\t%f\n", i, matched_PDF_RGB[i][0], matched_PDF_RGB[i][1], matched_PDF_RGB[i][2]);
		// write transfer functions
		fprintf(f_trans_func_matching_YUV, "%d\t%d\n", i, matched_PDF_RGB[i]);
	}

	// memory release
	free(PDF_RGB);
	free(CDF_YUV);
	free(PDF_ref);
	free(CDF_ref);
	fclose(f_PDF_RGB);
	fclose(f_hist_matching_PDF_RGB);
	fclose(f_trans_func_matching_YUV);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Matched_RGB", WINDOW_AUTOSIZE);
	imshow("Matched_RGB", matched_YUV);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram equalization
void hist_eq(Mat& input, Mat& equalized, G* trans_func, float* CDF) {

	// compute transfer function
	for (int i = 0; i < L; i++)
		trans_func[i] = (G)((L - 1) * CDF[i]);

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}

// histogram matching
void hist_matching_color(Mat& input, Mat& matched, G* trans_func_eq, G* trans_func_ref, G* trans_func_matching) {
	G inverse_trans_func_ref[L] = { 0 };

	//compute inverse function of trans_func_ref
	for (int s = 0; s < L; s++) {
		for (int z = 0; z < L; z++) {
			if (s == trans_func_ref[z]) {
				inverse_trans_func_ref[s] = z;
				break;
			}
			if (z == L - 1)
				inverse_trans_func_ref[s] = inverse_trans_func_ref[s - 1];
		}
	}

	//z = G^(-1)(T(r))
	for (int i = 0; i < L; i++)
		trans_func_matching[i] = inverse_trans_func_ref[trans_func_eq[i]];

	//perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			matched.at<G>(i, j) = trans_func_matching[input.at<G>(i, j)];
}