1. stitching.cpp

purpose of this code: to stitch 2 different image file into one image

input file: stitchingL.jpg, stitchingR.jpg
output file: result.png


functions
- Mat cal_affine(int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[], int number_of_points)   :
	purpose: estimate affine transform
	input:
		int ptl_x[], int ptl_y[], int ptr_x[], int ptr_y[]  :  corresponding pixels
		int number_of_points  :  the number of corresponding pixels
	output:
		affineM  :  affine matrix
- void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int bound_l, int bound_u, float alpha)   :
	purpose: Blend two images
	input:
		const Mat I1, const Mat I2  :  two image array class
		Mat& I_f  :  result image array class
		int bound_l, int bound_u  : upper bound and left bound of merged image
		float alpha  :  blending ratio



2. rotate_skeleton_v2.cpp

purpose of this code: to rotate given image at the given angle

input file: lena.jph
output file: None

functions
- Mat myrotate(const Mat input, float angle, const char* opt)
	purpose: return rotated image array class
	input:
		const Mat input  :  input image array class
		float angle  :  input angle
		const char* opt  :  interpolation option