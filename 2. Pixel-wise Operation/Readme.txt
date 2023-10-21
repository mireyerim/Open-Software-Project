1. PDF_CDF.cpp
: color image를 gray scale image로 변환 후, PDF와 CDF를 계산하여 txt file로 내보내는 코드
	header file: hist_func.h
	input: input.jpg
	output: RGB(image), Grayscale(image), PDF.txt, CDF.txt

2. hist_stretching.cpp
: color image를 input으로 받아서 gray scale로 변환 후, input image의 contrast를 높여 image quality를 높이기 위해 histrogram stretching을 실행하는 코드
	header file: hist_func.h
	input: input.jpg
	output: Grayscale (image), Stretched (image), PDF.txt, stretched_PDF.txt, trans_func_stretch.txt

3. hist_eq.cpp
: color image를 input으로 받아서 gray scale image로 변환 후, input image의 contrast를 높여 image quality를 높이기 위해 histogram equalization을 실행하는 코드
	header file: hist_func.h
	input: input.jpg
	output: Grayscale (image), Equalized (image), PDF.txt, equalized_PDF_gray.txt, trans_func_eq.txt

4. hist_eq_RGB.cpp
: color image를 input으로 받아서 input image의 contrast를 높여 image quality를 높이기 위해 R, G, B channel에 각각 histogram equalization을 실행하는 코드
	header file: hist_func.h
	input: input.jpg
	output: RGB (image), Equalized_RGB (image), PDF_RGB.txt, equalized_PDF_RGB.txt, trans_func_eq_RGB.txt

5. hist_eq_YUV.cpp
: color image를 input으로 받아서 input image의 contrast를 높여 image quality를 높이고, color distortion을 줄이기 위해 color image를 YUV scale로 변환 후, Y channel에 histogram equalization을 실행하는 코드
	header file: hist_func.h
	input: input.jpg
	output: PDF_RGB.txt, equalized_PDF_YUV.txt, trans_func_eq_YUV.txt

6. hist_matching_gray.cpp
: performing the histogram matching for a grayscale image
	input: original image
	output: histogram matched image

7. hist_matching_color.cpp
: performing the histogram matching for a rgb scale image
	input: original image
	output: histogram matched image
