1. LoG.cpp
purpose: Laplacian of Gaussian을 사용하여 gray scale의 input image를 Edge detection
input: lena.jpg
output: Grayscale (input image), Gaussian blur, Laplacian filter

2. LoG_RGB.cpp
purpose: Laplacian of Gaussian을 사용하여 RGB scale의 input image를 Edge detection
input: lena.jpg
output: RGB scale (input image), Gaussian blur, Laplacian filter

3. CannyEdgeDetector.cpp
purpose: OpenCV의 void cv::Canny function을 사용해서 Edge detection
input: lena.jpg
output: Grayscale (input image), Canny

4. HarrisCornerDetector.cpp
purpose: OpenCV의 void cv::cornerHarris function을 이용해 corner detection을 수행하고, cornerSubPix와 Non-max suppression을 사용하여 더 정밀한 corner를 추출
input: lena.jpg, checkerboard.png, checkerboard2.jpg
output: Harris Response, Harris Corner, Harris Corner (subpixel), Harris Corner (Non-max)
