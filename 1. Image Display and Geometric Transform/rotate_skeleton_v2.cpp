#include <opencv2/opencv.hpp>          //include header files
#include <iostream>                    //include header files

using namespace cv;                    //cv 네임스페이스 지정 --> 키워드 앞에 cv::를 붙이지 않아도 됨 

template <typename T>                  //이름을 T로 해서 아래 선언된 함수를 템플릿으로 지정
Mat myrotate(const Mat input, float angle, const char* opt);   //myrotate 함수 생성 (input 이미지, 회전각도, interpolation option)

int main()                             //main 함수
{
	Mat input, rotated;                //input, rotated image의 Mat class 생성

	input = imread("lena.jpg");        //Read each image

	if (!input.data) {                 //Check for invalid input
		std::cout << "Could not open or find the image" << std::endl;    //input image가 invalid이면 다음 message 출력
		return -1;                     //프로그램 종료    
	}

	// original image
	namedWindow("image");              //새로운 윈도우창 띄우기, WINDOW_AUTOSIZE가 default
	imshow("image", input);            //읽어온 Mat 데이터를 윈도우 창에 표시하기

	rotated = myrotate<Vec3b>(input, 45, "bilinear");   //rotated image는 3채널, uchar 타입의 벡터 객체, input image를 45도 회전, bilinear 방식으로 interpolation 

	// rotated image
	namedWindow("rotated");            //rotated 이름의 새로운 윈도우 창 띄우기, autosize
	imshow("rotated", rotated);        //rotated image를 'rotated' window에 표시하기

	waitKey(0);                        //키 입력이 있을 때까지 기다림

	return 0;                          //main 함수 종료
}

template <typename T>                  //아래 함수에 대한 템플릿 제공
Mat myrotate(const Mat input, float angle, const char* opt) {      //input class를 angle만큼 회전, interpolation option은 opt
	int row = input.rows;              //input 클래스의 row 수 받아오기
	int col = input.cols;              //input 클래스의 column 수 받아오기

	float radian = angle * CV_PI / 180;    //회전각도를 라디안각으로 바꾸기

	//A rectangle surrounding a rotated image
	float sq_row = ceil(row * sin(radian) + col * cos(radian));    //rectangle의 row 값 구하기
	float sq_col = ceil(col * sin(radian) + row * cos(radian));    //rectangle의 col 값 구하기

	Mat output = Mat::zeros(sq_row, sq_col, input.type());         //'output'으로 rectangle 클래스 생성, row=sq_row, col=sq_col, interpolation 방식은 input image를 따름

	for (int i = 0; i < sq_row; i++) {              //inverse warping
		for (int j = 0; j < sq_col; j++) {          //rectangle 범위 내에서
			float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;   //x는 inverse warping해서 구해진 원래 image의 위치값(실수)
			float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;   //y는 inverse warping해서 구해진 원래 image의 위치값(실수)

			if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) {   //x, y가 input image안의 값이면
				if (!strcmp(opt, "nearest")) {    //interpolation option이 bilinear이면
					int x0 = (int)round(x);       //위에서 구한 실수값을 반올림해서 가장 가까운 x좌표 찾기
					int y0 = (int)round(y);       //위에서 구한 실수값을 반올림해서 가장 가까운 y좌표 찾기 
					
					output.at<Vec3b>(i, j) = input.at<Vec3b>(y0, x0);   //nearest neighbor 방식으로 (i, j)좌표에 있는 픽셀의 B, G, R 데이터 반환

				}
				else if (!strcmp(opt, "bilinear")) {    //interpolation option이 bilinear이면
					float y1 = floor(y);      //위에서 구한 실수값에서 내림한 y좌표
					float y2 = ceil(y);       //위에서 구한 실수값에서 올림한 y좌표
					float x1 = floor(x);      //위에서 구한 실수값에서 내림한 x좌표
					float x2 = ceil(x);       //위에서 구한 실수값에서 올림한 x좌표

					float mu = y - y1;        //실수 y의 소수부
					float lambda = x - x1;    //실수 x의 소수부

					output.at<Vec3b>(i, j) = lambda * (mu * input.at<Vec3b>(y2, x2) + (1 - mu) * input.at<Vec3b>(y1, x2)) +
						(1 - lambda) * (mu * input.at<Vec3b>(y2, x1) + (1 - mu) * input.at<Vec3b>(y1, x1));  //bilinear interpolation 방식으로 (i, j)좌표에 있는 픽셀의 B,G,R데이터 반환

				}
			}
			else            //x, y가 input image 안의 좌표가 아니면
				continue;   //다음좌표로 넘어감
		}
	}

	return output;          //rotated된, output 객체 반환
}