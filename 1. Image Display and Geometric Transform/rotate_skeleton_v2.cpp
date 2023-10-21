#include <opencv2/opencv.hpp>          //include header files
#include <iostream>                    //include header files

using namespace cv;                    //cv ���ӽ����̽� ���� --> Ű���� �տ� cv::�� ������ �ʾƵ� �� 

template <typename T>                  //�̸��� T�� �ؼ� �Ʒ� ����� �Լ��� ���ø����� ����
Mat myrotate(const Mat input, float angle, const char* opt);   //myrotate �Լ� ���� (input �̹���, ȸ������, interpolation option)

int main()                             //main �Լ�
{
	Mat input, rotated;                //input, rotated image�� Mat class ����

	input = imread("lena.jpg");        //Read each image

	if (!input.data) {                 //Check for invalid input
		std::cout << "Could not open or find the image" << std::endl;    //input image�� invalid�̸� ���� message ���
		return -1;                     //���α׷� ����    
	}

	// original image
	namedWindow("image");              //���ο� ������â ����, WINDOW_AUTOSIZE�� default
	imshow("image", input);            //�о�� Mat �����͸� ������ â�� ǥ���ϱ�

	rotated = myrotate<Vec3b>(input, 45, "bilinear");   //rotated image�� 3ä��, uchar Ÿ���� ���� ��ü, input image�� 45�� ȸ��, bilinear ������� interpolation 

	// rotated image
	namedWindow("rotated");            //rotated �̸��� ���ο� ������ â ����, autosize
	imshow("rotated", rotated);        //rotated image�� 'rotated' window�� ǥ���ϱ�

	waitKey(0);                        //Ű �Է��� ���� ������ ��ٸ�

	return 0;                          //main �Լ� ����
}

template <typename T>                  //�Ʒ� �Լ��� ���� ���ø� ����
Mat myrotate(const Mat input, float angle, const char* opt) {      //input class�� angle��ŭ ȸ��, interpolation option�� opt
	int row = input.rows;              //input Ŭ������ row �� �޾ƿ���
	int col = input.cols;              //input Ŭ������ column �� �޾ƿ���

	float radian = angle * CV_PI / 180;    //ȸ�������� ���Ȱ����� �ٲٱ�

	//A rectangle surrounding a rotated image
	float sq_row = ceil(row * sin(radian) + col * cos(radian));    //rectangle�� row �� ���ϱ�
	float sq_col = ceil(col * sin(radian) + row * cos(radian));    //rectangle�� col �� ���ϱ�

	Mat output = Mat::zeros(sq_row, sq_col, input.type());         //'output'���� rectangle Ŭ���� ����, row=sq_row, col=sq_col, interpolation ����� input image�� ����

	for (int i = 0; i < sq_row; i++) {              //inverse warping
		for (int j = 0; j < sq_col; j++) {          //rectangle ���� ������
			float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;   //x�� inverse warping�ؼ� ������ ���� image�� ��ġ��(�Ǽ�)
			float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;   //y�� inverse warping�ؼ� ������ ���� image�� ��ġ��(�Ǽ�)

			if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) {   //x, y�� input image���� ���̸�
				if (!strcmp(opt, "nearest")) {    //interpolation option�� bilinear�̸�
					int x0 = (int)round(x);       //������ ���� �Ǽ����� �ݿø��ؼ� ���� ����� x��ǥ ã��
					int y0 = (int)round(y);       //������ ���� �Ǽ����� �ݿø��ؼ� ���� ����� y��ǥ ã�� 
					
					output.at<Vec3b>(i, j) = input.at<Vec3b>(y0, x0);   //nearest neighbor ������� (i, j)��ǥ�� �ִ� �ȼ��� B, G, R ������ ��ȯ

				}
				else if (!strcmp(opt, "bilinear")) {    //interpolation option�� bilinear�̸�
					float y1 = floor(y);      //������ ���� �Ǽ������� ������ y��ǥ
					float y2 = ceil(y);       //������ ���� �Ǽ������� �ø��� y��ǥ
					float x1 = floor(x);      //������ ���� �Ǽ������� ������ x��ǥ
					float x2 = ceil(x);       //������ ���� �Ǽ������� �ø��� x��ǥ

					float mu = y - y1;        //�Ǽ� y�� �Ҽ���
					float lambda = x - x1;    //�Ǽ� x�� �Ҽ���

					output.at<Vec3b>(i, j) = lambda * (mu * input.at<Vec3b>(y2, x2) + (1 - mu) * input.at<Vec3b>(y1, x2)) +
						(1 - lambda) * (mu * input.at<Vec3b>(y2, x1) + (1 - mu) * input.at<Vec3b>(y1, x1));  //bilinear interpolation ������� (i, j)��ǥ�� �ִ� �ȼ��� B,G,R������ ��ȯ

				}
			}
			else            //x, y�� input image ���� ��ǥ�� �ƴϸ�
				continue;   //������ǥ�� �Ѿ
		}
	}

	return output;          //rotated��, output ��ü ��ȯ
}