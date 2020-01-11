#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

#include "sizeMeasure.hpp"
#include "crackDetector.hpp"
#include "rotateImage.hpp"
#include "segmentation.hpp"

int main()
{
	vector<Vec2i> vertexSorted;		//��������
	Mat _srcImage;
	
	int modelNumber;		//����ͺţ�������judgement������������  0: PQ��  1: EE��  2:EC&ED��	-1: ����ߴ���������޷��ж��ͺ�
	const int mode = 2;		//��ͼ�����š�����(a+b��):1   ����(e+d��):2    ���棨c+f��):3
	_srcImage = imread("../imgs/����/9.bmp", 0);		//����ͼƬ�������ǲ�Ҫ����45��
	bitwise_not(_srcImage, _srcImage);
	double angle = fabs(getAngle(_srcImage));
	bool flag = true;
	if (angle > 45)
	{
		angle = 90 - angle;
		flag = false;
	}
	//cout << "�Ƕȣ�" << angle << endl;
	IplImage  src = IplImage(_srcImage);
	IplImage* dst = rotateImage(&src, angle, flag);
	_srcImage = cvarrToMat(dst);		//��ת����֮������ͼ��

	//cout << _srcImage.cols << "*" << _srcImage.rows << endl;

	vertexRecognition(_srcImage, mode, vertexSorted);	//��vertexSorted�����ź���Ķ�������
	judgement(vertexSorted, mode, modelNumber);			//modelNumber��������ͺŵı��
	if (mode != 1)
	{
		switch (modelNumber)
		{
		case 0: cout << "PQ��" << endl; break;
		case 1: cout << "EE��" << endl; break;
		case 2: cout << "EC&ED��" << endl; break;
		default:cout << "�ͺ��жϴ���" << endl; break;
		}
	}
	bitwise_not(_srcImage, _srcImage);		//����ԭͼ�����ݸ���������

	switch (mode)
	{
	case 1:
		Fenge5(_srcImage, Point(vertexSorted[0][1], vertexSorted[0][0]), Point(vertexSorted[3][1], vertexSorted[3][0]));
		break;
	case 2:
		transpose(_srcImage, _srcImage);
		flip(_srcImage, _srcImage, 0);

		if (modelNumber == 0)
		{
			if (vertexSorted.size() == 18)
			{
				vertexSorted.insert(vertexSorted.begin() + 9, 6, Vec2i(0, 0));
			}
			else if (vertexSorted.size() == 21)
			{
				vertexSorted.insert(vertexSorted.begin() + 9, 3, Vec2i(0, 0));
			}
		}
		
		
		for (auto i = 0; i < vertexSorted.size(); ++i)
		{
			if (!vertexSorted[i][0] || !vertexSorted[i][1]) continue;	//����Ϊ0�ĵ㣬��ת��
			unsigned temp = vertexSorted[i][1];
			vertexSorted[i][1] = vertexSorted[i][0];
			vertexSorted[i][0] = _srcImage.rows - temp;
		}
		
		

		if (modelNumber == 0)
			Fenge3(_srcImage, vertexSorted);
		else if (modelNumber == 2)
			Fenge4(_srcImage, vertexSorted);
		break;
	case 3:		//����
		int center_x = (vertexSorted[0][1] + vertexSorted[1][1] + vertexSorted[2][1] + vertexSorted[3][1]) / 4;
		int center_y = (vertexSorted[0][0] + vertexSorted[1][0] + vertexSorted[2][0] + vertexSorted[3][0]) / 4;
		int radius[2] = { 0,0 };
		vector<Vec2i> lineEdge;
		segmentation(_srcImage, modelNumber, center_x, center_y, radius, lineEdge);
		lineEdge.push_back(Vec2i(radius[0], radius[1]));
		/*for (auto i = 0; i < lineEdge.size(); ++i)
		{
			cout << lineEdge[i][1] << "," << lineEdge[i][0] << endl;
		}*/
		
		if (modelNumber == 0)
			Fenge1(_srcImage, Point(center_x, center_y), Point(vertexSorted[0][1], vertexSorted[0][0]), Point(vertexSorted[3][1], vertexSorted[3][0]), lineEdge);
		else if (modelNumber == 2)
			Fenge2(_srcImage, Point(center_x, center_y), Point(vertexSorted[0][1], vertexSorted[0][0]), Point(vertexSorted[3][1], vertexSorted[3][0]), radius);
		
		break;
	}

	//for (unsigned i = 0; i < vertexSorted.size(); i++)		//�����������
	//	cout << vertexSorted[i][1] << "," << vertexSorted[i][0] << endl;
	getchar();
	waitKey(0);
	return 0;
}
