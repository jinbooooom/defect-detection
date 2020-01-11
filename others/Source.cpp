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
	vector<Vec2i> vertexSorted;		//顶点容器
	Mat _srcImage;
	
	int modelNumber;		//零件型号，由下面judgement函数参数带回  0: PQ型  1: EE型  2:EC&ED型	-1: 零件尺寸比例错误，无法判断型号
	const int mode = 2;		//视图方向编号。上面(a+b面):1   侧面(e+d面):2    正面（c+f面):3
	_srcImage = imread("../imgs/侧面/9.bmp", 0);		//读入图片，零件倾角不要大于45度
	bitwise_not(_srcImage, _srcImage);
	double angle = fabs(getAngle(_srcImage));
	bool flag = true;
	if (angle > 45)
	{
		angle = 90 - angle;
		flag = false;
	}
	//cout << "角度：" << angle << endl;
	IplImage  src = IplImage(_srcImage);
	IplImage* dst = rotateImage(&src, angle, flag);
	_srcImage = cvarrToMat(dst);		//旋转放正之后的零件图像

	//cout << _srcImage.cols << "*" << _srcImage.rows << endl;

	vertexRecognition(_srcImage, mode, vertexSorted);	//由vertexSorted带回排好序的顶点坐标
	judgement(vertexSorted, mode, modelNumber);			//modelNumber带回零件型号的编号
	if (mode != 1)
	{
		switch (modelNumber)
		{
		case 0: cout << "PQ型" << endl; break;
		case 1: cout << "EE型" << endl; break;
		case 2: cout << "EC&ED型" << endl; break;
		default:cout << "型号判断错误" << endl; break;
		}
	}
	bitwise_not(_srcImage, _srcImage);		//还到原图，传递给其他函数

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
			if (!vertexSorted[i][0] || !vertexSorted[i][1]) continue;	//坐标为0的点，不转换
			unsigned temp = vertexSorted[i][1];
			vertexSorted[i][1] = vertexSorted[i][0];
			vertexSorted[i][0] = _srcImage.rows - temp;
		}
		
		

		if (modelNumber == 0)
			Fenge3(_srcImage, vertexSorted);
		else if (modelNumber == 2)
			Fenge4(_srcImage, vertexSorted);
		break;
	case 3:		//正面
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

	//for (unsigned i = 0; i < vertexSorted.size(); i++)		//输出顶点坐标
	//	cout << vertexSorted[i][1] << "," << vertexSorted[i][0] << endl;
	getchar();
	waitKey(0);
	return 0;
}
