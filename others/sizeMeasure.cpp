#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#include "sizeMeasure.hpp"

inline double length(Vec2i p1, Vec2i p2)		//求两点欧式(Euclidean)距离的内联函数
{
	return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}

inline double oLength(Vec2i p1, Vec2i p2)		//求两点cityBlock距离的内联函数
{
	return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]);
}

bool judgement(vector<Vec2i> &vertexSorted, int mode, int &modelNumber, const int edgeError, const double ratioError)
{
	double radius, ratio, edge[4], internalLength[4];
	if (mode != 2)		//非侧面图，都作为普通矩形处理
	{
		edge[0] = length(vertexSorted[0], vertexSorted[1]);
		edge[1] = length(vertexSorted[2], vertexSorted[3]);
		edge[2] = length(vertexSorted[0], vertexSorted[2]);
		edge[3] = length(vertexSorted[1], vertexSorted[3]);

		ratio = edge[2] / edge[0];
		radius = 0;
		if (mode == 3)
		{
			if (fabs(edge[2] / edge[0] - 1.47) < ratioError)
			{
				modelNumber = 0;	//PQ型
			}
			else if (fabs(edge[2] / edge[0] - 2.60) < ratioError)
			{
				modelNumber = 2;	//EC&ED型
			}
		}
	}
	else			//侧视图各个情况下的距离计算
	{
		switch (vertexSorted.size())
		{
		case 24:
			edge[0] = length(vertexSorted[0], vertexSorted[21]);
			edge[1] = length(vertexSorted[2], vertexSorted[23]);
			edge[2] = length(vertexSorted[0], vertexSorted[2]);
			edge[3] = length(vertexSorted[21], vertexSorted[23]);
			internalLength[0] = length(vertexSorted[0], vertexSorted[3]);
			internalLength[1] = length(vertexSorted[18], vertexSorted[21]);
			internalLength[2] = length(vertexSorted[3], vertexSorted[6]);
			internalLength[3] = length(vertexSorted[15], vertexSorted[18]);

			ratio = edge[0] / edge[2];
			radius = length(vertexSorted[6], vertexSorted[15]) / 2;	//半径
			break;
		case 18:
			edge[0] = length(vertexSorted[0], vertexSorted[15]);
			edge[1] = length(vertexSorted[2], vertexSorted[17]);
			edge[2] = length(vertexSorted[0], vertexSorted[2]);
			edge[3] = length(vertexSorted[15], vertexSorted[17]);
			internalLength[0] = length(vertexSorted[0], vertexSorted[3]);
			internalLength[1] = length(vertexSorted[12], vertexSorted[15]);
			internalLength[2] = length(vertexSorted[3], vertexSorted[6]);
			internalLength[3] = length(vertexSorted[9], vertexSorted[12]);

			ratio = edge[0] / edge[2];
			radius = length(vertexSorted[6], vertexSorted[9]) / 2;	//半径
			break;
		case 21:
			edge[0] = length(vertexSorted[0], vertexSorted[18]);
			edge[1] = length(vertexSorted[2], vertexSorted[20]);
			edge[2] = length(vertexSorted[0], vertexSorted[2]);
			edge[3] = length(vertexSorted[18], vertexSorted[20]);
			internalLength[0] = length(vertexSorted[0], vertexSorted[3]);
			internalLength[1] = length(vertexSorted[15], vertexSorted[18]);
			internalLength[2] = length(vertexSorted[3], vertexSorted[6]);
			internalLength[3] = length(vertexSorted[12], vertexSorted[15]);


			ratio = edge[0] / edge[2];
			radius = length(vertexSorted[6], vertexSorted[12]) / 2;	//半径
			break;
		default:
			cerr << "Vertex Error!" << endl;		//零件某部分过长或过短，导致顶点判断错误
			break;
		}

		if (fabs(ratio - 3.15) < ratioError)
		{
			if (fabs(radius / internalLength[0] - 1.12) < ratioError)
			{
				//PQ
				modelNumber = 0;
			}
			else
			{
				//cout << "EE型" << endl;
				modelNumber = 1;
			}
		}
		else if (fabs(ratio - 2.16) < ratioError)
		{
			//EC&ED
			modelNumber = 2;
		}
		else
		{
			cout << "Ratio Error!" << endl;		//零件侧视图的长宽比例出错
			modelNumber = -1;
			return false;
		}
		cout << "圆柱半径: " << radius << endl;
		//cout << radius / internalLength[0] << endl;

		cout << "内部尺寸1相差: " << internalLength[0] - internalLength[1] << endl;
		cout << "内部尺寸2相差: " << internalLength[2] - internalLength[3] << endl;
	}
	
	cout << "对称边长度差：" << edge[0] - edge[1] << ", " << edge[2] - edge[3] << endl;
	cout << "长宽比: " << ratio << endl;

	if (fabs(edge[0] - edge[1]) > edgeError ||		//判断条件
		fabs(edge[2] - edge[3]) > edgeError)
	{
		cout << "产品尺寸不合格" << endl;
		return false;
	}
	else
	{
		cout << "尺寸合格，"
			 << "长宽分别为：" << (edge[0] + edge[1]) / 2 << "， " << (edge[2] + edge[3]) / 2 << endl;
		return true;
	}
}

void vertexRecognition(Mat _srcImage, int direction, vector<Vec2i> &vertexSorted)		//零件的顶点，如果是侧面图则返回排好序的所有顶点，如果是其他面的，只返回4个拐角的顶点坐标
{
	Mat srcImage, binImage;
	/**********************预处理图像（二值化+滤波）*****************************************/
	Mat dst_x, dst_y;
	int canny;
	int MASK, AMASK;
	int houghThre, minLength, maxGap;
	int gaussianMask;
	int virtualCP;
	double threPercent;

	switch (direction)		//不同面的参数设定
	{
	case 1:		//上面		
		MASK = 5; AMASK = 30;
		houghThre = 50; minLength = 50; maxGap = 10;
		gaussianMask = 5;
		threPercent = 0.01;
		canny = 50; virtualCP = 8;  break;
	case 2:		//侧面
		MASK = 5; AMASK = 40;
		houghThre = 30; minLength = 30; maxGap = 5;
		gaussianMask = 9;
		threPercent = 3;
		canny = 100; virtualCP = 8; break;		//100
	case 3:		//正面
		MASK = 5; AMASK = 15;
		houghThre = 40; minLength = 40; maxGap = 15;
		gaussianMask = 9;
		threPercent = 1;
		canny = 130; virtualCP = 8;break;
		/*MASK = 5; AMASK = 15;
		houghThre = 80; minLength = 80; maxGap = 25;
		gaussianMask = 13;
		threPercent = 1;
		canny = 30; virtualCP = 8; break;*/
	default:
		cerr << "Mode Error!" << endl; 
		break;
	}

	//先做边界锐化来提取边界，如果直接二值化会不便于后续hough变换
	GaussianBlur(_srcImage, srcImage, Size(gaussianMask, gaussianMask), 0, 0);
	Canny(srcImage, binImage, canny, canny, 3, true);

	/*****************************边界直线**********************************************/
	vector<Vec4i> lines;
	HoughLinesP(binImage, lines, 1, CV_PI / 180, houghThre, minLength, maxGap);
	Mat srcImage_BGR;								//画边界直线
	cvtColor(binImage, srcImage_BGR, CV_GRAY2BGR);

	for (vector<Vec4i>::size_type i = 0; i < lines.size(); i++)
		line(srcImage_BGR, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 1, CV_AA);
	/******************************************************************************/

	/****************求直线交点****************************/
	double k, k1;
	vector<Vec2i> crossP;
	for (auto i = 0; i < lines.size() - 1; ++i)		//此处计算过程可以优化
	{
		for (auto j = i + 1; j < lines.size(); ++j)
		{
			if (lines[i][1] == lines[i][3])		//直线i水平
			{
				if (lines[j][0] == lines[j][2])	//直线j竖直
				{
					crossP.push_back(Vec2i(lines[j][0], lines[i][1]));
					continue;
				}
				k1 = (double)(lines[j][1] - lines[j][3]) / (lines[j][0] - lines[j][2]);
				if (fabs(k1) < 0.1) continue;  //直线j接近水平

				crossP.push_back(Vec2i((lines[i][1] - lines[j][1]) / k1 + lines[j][0], lines[i][1]));
				continue;
			}

			if (lines[i][0] == lines[i][2])		//直线i竖直
			{
				if (lines[j][1] == lines[j][3])	//直线j水平
				{
					crossP.push_back(Vec2i(lines[i][0], lines[j][1]));
					continue;
				}
				k1 = (double)(lines[j][1] - lines[j][3]) / (lines[j][0] - lines[j][2]);
				if (fabs(k1) > 10) continue;	//直线j接近竖直
				crossP.push_back(Vec2i(lines[i][0], (lines[i][0] - lines[j][0]) * k1 + lines[j][1]));
				continue;
			}
			//如果可以执行到这里，直线i一定是斜线
			if (lines[j][1] == lines[j][3])		//直线j水平
			{
				k = (double)(lines[i][1] - lines[i][3]) / (lines[i][0] - lines[i][2]);
				if (fabs(k) < 0.1) continue;	//直线i也接近水平
				crossP.push_back(Vec2i((lines[j][1] - lines[i][1]) / k + lines[i][0], lines[j][1]));
				continue;
			}
			if (lines[j][0] == lines[j][2])		//直线j竖直
			{
				k = (double)(lines[i][1] - lines[i][3]) / (lines[i][0] - lines[i][2]);
				if (fabs(k) > 10) continue;	//直线i也接近竖直
				crossP.push_back(Vec2i(lines[j][0], (lines[j][0] - lines[i][0]) * k + lines[i][1]));
				continue;
			}
			//执行到这里。i和j都是斜线

			//若两条都是斜线，则代入公式求交点坐标
			k = (double)(lines[i][1] - lines[i][3]) / (lines[i][0] - lines[i][2]);
			k1 = (double)(lines[j][1] - lines[j][3]) / (lines[j][0] - lines[j][2]);
			if (fabs(k - k1) < 1) continue;
			if (fabs(k) > 10 && fabs(k1) > 10) continue;
			//if (fabs(k) < 0.1 && fabs(k1) < 0.1) continue;
			crossP.push_back(Vec2i((lines[i][0] * k - lines[j][0] * k1 + lines[j][1] - lines[i][1]) / (k - k1),
				(lines[i][1] / k - lines[j][1] / k1 + lines[j][0] - lines[i][0]) / (1 / k - 1 / k1)));
		}
	}


	//舍弃部分交点
	for (auto i = 0; i < crossP.size(); ++i)
	{
		if (crossP[i][0] + virtualCP >= binImage.cols || crossP[i][0] - virtualCP <= 0 ||			//交点超出图片范围的舍弃掉
			crossP[i][1] + virtualCP >= binImage.rows || crossP[i][1] - virtualCP <= 0)
		{
			crossP.erase(crossP.begin() + i);
			i--;
			continue;
		}


		bool flag = false;
		for (auto j = -virtualCP; j <= virtualCP; ++j)				//舍弃虚拟交点，即交点附近没有白色像素的交点舍弃
		{
			for (auto k = -virtualCP; k <= virtualCP; ++k)
			{
				if (binImage.at<unsigned char>(crossP[i][1] + j, crossP[i][0] + k))	//Assertion
				{
					flag = true;
					break;
				}
			}
			if (flag) break;
		}
		if (!flag)
		{
			crossP.erase(crossP.begin() + i);
			i--;
			continue;
		}

	}
	//cout << crossP.size() << " points has been found." << endl;
	for (auto i = 0; i < crossP.size(); ++i)		//画交点不能去掉，下面统计要用
		circle(srcImage_BGR, Point(crossP[i][0], crossP[i][1]), 1, Scalar(0, 255, 255), -1);
	/*************************************************************************/


	/******************遍历图像，统计一个mask内交点的数量******************/
	vector<Vec2i> vertex;	//顶点容器
	for (;;)
	{
		Mat counterM(Size(srcImage_BGR.cols / MASK, srcImage_BGR.rows / MASK), CV_8UC1);		//Size(cols(x),rows(y)) Mat.at(rows(y),cols(x))
		for (auto i = 0; i < srcImage_BGR.rows / MASK; ++i)
		{
			for (auto j = 0; j < srcImage_BGR.cols / MASK; ++j)
			{
				unsigned char counter = 0;
				for (auto a = 0; a < MASK; ++a)
				{
					for (auto b = 0; b < MASK; ++b)
					{
						if (srcImage_BGR.at<Vec3b>(i * MASK + a, j * MASK + b) == Vec3b(0, 255, 255))
						{
							counter++;
						}
					}
				}
				counterM.at<unsigned char>(i, j) = counter;
			}
		}

		/*********************求顶点*******************************/


		int max_x = 0, max_y = 0;
		for (;;)		//自动调整识别顶点数目
		{
			for (auto j = 1; j < counterM.rows; ++j)	//找出所有mask内交点最多的mask，并记录
			{
				for (auto k = 1; k < counterM.cols; ++k)
				{
					if (counterM.at<unsigned char>(j, k) > counterM.at<unsigned char>(max_x, max_y))
					{
						max_x = j;
						max_y = k;
					}
				}
			}
			//跳出循环条件
			if (counterM.at<unsigned char>(max_x, max_y) < threPercent)
			{
				break;
			}


			//以上面找出的mask为中心，向四周扩展AMASK大小
			int average_x = 0, average_y = 0, div = 0;
			for (int a = max_x * MASK - AMASK; a < max_x * MASK + AMASK; ++a)
			{
				if (a >= srcImage_BGR.rows || a <= 0) continue;
				for (int b = max_y * MASK - AMASK; b < max_y * MASK + AMASK; ++b)
				{
					if (b >= srcImage_BGR.cols || b <= 0) continue;
					if (srcImage_BGR.at<Vec3b>(a, b) == Vec3b(0, 255, 255))
					{
						average_x += a;
						average_y += b;
						div++;
					}
				}
			}
			if (!div)
			{
				//cerr << "ERROR" << endl;
				continue;
			}
			vertex.push_back(Vec2i(average_x / div, average_y / div));
			for (int a = max_x - AMASK / MASK; a < max_x + AMASK / MASK; ++a)	//已经找过顶点的地方标记为0
			{
				if (a >= counterM.rows || a <= 0) continue;
				for (int b = max_y - AMASK / MASK; b < max_y + AMASK / MASK; ++b)
				{
					if (b >= counterM.cols || b <= 0) continue;
					counterM.at<unsigned char>(a, b) = 0;
				}
			}
		}

		if (vertex.size() < 18 && direction == 2)		//对于侧视图较小的零件，另做处理
		{
			vertex.clear();
			AMASK = 20;
			continue;
		}
		else
		{
			for (unsigned a = 0; a < vertex.size(); ++a)
				circle(srcImage_BGR, Point(vertex[a][1], vertex[a][0]), 3, Scalar(0, 255, 0), -1);
			break;
		}
	}
	/****************************************************************************/

	/***************************标记4个拐角顶点**********************************************/
	vertexSorted.push_back(Vec2i(0, 0)); vertexSorted.push_back(Vec2i(0, 0)); vertexSorted.push_back(Vec2i(0, 0)); vertexSorted.push_back(Vec2i(0, 0));
	for (unsigned k = 1; k < vertex.size(); ++k)	//编号
	{
		if (oLength(vertex[k], Vec2i(0, 0)) < oLength(vertex[vertexSorted[0][0]], Vec2i(0, 0))) vertexSorted[0][0] = k;	//零件左上角 编号 0
		if (oLength(vertex[k], Vec2i(0, binImage.cols)) < oLength(vertex[vertexSorted[1][0]], Vec2i(0, binImage.cols))) vertexSorted[1][0] = k;	//零件右上角 编号 1
		if (oLength(vertex[k], Vec2i(binImage.rows, 0)) < oLength(vertex[vertexSorted[2][0]], Vec2i(binImage.rows, 0))) vertexSorted[2][0] = k;	//零件左下角 编号 2
		if (oLength(vertex[k], Vec2i(binImage.rows, binImage.cols)) < oLength(vertex[vertexSorted[3][0]], Vec2i(binImage.rows, binImage.cols))) vertexSorted[3][0] = k;	//零件右下角 编号 3
	}
	vertexSorted[0] = vertex[vertexSorted[0][0]];	//画四个顶点
	vertexSorted[1] = vertex[vertexSorted[1][0]];
	vertexSorted[2] = vertex[vertexSorted[2][0]];
	vertexSorted[3] = vertex[vertexSorted[3][0]];
	for (unsigned i = 0; i < vertexSorted.size(); ++i)		//画顶点
		circle(srcImage_BGR, Point(vertexSorted[i][1], vertexSorted[i][0]), 2, Scalar(255, 0, 0), -1);
	/************************************************************************************************/


	/**********************侧面，顶点排序**********************************/
	if (direction == 2)
	{
		//cout << vertex.size() << " vertexes found" << endl;
		vertexSorted.clear();
		//顶点排序，开口朝上
		unsigned int temp = vertex.size();
		for (unsigned i = 0; i < temp; ++i)		//根据横坐标顺序排序
		{
			unsigned position = 0;
			for (unsigned j = 1; j < vertex.size(); ++j)
			{
				if (vertex[position][1] > vertex[j][1]) position = j;
			}
			vertexSorted.push_back(vertex[position]);
			vertex.erase(vertex.begin() + position);
		}

		for (unsigned i = 0; i < temp / 3; ++i)
		{
			unsigned position = 1;
			for (unsigned j = 0; j < 3; ++j)
			{
				for (unsigned k = j; k < 3; ++k)
				{
					Vec2i temps;
					if (vertexSorted[i * 3 + k][0] < vertexSorted[i * 3 + j][0])
					{
						temps = vertexSorted[i * 3 + j];
						vertexSorted[i * 3 + j] = vertexSorted[i * 3 + k];
						vertexSorted[i * 3 + k] = temps;
					}

				}
			}
		}
	}
	/*****************************************************************************/

	//标记结果图
	//Mat reSrcImage;
	//resize(srcImage_BGR, reSrcImage, Size(1000, 700));
	//imshow("srcBGR", reSrcImage);
}