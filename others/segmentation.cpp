#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
using namespace cv;
using namespace std;
#include "segmentation.hpp"

inline double length(Vec2i p1, Vec2i p2);

void segmentation(Mat _srcImage, int modelNumber, int &center_x, int center_y, int *radius, vector<Vec2i> &lineEdge)
{
	Mat binImage;
	bool flag;
	
	vector<Vec2i> circlesEdge;
	flag = false;

	int r1, r2;
	int mid2edge[2];		//0:靠圆心点    1:靠边缘点
	int cannyThre = 100, gaussianSize = 1;
	int edgeError, arcError;
	int linePosition = 45;		//靠圆心的点的位置
	int linePosition_ = 40;
	switch (modelNumber)		//根据型号选择参数
	{
	case 0: r1 = 144; r2 = 290; mid2edge[0] = 40; mid2edge[1] = 50; cannyThre = 120; gaussianSize = 1; edgeError = 9 ; arcError = 4; linePosition = 45; linePosition_ = 40; break;		//9
	case 2: r1 = 94;  r2 = 232; mid2edge[0] = 0; mid2edge[1] = 0; cannyThre = 100; gaussianSize = 3; edgeError = 0; arcError = 4; linePosition = 25; linePosition_ = 60; break;		//13
	}
	GaussianBlur(_srcImage, _srcImage, Size(gaussianSize,gaussianSize), 0, 0);
	Canny(_srcImage, binImage, cannyThre, cannyThre, 3, true);
	

	/******************************搜圆柱半径****************************************/
	for (auto k = -20; k < 20; k++)
	{
		flag = false;
		for (auto i = center_x + r1 - 30; i < center_x + r1 + 30; ++i)
		{
			if (binImage.at<unsigned char>(center_y + k, i) == 255)
			{
				for (auto j = center_x - r1 + 30; j > center_x - r1 - 30; --j)
				{
					if (binImage.at<unsigned char>(center_y + k, j) == 255 &&
						fabs(length(Vec2i(center_y + k, i), Vec2i(center_y, center_x)) - length(Vec2i(center_y + k, j), Vec2i(center_y, center_x))) < 4)
					{
						circlesEdge.push_back(Vec2i(center_y + k, i));
						circlesEdge.push_back(Vec2i(center_y + k, j));
						flag = true;
						break;
					}
				}
			}
			if (flag) break;
		}
	}

	
	center_x = 0;
	for (auto i = 0; i < circlesEdge.size(); i++)
	{
		center_x += circlesEdge[i][1];
	}
	center_x /= circlesEdge.size();
	
	for (auto i = 0; i < circlesEdge.size(); ++i)
	{
		radius[0] += length(circlesEdge[i], Vec2i(center_y, center_x));
	}
	if (radius[0])
		radius[0] /= circlesEdge.size();
	//cout << radius[0] << "," << circlesEdge.size() << endl;



	/****************************搜上下两个圆弧所在的圆的半径******************************************/
	circlesEdge.clear();
	for (auto k = -5; k < 5; ++k)
	{
		flag = false;
		for (auto i = center_y + r2 - 15; i < center_y + r2 + 15; ++i)
		{
			if (binImage.at<unsigned char>(i, center_x + k) == 255)
			{
				for (auto j = center_y - r2 + 15; j > center_y - r2 - 15; --j)
				{
					if (binImage.at<unsigned char>(j, center_x + k) == 255 &&
						fabs(length(Vec2i(i, center_x + k), Vec2i(center_y, center_x)) - length(Vec2i(j, center_x + k), Vec2i(center_y, center_x))) < arcError)
					{
						circlesEdge.push_back(Vec2i(i, center_x + k));
						circlesEdge.push_back(Vec2i(j, center_x + k));
						flag = true;
						break;
					}
				}
			}
			if (flag) break;
		}
	}

	for (auto i = 0; i < circlesEdge.size(); ++i)
	{
		radius[1] += length(circlesEdge[i], Vec2i(center_y, center_x));
	}
	if(radius[1])
		radius[1] /= circlesEdge.size();
	//cout << radius[1] << "," << circlesEdge.size() << endl;

	/*************************搜上下四个边的边界上的点，每个边两个点*****************************/

	
	flag = false;
	for (auto i = center_x - radius[0] - mid2edge[0] - 30; i < center_x - radius[0] - mid2edge[0] + 30; ++i)
	{
		if (binImage.at<unsigned char>(center_y + radius[0] - linePosition, i) == 255)
		{
			for (auto j = center_x + radius[0] + mid2edge[0] + 30; j > center_x + radius[0] + mid2edge[0] - 30; --j)
			{
				if (binImage.at<unsigned char>(center_y + radius[0] - linePosition, j) == 255 &&
					fabs(length(Vec2i(center_y + radius[0] - linePosition, i), Vec2i(center_y, center_x)) - length(Vec2i(center_y + radius[0] - linePosition, j), Vec2i(center_y, center_x))) < 6)
				{
					lineEdge.push_back(Vec2i(center_y + radius[0] - linePosition, i + edgeError));
					lineEdge.push_back(Vec2i(center_y + radius[0] - linePosition, j - edgeError));
					flag = true;
					break;
				}
			}
		}
		if (flag) break;
	}

	flag = false;
	for (auto i = center_x - radius[0] - mid2edge[1] - 30; i < center_x - radius[0] - mid2edge[1] + 30; ++i)
	{
		if (binImage.at<unsigned char>(center_y + radius[0] + linePosition_, i) == 255)
		{
			for (auto j = center_x + radius[0] + mid2edge[1] + 30; j > center_x + radius[0] + mid2edge[1] - 30; --j)
			{
				if (binImage.at<unsigned char>(center_y + radius[0] + linePosition_, j) == 255 &&
					fabs(length(Vec2i(center_y + radius[0] + linePosition_, i), Vec2i(center_y, center_x)) - length(Vec2i(center_y + radius[0] + linePosition_, j), Vec2i(center_y, center_x))) < 6)
				{
					lineEdge.push_back(Vec2i(center_y + radius[0] + linePosition_, i + edgeError));
					lineEdge.push_back(Vec2i(center_y + radius[0] + linePosition_, j - edgeError));
					flag = true;
					break;
				}
			}
		}
		if (flag) break;
	}

	flag = false;
	for (auto i = center_x - radius[0] - mid2edge[0] - 30; i < center_x - radius[0] - mid2edge[0] + 30; ++i)
	{
		if (binImage.at<unsigned char>(center_y - radius[0] + linePosition, i) == 255)
		{
			for (auto j = center_x + radius[0] + mid2edge[0] + 30; j > center_x + radius[0] + mid2edge[0] - 30; --j)
			{
				if (binImage.at<unsigned char>(center_y - radius[0] + linePosition, j) == 255 &&
					fabs(length(Vec2i(center_y - radius[0] + linePosition, i), Vec2i(center_y, center_x)) - length(Vec2i(center_y - radius[0] + linePosition, j), Vec2i(center_y, center_x))) < 6)
				{
					lineEdge.push_back(Vec2i(center_y - radius[0] + linePosition, i + edgeError));
					lineEdge.push_back(Vec2i(center_y - radius[0] + linePosition, j - edgeError));
					flag = true;
					break;
				}
			}
		}
		if (flag) break;
	}

	flag = false;
	for (auto i = center_x - radius[0] - mid2edge[1] - 30; i < center_x - radius[0] - mid2edge[1] + 30; ++i)
	{
		if (binImage.at<unsigned char>(center_y - radius[0] - linePosition_, i) == 255)
		{
			for (auto j = center_x + radius[0] +mid2edge[1] + 30; j > center_x + radius[0] + mid2edge[1] - 30; --j)
			{
				if (binImage.at<unsigned char>(center_y - radius[0] - linePosition_, j) == 255 &&
					fabs(length(Vec2i(center_y - radius[0] - linePosition_, i), Vec2i(center_y, center_x)) - length(Vec2i(center_y - radius[0] - linePosition_, j), Vec2i(center_y, center_x))) < 6)
				{
					lineEdge.push_back(Vec2i(center_y - radius[0] - linePosition_, i + edgeError));
					lineEdge.push_back(Vec2i(center_y - radius[0] - linePosition_, j - edgeError));
					flag = true;
					break;
				}
			}
		}
		if (flag) break;
	}





	//cvtColor(binImage, binImage, CV_GRAY2BGR);
	//for (auto i = 0; i < lineEdge.size(); ++i)
	//{
	//	circle(binImage, Point(lineEdge[i][1], lineEdge[i][0]), 1, Scalar(0, 0, 255), -1);
	//	//cout << lineEdge[i][1] << "," << lineEdge[i][0] << endl;
	//}
	//circle(binImage, Point(center_x, center_y), radius[0], Scalar(0, 0, 255), 1);
	//circle(binImage, Point(center_x, center_y), radius[1], Scalar(0, 0, 255), 1);
	//resize(binImage, binImage, Size(1150, 800));
	//imshow("canny", binImage);*/
}