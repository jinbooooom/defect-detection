#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h"


using namespace cv;
using namespace std;

#include "rotateImage.hpp"

IplImage* rotateImage(IplImage* src, double angle, bool clockwise)
{
	IplImage* dst = NULL;
	int width =
		(double)(src->height * sin(angle * CV_PI / 180.0)) +
		(double)(src->width * cos(angle * CV_PI / 180.0)) + 1;
	int height =
		(double)(src->height * cos(angle * CV_PI / 180.0)) +
		(double)(src->width * sin(angle * CV_PI / 180.0)) + 1;
	int tempLength = sqrt((double)src->width * src->width + src->height * src->height) + 10;
	int tempX = (tempLength + 1) / 2 - src->width / 2;
	int tempY = (tempLength + 1) / 2 - src->height / 2;
	int flag = -1;

	dst = cvCreateImage(cvSize(width, height), src->depth, src->nChannels);
	cvZero(dst);
	IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), src->depth, src->nChannels);
	cvZero(temp);

	cvSetImageROI(temp, cvRect(tempX, tempY, src->width, src->height));
	cvCopy(src, temp, NULL);
	cvResetImageROI(temp);

	if (clockwise)
		flag = 1;

	float m[6];
	int w = temp->width;
	int h = temp->height;
	m[0] = (float)cos(flag * angle * CV_PI / 180.);
	m[1] = (float)sin(flag * angle * CV_PI / 180.);
	m[3] = -m[1];
	m[4] = m[0];
	// 将旋转中心移至图像中间  
	m[2] = w * 0.5f;
	m[5] = h * 0.5f;

	CvMat M = cvMat(2, 3, CV_32F, m);
	
	cvGetQuadrangleSubPix(temp, dst, &M);
//	warpAffine(cvarrToMat(temp), cvarrToMat(dst),M,);
	cvReleaseImage(&temp);
	return dst;
}


double getAngle(Mat &img)
{
	Mat img_color, img_cny, _img;
	GaussianBlur(img, _img, Size(15, 15), 0, 0);
	Canny(_img, img_cny, 100, 100, 5);
	vector<vector<Point>> contours;                                         //存放轮廓点的二维点类数组容器
	findContours(img_cny, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	cvtColor(img_cny, img_color, CV_GRAY2BGR);
	
	RotatedRect maxrect;
	double maxlength = 0;
	for (int j = 0; j < contours.size(); j++)
	{
		RotatedRect nrect = minAreaRect(contours[j]);
		if ((nrect.size.height + nrect.size.width)>maxlength)                  //找到周长最大的轮廓即为零件的外轮廓
		{
			maxrect = minAreaRect(contours[j]);
			maxlength = nrect.size.height + nrect.size.width;
		}
	}
	Point2f topoint[4];
	maxrect.points(topoint);                                             //求零件外轮廓的外接旋转矩形以求零件倾斜角度
	//for (int i = 0; i < 4; i++)
		//line(img_color, topoint[i], topoint[(i + 1) % 4], Scalar(0, 255, 255));
	//imshow("color", img_color);
	return maxrect.angle;                                                //返回零件倾斜角度，逆时针为负，顺时针为正
}