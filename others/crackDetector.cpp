#include "opencv2/core/core.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

#include "crackDetector.hpp"

double length(Point p1, Point p2)		//求两点距离
{
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}
void gamma(Mat &img)                       //对图像进行gamma校正
{
	img.convertTo(img, CV_32FC1);    //转化成32位浮点数
	float *ptr = img.ptr<float>(0);
	for (int i = 0; i < img.rows*img.cols; i++)
		ptr[i] = pow(ptr[i], 0.5);
	normalize(img, img, 0, 255, NORM_MINMAX, CV_8UC1);  //归一化
}
Mat grad(Mat &img)                                        //输入gamma校正后图像，返回其梯度图
{
	normalize(img, img, 0, 255, NORM_MINMAX, CV_32FC1);  //转换成浮点数
	Mat img_grad;
	img.copyTo(img_grad);                            //用以存放梯度幅值

	for (int i = 1; i < img.rows - 1; i++)          //遍历图像除边缘外所有像素求梯度幅值和角度
	{
		float *ptr_grad = img_grad.ptr<float>(i);
		ptr_grad++;
		float* ptr = img.ptr<float>(i);
		for (int j = 1; j < img.cols - 1; j++)
		{
			float Gx = ptr[j + 1] - ptr[j - 1];
			float Gy = ptr[j + img.cols] - ptr[j - img.cols];
			*ptr_grad = sqrt(Gx*Gx + Gy*Gy);
			ptr_grad++;
		}
	}
	normalize(img, img, 0, 255, NORM_MINMAX, CV_8UC1);
	normalize(img_grad, img_grad, 0, 255, NORM_MINMAX, CV_8UC1);
	return img_grad;
}
int maxval(int *ptr, int num)
{
	int max = ptr[0];
	for (int i = 1; i < num; i++)
		if (max < ptr[i])
			max = ptr[i];
	return max;
}
Mat histcount(Mat &img, int *ptri, uchar *ptr, int &count)   //画灰度图的直方图,ptri为存放图像灰度级与其数量的对应关系的数组名，ptr指向图像第一个像素。
{
	int maxheight = 256;
	for (int i = 0; i < img.rows*img.cols; i++)
	{
		if (ptr[i] != 255)
		{
			++ptri[ptr[i]];
			count++;
		}
	}	                                 //统计原图各灰度级像素点个数
	ptri[255] = 0;
	int max = maxval(ptri, 256);
	Mat hist(256, 512, CV_8UC3);                         //直方图宽512，高256
	for (int i = 0; i < 256; i++)
	{
		double height = ptri[i] * maxheight / (1.0*max);
		rectangle(hist, Point(i * 2, 255), Point((i + 1) * 2 - 1, 255 - height), Scalar(0, 0, 255));
	}
	return hist;
}
void histogramequ(Mat &img, Mat img_htg)                       //直方图均衡函数
{
	int count = 0;
	uchar* ptr = img.ptr<uchar>(0);
	uchar* ptr_htg = img_htg.ptr<uchar>(0);                   //用以存放原图各灰度级像素点个数
	int grayori[256] = { 0 };
	double  grayp[256] = { 0 };
	int graynew[256] = { 0 };                                //用以直方图均衡后各灰度级映射关系
	Mat histori/*(256, 512, CV_8UC3)*/;
	histori = histcount(img, grayori, ptr, count);
	//	imshow("hist", histori);                                //画原图像直方图
	for (int i = 0; i < 256; i++)
		grayp[i] = grayori[i] / (1.0*count);               //原图各灰度级像素点个数占总像素个数比率
	double sum = 0;
	int sumi;
	for (int i = 0; i < 256; i++)
	{
		sum += grayp[i];
		sumi = int(255 * sum + 0.5);

		graynew[i] = sumi;                      //构造映射关系
	}
	Mat hist_htg/*(256, 512, CV_8UC3)*/;
	for (int i = 0; i < img.rows*img.cols; i++)        //直方图均衡
	{
		if (ptr_htg[i] != 255)
		{
			ptr_htg[i] = graynew[ptr[i]];
		}
	}
	count = 0;
	memset(graynew, 0, sizeof(int) * 256);               //将存储映射关系的数组清零用以存放均衡后图像的直方图统计数据
	hist_htg = histcount(img_htg, graynew, ptr_htg, count);
	//	imshow("hist_htg", hist_htg);                     //均衡后图像直方图
}
void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	int RemoveCount = 0;       //记录除去的个数  
							   //记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查  
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
//		cout << "Mode: 去除小区域. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
				if (iData[j] < 10)
					iLabel[j] = 3;
		}
	}
	else
	{
//		cout << "Mode: 去除孔洞. ";
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] > 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}

	vector<Point> NeihborPos;  //记录邻域点位置  
	NeihborPos.push_back(Point(-1, 0));
	NeihborPos.push_back(Point(1, 0));
	NeihborPos.push_back(Point(0, -1));
	NeihborPos.push_back(Point(0, 1));
	if (NeihborMode == 1)
	{
//		cout << "Neighbor mode: 8邻域." << endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
//	else cout << "Neighbor mode: 4邻域." << endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//开始检测  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********开始该点处的检查**********  
				vector<Point2i> GrowBuffer;                                      //堆栈，用于存储生长点  
				GrowBuffer.push_back(Point2i(j, i));
				Pointlabel.at<uchar>(i, j) = 1;
				int CheckResult = 0;                                               //用于判断结果（是否超出大小），0为未超出，1为超出  

				for (int z = 0; z<GrowBuffer.size(); z++)
				{

					for (int q = 0; q<NeihborCount; q++)                                      //检查四个邻域点  
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX<Src.cols&&CurrY >= 0 && CurrY<Src.rows)  //防止越界  
						{
							if (Pointlabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //邻域点加入buffer  
								Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //更新邻域点的检查标签，避免重复检查  
							}
						}
					}

				}
				if (GrowBuffer.size()>AreaLimit) CheckResult = 2;                 //判断结果（是否超出限定的大小），1为未超出，2为超出  
				else { CheckResult = 1;   RemoveCount++; }
				for (int z = 0; z<GrowBuffer.size(); z++)                         //更新Label记录  
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
				}
				//********结束该点处的检查**********  


			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//开始反转面积过小的区域  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iData = Src.ptr<uchar>(i);
		uchar* iDstData = Dst.ptr<uchar>(i);
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 2)
			{
				iDstData[j] = CheckMode;
			}
			else if (iLabel[j] == 3)
			{
				iDstData[j] = iData[j];
			}
		}
	}

//	cout << RemoveCount << " objects removed." << endl;
}

void  StainTest(Mat &imgOri, Mat &imgSeg, const int ComponentType)            //第一个参数为零件BGR图，第二个参数为零件分区grey图，第三个参数为零件型号
{
	int adaThreshold;
	if (ComponentType == 1)
		adaThreshold = 5;
	if (ComponentType == 2)
		adaThreshold = 6;
	Mat img_copy, img_color;
	imgSeg.copyTo(img_copy);                     //拷贝分区图用来处理
	img_copy = grad(img_copy);
	img_copy = grad(img_copy);
	//	imshow("gg", img_copy);
	adaptiveThreshold(img_copy, img_copy, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 21, adaThreshold);   //自适应二值化，找到油污

	Mat element = getStructuringElement(MORPH_RECT, Size(2, 2));             //开操作预处理
	morphologyEx(img_copy, img_copy, MORPH_OPEN, element);
	//	imshow("bin", img_copy);                                                 

	Mat imgStain(img_copy.rows, img_copy.cols, img_copy.type(), Scalar(0, 0, 0));       //创建油污图，将分区图上找到的可能油污点映射到此图上
	vector<Point> stain;
	vector<vector<Point>> contours;                                         //存放轮廓点的二维点类数组容器
	findContours(img_copy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);   //定位找到的可能油污点画轮廓确定坐标位置以便映射到imgStain上
																			//	cvtColor(imgOri, img_color, CV_GRAY2BGR);                            //转为BGR以便画出轮廓，
	int stainCount = 0;                                                 //统计在imgSeg图上初步符合要求的可能油污点个数
	for (int j = 0; j < contours.size(); j++)
	{
		RotatedRect nrect = minAreaRect(contours[j]);
		if ((nrect.size.width < 8) && (nrect.size.height < 8))              //筛选油污点，稍微大的轮廓剔除
		{
			Point2f topoint[4];
			nrect.points(topoint);
			stain.push_back(nrect.center);
			stainCount++;
			//for (int i = 0; i < 4; i++)
			//	line(img_color, topoint[i], topoint[(i + 1) % 4], Scalar(0, 0, 255));
		}
	}
	//	imshow("con", img_color);
	for (int i = 0; i < stainCount; i++)
		circle(imgStain, stain[i], 6, Scalar(255, 255, 255), -1, 8, 0);                    //映射油污点
																						   //	imshow("circle",imgStain);

	vector<vector<Point>> staincontours;                                         //存放轮廓点的二维点类数组容器
	findContours(imgStain, staincontours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);   //找点密集的区域即为油污所在
	for (int j = 0; j < staincontours.size(); j++)
	{
		RotatedRect stainrect = minAreaRect(staincontours[j]);
		if ((stainrect.size.width >30) && (stainrect.size.height > 30))           //筛选符合条件的油污块
		{
			Point2f topoint[4];
			stainrect.points(topoint);
			for (int i = 0; i < 4; i++)
				line(imgOri, topoint[i], topoint[(i + 1) % 4], Scalar(0, 255, 0));     //在零件原始图上用绿线标示油污所在地方

			Point IntPoint[1][4];                                                           //用白色像素填充油污处，防止被缺陷检测误检
			IntPoint[0][0].x = topoint[0].x; IntPoint[0][0].y = topoint[0].y;
			IntPoint[0][1].x = topoint[1].x; IntPoint[0][1].y = topoint[1].y;
			IntPoint[0][2].x = topoint[2].x; IntPoint[0][2].y = topoint[2].y;
			IntPoint[0][3].x = topoint[3].x; IntPoint[0][3].y = topoint[3].y;
			const Point* ppt[1] = { IntPoint[0] };
			int npt[] = { 4 };
			polylines(imgSeg, ppt, npt, 1, 1, Scalar(255, 255, 255), 1, 8, 0);
			fillPoly(imgSeg, ppt, npt, 1, Scalar(255, 255, 255));
		}
	}
//	imshow("staincontours", imgOri);
}
void  FlawTest(Mat &imgOri, Mat &imgSeg, const int ComponentType, int ComponentSeg)//第一个参数为零件BGR图，第二个参数为零件分区grey图，第三个参数为零件型号,第四个参数为零件分区图编号
{
	int Gaussksize, canny1, canny2, canny3, maxgrey, mingrey, areaThreshold, lenThreshold, ratioThreshold;
	if (ComponentType == 1)                                                                        //   不同型号，不同分区参数选择
	{
		if ((ComponentSeg == 0) || (ComponentSeg == 4))
		{
			Gaussksize = 5;
			canny1 = 200;
			canny2 = 70;
			canny3 = 140;
			maxgrey = 225;
			mingrey = 15;
			areaThreshold = 32;
			lenThreshold = 6;
			ratioThreshold = 2;
		}
		if (ComponentSeg == 2)
		{
			Gaussksize = 7;
			canny1 = 200;
			canny2 = 70;
			canny3 = 140;
			maxgrey = 225;
			mingrey = 5;
			areaThreshold = 32;
			lenThreshold = 8;
			ratioThreshold = 2;
		}
		if ((ComponentSeg == 1) || (ComponentSeg == 3))
		{
			Gaussksize = 5;
			canny1 = 150;
			canny2 = 20;
			canny3 = 70;
			maxgrey = 255;
			mingrey = 50;
			areaThreshold = 210;
			lenThreshold = 12;
			ratioThreshold = 3.0;
		}
	}
	if (ComponentType == 2)                                                                        //   不同型号，不同分区参数选择
	{
		if ((ComponentSeg == 0) || (ComponentSeg == 4))
		{
			Gaussksize = 5;
			canny1 = 200;
			canny2 = 70;
			canny3 = 140;
			maxgrey = 225;
			mingrey = 15;
			areaThreshold = 50;
			lenThreshold = 5;
			ratioThreshold = 4;
		}
		if (ComponentSeg == 2)
		{
			Gaussksize = 5;
			canny1 = 200;
			canny2 = 70;
			canny3 = 140;
			maxgrey = 240;
			mingrey = 15;
			areaThreshold = 40;
			lenThreshold = 6;
			ratioThreshold = 4;
		}
		if ((ComponentSeg == 1) || (ComponentSeg == 3))
		{
			Gaussksize = 5;
			canny1 = 200;
			canny2 = 50;
			canny3 = 100;
			maxgrey = 255;
			mingrey = 32;
			areaThreshold = 100;
			lenThreshold = 14;
			ratioThreshold = 1.8;
		}
	}
	if (ComponentType == 3)
	{
		Gaussksize = 7;
		canny1 = 150;
		canny2 = 70;
		canny3 = 80;
		maxgrey = 225;
		mingrey = 15;
		areaThreshold = 80;
		lenThreshold = 14;
		ratioThreshold = 2;
	}

//	imshow("ori", imgSeg);
	Mat img_copy, img_htg, img_canny, img_color;
	imgSeg.copyTo(img_copy);                                                                   //求零件的外接正矩形，以便后续处理
	adaptiveThreshold(img_copy, img_copy, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 1);
//	Canny(img_copy,img_copy,5,5,5,true);
	vector<vector<Point>> contoursc;
	findContours(img_copy, contoursc, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	int maxi = 0;
	for (int i = 1; i < contoursc.size(); i++)
		if (contoursc[i].size()>contoursc[maxi].size())
			maxi = i;
	Rect roiRect = boundingRect(contoursc[maxi]);

	Mat img_canny1;
	imgSeg.copyTo(img_htg);
	imgSeg.copyTo(img_canny);
	histogramequ(img_htg, img_htg);                                     //直方图均衡，增强对比度，更好区分缺陷
	//imshow("htgequ", img_htg);
	GaussianBlur(img_canny, img_canny, Size(Gaussksize, Gaussksize), 0, 0);      //高斯滤波，去除小噪声
	Canny(img_canny, img_canny1, canny1, canny1, 3);                                   //用阈值很大的Canny算子提取零件的最外层轮廓边缘
	//imshow("canny1", img_canny1);
	Canny(img_canny, img_canny, canny2, canny3, 3);
	//imshow("canny0", img_canny);
	img_canny = img_canny - img_canny1;
//	imshow("canny", img_canny);

	Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));
	dilate(img_canny, img_canny, element);
//	imshow("dilate", img_canny);
	uchar *ptr_htg = img_htg.ptr<uchar>(0);
	uchar *ptr_cny = img_canny.ptr<uchar>(0);
	for (int i = 0; i < imgSeg.rows*imgSeg.cols; i++)
	{
		if (ptr_htg[i] != 255)
			if (ptr_cny[i] == 255)
				if ((ptr_htg[i]>maxgrey) || (ptr_htg[i] < mingrey))
					ptr_htg[i] = 255;
				else
					ptr_htg[i] = 0;
			else
				ptr_htg[i] = 0;
		else
			ptr_htg[i] = 0;

	}
//	imshow("flaw", img_htg);
	if (ComponentType == 2)
		RemoveSmallRegion(img_htg, img_htg, 6, 1, 1);

	Mat img_htgroi = img_htg(roiRect);

	if ((ComponentType == 1) && (ComponentSeg == 0))                                                         //零件最上部分左下角和右下角涂黑
	{
		int iStart = int(0.72*img_htgroi.rows);
		for (int i = iStart; i < img_htgroi.rows; i++)
		{
			uchar *ptr_roi = img_htgroi.ptr<uchar>(i);
			int iDelta = (i - iStart);
			for (int j = img_htgroi.cols - iDelta; j < img_htgroi.cols; j++)
				ptr_roi[j] = 0;
		}
		for (int i = iStart; i < img_htgroi.rows; i++)
		{
			uchar *ptr_roi = img_htgroi.ptr<uchar>(i);
			int iDelta = (i - iStart);
			for (int j = 0; j <iDelta; j++)
				ptr_roi[j] = 0;
		}
		iStart = int(0.88*img_htgroi.rows);
		for (int i = iStart; i < img_htgroi.rows; i++)
		{
			uchar *ptr_roi = img_htgroi.ptr<uchar>(i);
			for (int j = 0; j < img_htgroi.cols; j++)
				ptr_roi[j] = 0;
		}
	}
	if ((ComponentType == 1) && (ComponentSeg == 4))                                                                   //零件最下部分左上角和右上角涂黑
	{
		int iEnd = int(0.20*img_htgroi.rows);
		for (int i = 0; i < iEnd; i++)
		{
			uchar *ptr_roi = img_htgroi.ptr<uchar>(i);
			int iDelta = iEnd - i;
			for (int j = img_htgroi.cols - iDelta; j < img_htgroi.cols; j++)
				ptr_roi[j] = 0;
		}
		for (int i = 0; i < iEnd; i++)
		{
			uchar *ptr_roi = img_htgroi.ptr<uchar>(i);
			int iDelta = iEnd - i;
			for (int j = 0; j <iDelta; j++)
				ptr_roi[j] = 0;
		}
		iEnd = int(0.25*img_htgroi.rows);                //0.21
		for (int i = 0; i < iEnd; i++)
		{
			uchar *ptr_roi = img_htgroi.ptr<uchar>(i);
			for (int j = 0; j < img_htgroi.cols; j++)
				ptr_roi[j] = 0;
		}
	}
	element = getStructuringElement(MORPH_RECT, Size(4, 4));
	dilate(img_htgroi, img_htgroi, element);
//	imshow("remove", img_htgroi);

	vector<vector<Point>> contours;                                         //存放轮廓点的二维点类数组容器
	findContours(img_htgroi, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);   //
//	cvtColor(imgOri, img_color, CV_GRAY2BGR);                               //转为BGR以便画出轮廓
	for (int j = 0; j < contours.size(); j++)
	{
		double area = contourArea(contours[j]);
		if (area>areaThreshold)
		{
			RotatedRect nrect = minAreaRect(contours[j]);
			int shortLen = nrect.size.height > nrect.size.width ? nrect.size.width : nrect.size.height;
			int longLen = nrect.size.width + nrect.size.height - shortLen;
			float ratio = longLen / shortLen;
			if (shortLen >= lenThreshold)
			{
				Point2f topoint[4];
				nrect.points(topoint);
				for (int i = 0; i < 4; i++)                                                 //计算有效区域对应原图中的坐标，以便在原图上标示
				{
					topoint[i].x += roiRect.x;
					topoint[i].y += roiRect.y;
				}
				for (int i = 0; i < 4; i++)
					line(imgOri, topoint[i], topoint[(i + 1) % 4], Scalar(0, 0, 255));
			}
			else
			{
				if (ratio<ratioThreshold)
				{
					Point2f topoint[4];
					nrect.points(topoint);
					for (int i = 0; i < 4; i++)
					{
						topoint[i].x += roiRect.x;
						topoint[i].y += roiRect.y;
					}
					for (int i = 0; i < 4; i++)
						line(imgOri, topoint[i], topoint[(i + 1) % 4], Scalar(0, 0, 255));
				}
			}
		}

	}
//	namedWindow("con", CV_WINDOW_NORMAL);
//	imshow("con", imgOri);
}
void SiltTest(Mat &imgOri, Mat &imgSeg)
{

	Mat dst, sub, img_color, img_copy;
	imgSeg.copyTo(dst);
	imgSeg.copyTo(img_copy);
	gamma(img_copy);
	img_copy = grad(img_copy);
	threshold(img_copy, img_copy, 20, 255, THRESH_BINARY);
//	imshow("bin0", img_copy);
	Mat element = getStructuringElement(MORPH_RECT, Size(2, 2));
	morphologyEx(img_copy, img_copy, MORPH_CLOSE, element);
	//	imshow("bin",img_copy);
	RemoveSmallRegion(img_copy, img_copy, 15, 1, 1);
	dilate(img_copy, img_copy, element);
	//	imshow("dilate1", img_copy);

	Canny(dst, sub, 230, 230, 3);
//	imshow("canny1", sub);
	Canny(dst, dst, 40, 120, 3);
//	imshow("canny", dst);
	sub = dst - sub;
	dilate(sub, sub, element);
//	imshow("dilate2",sub);


	uchar *ptr = img_copy.ptr<uchar>(0);
	uchar *ptrp = sub.ptr<uchar>(0);
	for (int i = 0; i < img_copy.rows*img_copy.cols; i++)
	{
		if ((*ptr == 255) && (*ptrp == 255))
			*ptr = 255;
		else
			*ptr = 0;
		ptr++;
		ptrp++;
	}
	dilate(img_copy, img_copy, element);
	dilate(img_copy, img_copy, element);
	dilate(img_copy, img_copy, element);
//	imshow("silt",img_copy);


	vector<vector<Point>> contours;
	findContours(img_copy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	for (int j = 0; j < contours.size(); j++)
	{
		double area = contourArea(contours[j]);
		if (area>10)
		{
			RotatedRect nrect = minAreaRect(contours[j]);
			float ratio = nrect.size.width / nrect.size.height;
			float shortlen = nrect.size.width > nrect.size.height ? nrect.size.height : nrect.size.width;
			if ((ratio<0.3 || ratio>3.3) && (shortlen>8))                                 //长宽比符合要求的且短边大于8的为裂缝轮廓
			{
				Point2f topoint[4];
				nrect.points(topoint);
				for (int i = 0; i < 4; i++)
					line(imgOri, topoint[i], topoint[(i + 1) % 4], Scalar(255, 0, 0));
			}
		}
	}
}

void Fenge1(Mat &src, Point O, Point A, Point D, vector<Vec2i> &vertex)
{
	int R1 = vertex[8][0];  //小半径
	int R2 = vertex[8][1];  //大半径
	int t1, t2, t3, t4;
	float k1, k2, k3, k4; //斜率
	int d, s;//s代表边缘误差，d代表平移（微小）
	Point A1 = Point(vertex[4][1], vertex[4][0]);
	Point B1 = Point(vertex[6][1], vertex[6][0]);
	Point A2 = Point(vertex[5][1], vertex[5][0]);
	Point B2 = Point(vertex[7][1], vertex[7][0]);
	Point A3 = Point(vertex[0][1], vertex[0][0]);
	Point B3 = Point(vertex[2][1], vertex[2][0]);
	Point A4 = Point(vertex[1][1], vertex[1][0]);
	Point B4 = Point(vertex[3][1], vertex[3][0]);
	k1 = float((B1.y - A1.y) / (B1.x - A1.x));
	k2 = float((B2.y - A2.y) / (B2.x - A2.x));
	k3 = float((B3.y - A3.y) / (B3.x - A3.x));
	k4 = float((B4.y - A4.y) / (B4.x - A4.x));
	Mat Lab = src;
	int row = Lab.rows;
	int col = Lab.cols;
	int i, j;
	vector<Mat> img(5);                        //分割成五个区域,用Mat类数组容器存放分割区域图片
	Mat copy(Lab.rows, Lab.cols, Lab.type(), Scalar(255, 255, 255));
	copy.copyTo(img[0]);
	copy.copyTo(img[1]);
	copy.copyTo(img[2]);
	copy.copyTo(img[3]);
	copy.copyTo(img[4]);
	d = 2;//边界差
	s = 6;
	Point P0 = Point(O.x - R1, O.y);
	//求t1,t2,t3,t4

	k1 = 1 / k1;
	k2 = 1 / k2;
	k3 = 1 / k3;
	k4 = 1 / k4;
	/*cout << "k1:" << k1 << endl;
	cout << "k2:" << k2 << endl;
	cout << "k3:" << k3 << endl;
	cout << "k4:" << k4 << endl;*/
	for (i = A.y; i < A.y + 300; i++)
	{
		j = k1*i + A1.x - A1.y*k1;

		if (abs(length(O, Point(j, i)) - R2)<3)
		{
			t1 = i - A.y + 4;
			break;
		}
	}
	//j = k1*A.y + A1.x - A1.y*k1;
	//circle(src, Point(j,i),3, Scalar(0, 0, 0), -1);
	//imshow("ss", src);
	//waitKey(0);
	for (i = 1; i < 100; i++)
	{
		uchar *data1 = src.ptr<uchar>(P0.y + i);
		uchar *data2 = src.ptr<uchar>(P0.y - i);
		if (data1[P0.x] < 100 && data2[P0.x] < 100)
		{
			t2 = O.y - i - A.y;
			t4 = D.y - i - O.y;
			break;
		}
	}
	for (i = D.y - 250; i < D.y; i++)
	{
		j = k3*i - A3.y*k3 + A3.x;
		if (abs(length(O, Point(j, i)) - R2)<3)
		{
			t3 = D.y - i;
			break;
		}
	}

	//分割得到第一块
	for (i = A.y + d; i < A.y + t1 - d+2; i++)                //鄙人改过   +2
	{
		uchar *data1 = img[0].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = A.x + d; j < D.x - d; j++)
		{
			if (length(O, Point(j, i))>R2 + s + d+3)              //鄙人改过  +3
			{
				data1[j] = data2[j];
			}
		}
	}

		//namedWindow("img0", CV_WINDOW_NORMAL);
		//cv::imshow("img0", img[0]);
	//	imwrite("img0.bmp", img[0]);
	////////分割得到第二块
	for (i = A.y; i < A.y + t2 - s - d; i++)
	{
		uchar *data1 = img[1].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = k1*i + A1.x - A1.y*k1 + s; j < k2*i + A2.x - A2.y*k2 - s; j++)
		{
			if ((length(O, Point(j, i))<R2 - d) && length(O, Point(j, i))>R1 + d+2)     ////鄙人改过  +2
			{
				data1[j] = data2[j];
			}
		}
	}

	//cv::namedWindow("img1", CV_WINDOW_NORMAL);
	//cv::imshow("img1", img[1]);
	//imwrite("img1.bmp", img[1]);
	///分割得到第三块的圆
	for (i = O.y - R1 - 10; i < O.y + R1 + 10; i++)
	{
		uchar *data1 = img[2].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = O.x - R1 - 10; j < O.x + R1 + 10; j++)
		{
			if (length(O, Point(j, i))<R1 - d)
			{
				data1[j] = data2[j];
			}
		}
	}

	//cv::namedWindow("img2", CV_WINDOW_NORMAL);
	//cv::imshow("img2", img[2]);
	//imwrite("img2.bmp", img[2]);
	//分割得到第四块
	for (i = D.y - t4 + 2 * s + d; i < D.y; i++)
	{
		uchar *data1 = img[3].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = k3*i - A3.y*k3 + A3.x + d + s; j < k4*i - A4.y*k4 + A4.x - d - s; j++)
		{
			if ((length(O, Point(j, i))<R2 - d) && length(O, Point(j, i))>R1 + d+4)              //鄙人改过  +4
			{
				data1[j] = data2[j];
			}
		}
	}

	//cv::namedWindow("img3", CV_WINDOW_NORMAL);
	//cv::imshow("img3", img[3]);
	//imwrite("img3.bmp", img[3]);
	//分割得到第五块
	for (i = D.y - t3 + d; i <D.y - d; i++)                       //
	{
		uchar *data1 = img[4].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = A.x + d; j < D.x - d+3; j++)                                 //  鄙人改过  +4
		{
			if (length(O, Point(j, i))>R2 + d + s+4)                ////鄙人改过  +4
			{
				data1[j] = data2[j];
			}
		}
	}
	/*cv::namedWindow("img4", CV_WINDOW_NORMAL);
	cv::imshow("img4", img[4]);*/
	//imwrite("img4.bmp", img[4]);
	cvtColor(src, src, CV_GRAY2RGB);
	for (size_t i = 0; i < 5; i++)
	{
		if ((i == 1) || (i == 3))
			StainTest(src, img[i], 2);
		FlawTest(src, img[i], 2, i);
	}
//	namedWindow("TestResult",CV_WINDOW_NORMAL);
	imshow("TestResult",src);
}

void Fenge2(Mat &src, Point O, Point A, Point D, int *radiu)
{
	int R1 = radiu[0];  //小半径
	int R2 = radiu[1];  //大半径
	int t1, t2, t;
	int d = 3;//移动值   3
	int s = 6;//边缘误差值
	A.x = A.x - 11;
	D.x = D.x + 11;//补充阴影
	Mat Lab = src;
	int row = Lab.rows;
	int col = Lab.cols;
	int i, j;
	vector<Mat> img(5);                        //分割成五个区域,用Mat类数组容器存放分割区域图片
	Mat copy(Lab.rows, Lab.cols, Lab.type(), Scalar(255, 255, 255));
	copy.copyTo(img[0]);
	copy.copyTo(img[1]);
	copy.copyTo(img[2]);
	copy.copyTo(img[3]);
	copy.copyTo(img[4]);
	for (i = A.y; i < A.y + 200; i++)
	{
		if (length(O, Point(O.x - R1, i)) - R2 < 2)
		{
			t1 = i - A.y;
			break;
		}
	}
	for (i = D.y - 200; i < D.y; i++)
	{
		if (length(O, Point(O.x - R1, i)) - R2 < 2)
		{
			t2 = D.y - i;
			break;
		}
	}

	//分割得到第一块
	for (i = A.y; i <A.y + t1 - d; i++)
	{
		uchar *data1 = img[0].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = A.x; j < D.x; j++)
		{
			if (length(O, Point(j, i))>R2 + s + d)         
			{
				data1[j] = data2[j];
			}
		}
	}
	//namedWindow("img0", CV_WINDOW_NORMAL);
	//imshow("img0", img[0]);
	//imwrite("img0.bmp", img[0]);
	////////分割得到第二块
	for (i = A.y; i < O.y; i++)
	{
		uchar *data1 = img[1].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = O.x - R1 + d; j < O.x + R1 - d; j++)
		{
			if ((length(O, Point(j, i))<R2 - d) && length(O, Point(j, i))>R1 + d)                     
			{ 
				data1[j] = data2[j];
			}
		}
	}

	//namedWindow("img1", CV_WINDOW_NORMAL);
	//imshow("img1", img[1]);
	//imwrite("img1.bmp", img[1]);
	///分割得到第三块的圆
	for (i = O.y - R1 - s - d; i < O.y + R1 + s + d; i++)
	{
		uchar *data1 = img[2].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = O.x - R1 - s - d; j < O.x + R1 + s + d; j++)
		{
			if (length(O, Point(j, i))<R1 - 1 - d+3)                           //鄙人改过  +3
			{
				data1[j] = data2[j];
			}
		}
	}

	//namedWindow("img2", CV_WINDOW_NORMAL);
	//imshow("img2", img[2]);
	//imwrite("img2.bmp", img[2]);
	//分割得到第四块
	for (i = O.y; i < O.y + R2 - d; i++)
	{
		uchar *data1 = img[3].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = O.x - R1 + d; j < O.x + R1 - d; j++)
		{
			if ((length(O, Point(j, i))<R2 - d) && length(O, Point(j, i))>R1 + d+3)      //鄙人改过  +4
			{
				data1[j] = data2[j];
			}
		}
	}

	//namedWindow("img3", CV_WINDOW_NORMAL);
	//imshow("img3", img[3]);
	//imwrite("img3.bmp", img[3]);
	//分割得到第五块
	for (i = D.y - t1 - d; i <D.y; i++)
	{
		uchar *data1 = img[4].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = A.x; j < D.x - d; j++)
		{
			if (length(O, Point(j, i))>R2 + s + d)
			{
				data1[j] = data2[j];
			}
		}
	}
	//namedWindow("img4", CV_WINDOW_NORMAL);
	//imshow("img4", img[4]);
	//imwrite("img4.bmp", img[4]);
	//对五块分割的图像做检测处理
	cvtColor(src,src,CV_GRAY2BGR);

	for (size_t i = 0; i < 5; i++)
	{
		if ((i == 1) || (i == 3))
			StainTest(src, img[i], 1);
		FlawTest(src, img[i], 1, i);
	}
//	namedWindow("TestResult",CV_WINDOW_NORMAL);
	imshow("TestResult", src);
}

void Fenge3(Mat &src, vector<Vec2i> &vertex)
{
	//已知的参考数据
	int t2 = 145;
	//int t3 = 132;
	int t4 = t2;
	if (vertex[9][0] != 0 && vertex[10][0] != 0 && vertex[11][0] != 0 && vertex[12][0] != 0 && vertex[13][0] != 0 && vertex[14][0] != 0)
	{
		//更新固定值
		t2 = vertex[14][0] - vertex[20][0];
		//	t3 = vertex[11][0] - vertex[14][0];
		t4 = vertex[5][0] - vertex[11][0];
	}
	int d = 2;//移动值
	int s = 6;//边缘误差值
	Mat Lab = src;
	int row = Lab.rows;
	int col = Lab.cols;
	int i, j;
	vector<Mat> img(5);                        //分割成五个区域,用Mat类数组容器存放分割区域图片
	Mat copy(Lab.rows, Lab.cols, Lab.type(), Scalar(255, 255, 255));
	copy.copyTo(img[0]);
	copy.copyTo(img[1]);
	copy.copyTo(img[2]);
	copy.copyTo(img[3]);
	copy.copyTo(img[4]);

	//img1 = Lab(Range(A.x, D.x), Range(A.y, r1));
	//分割得到第一块
	for (i = vertex[21][0] + d + s; i <vertex[18][0] - d - s; i++)
	{
		uchar *data1 = img[0].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = vertex[21][1] + d; j < vertex[23][1] - d - s; j++)
		{
			data1[j] = data2[j];
		}
	}
	/*cv::namedWindow("img0", CV_WINDOW_NORMAL);
	cv::imshow("img0", img[0]);*/

	////////分割得到第二块
	for (i = vertex[19][0] + s; i < vertex[19][0] + t2 - d - s; i++)
	{
		uchar *data1 = img[1].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = vertex[19][1] + d + s; j <vertex[20][1] - d - s; j++)
		{
			data1[j] = data2[j];
		}
	}

	/*namedWindow("img1", CV_WINDOW_NORMAL);
	imshow("img1", img[1]);*/
	///分割得到第三块
	for (i = vertex[15][0] + d; i < vertex[6][0] - d; i++)
	{
		uchar *data1 = img[2].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = vertex[15][1] + d; j < vertex[16][1] - s - d; j++)
		{
			data1[j] = data2[j];
		}
	}
	for (i = vertex[20][0] + t2 + d + s; i <vertex[5][0] - t4 - d - s; i++)
	{
		uchar *data1 = img[2].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = vertex[16][1] - d - s; j < vertex[17][1] - d - s; j++)
		{
			data1[j] = data2[j];
		}
	}

	/*namedWindow("img2", CV_WINDOW_NORMAL);
	imshow("img2", img[2]);*/
	//分割得到第四块
	for (i = vertex[4][0] - t4 + s + d; i <vertex[4][0] - d - s; i++)
	{
		uchar *data1 = img[3].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = vertex[4][1] + d + s; j <vertex[5][1] - d - s; j++)
		{
			data1[j] = data2[j];
		}
	}

	/*namedWindow("img3", CV_WINDOW_NORMAL);
	imshow("img3", img[3]);*/
	//分割得到第五块
	for (i = vertex[3][0] + d + s; i <vertex[0][0] - d - s; i++)
	{
		uchar *data1 = img[4].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = vertex[3][1] + d; j < vertex[5][1] - d - s; j++)
		{
			data1[j] = data2[j];
		}
	}

	/*namedWindow("img4", CV_WINDOW_NORMAL);
	imshow("img4", img[4]);*/
	//对五块分区的图像做处理

}

void Fenge4(Mat &src, vector<Vec2i> &vertex)
{
	//已知的模板参数

	int d1 = 40;
	int d = 3;//移动值
	int s = 5; ////边缘误差值
	Mat Lab = src;
	int row = Lab.rows;
	int col = Lab.cols;
	int i, j;
	//分割成三个区域
	vector<Mat> img(3);                        //分割成三个区域,用Mat类数组容器存放分割区域图片
	Mat copy(Lab.rows, Lab.cols, Lab.type(), Scalar(255, 255, 255));
	copy.copyTo(img[0]);
	copy.copyTo(img[1]);
	copy.copyTo(img[2]);

	//分割得到第一块
	for (i = vertex[15][0] + d + s; i <vertex[12][0] - d - s; i++)
	{
		uchar *data1 = img[0].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = vertex[15][1] + d; j <vertex[16][1] + d1; j++)
		{
			data1[j] = data2[j];
		}
	}
	for (i = vertex[15][0] + d + s; i < vertex[0][0] - d - s; i++)
	{
		uchar *data1 = img[0].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = vertex[16][1] + d1; j < vertex[17][1] - d - s; j++)
		{
			data1[j] = data2[j];
		}
	}
	for (i = vertex[3][0] + d + s; i <vertex[0][0] - d - s; i++)
	{
		uchar *data1 = img[0].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = vertex[3][1] + d; j < vertex[4][1] + d1; j++)
		{
			data1[j] = data2[j];
		}
	}
	/*namedWindow("img0", CV_WINDOW_NORMAL);
	imshow("img0", img[0]);*/

	////////分割得到第二块
	for (i = vertex[13][0] + s + d; i < vertex[4][0] - s - d; i++)
	{
		uchar *data1 = img[1].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = vertex[13][1] + d + s; j < vertex[13][1] + d1; j++)
		{
			data1[j] = data2[j];
		}
	}


	/*namedWindow("img1", CV_WINDOW_NORMAL);
	imshow("img1", img[1]);*/
	///分割得到第三块
	for (i = vertex[9][0] + d; i < vertex[6][0] - d; i++)
	{
		uchar *data1 = img[2].ptr<uchar>(i);
		uchar *data2 = Lab.ptr<uchar>(i);
		for (j = vertex[9][1] + d; j < vertex[10][1] - d - s; j++)
		{
			data1[j] = data2[j];
		}

	}

	/*namedWindow("img2", CV_WINDOW_NORMAL);
	imshow("img2", img[2]);*/
	//对分区的三块图像做检测处理

}

void Fenge5(Mat &src, Point A5, Point D5)
{
	int d = 2;//移动值
	int s = 2; ////边缘误差值
	int row = src.rows;
	int col = src.cols;
	int i, j;
	Mat img1(src.rows, src.cols, src.type(), Scalar(255, 255, 255));
	//分割矩形
	for (i = A5.y + d + s-2; i <D5.y - d - s-2; i++)                      //
	{
		uchar *data1 = img1.ptr<uchar>(i);
		uchar *data2 = src.ptr<uchar>(i);
		for (j = A5.x + d + +3 + s; j < D5.x - d - 3 - s; j++)
		{
			data1[j] = data2[j];
		}
	}
	//namedWindow("img1", CV_WINDOW_NORMAL);
	imshow("img1", img1);
	cvtColor(src, src, CV_GRAY2BGR);
	FlawTest(src, img1, 3, 0);
	SiltTest(src, img1);
	//namedWindow("TestResult",CV_WINDOW_NORMAL);
	imshow("TestResult", src);
	//对分区后的图像做检测处理
}
