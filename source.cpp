#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	int RemoveCount = 0;       //记录除去的个数  
	//记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查  
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
		//cout << "Mode: 去除小区域. ";
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
		//cout << "Mode: 去除孔洞. ";
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
		/*	cout << "Neighbor mode: 8邻域." << endl;*/
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	//else cout << "Neighbor mode: 4邻域." << endl;
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

	/*cout << RemoveCount << " objects removed." << endl;*/
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
void histogramequ(Mat &img, Mat img_htg)                       //直方图均衡函数，输入输出图像大小一致
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
	memset(graynew, 0, sizeof(int)* 256);               //将存储映射关系的数组清零用以存放均衡后图像的直方图统计数据
	hist_htg = histcount(img_htg, graynew, ptr_htg, count);
	//	imshow("hist_htg", hist_htg);                     //均衡后图像直方图
}
Mat BackgroundEqu(Mat &srcImage)
{
	const int subWindowSize = 8;	//子窗口大小
	//子窗口灰度均值
	Mat subMaskIntensity(srcImage.rows / subWindowSize, srcImage.cols / subWindowSize, CV_8UC1);
	for (auto i = 0; i < subMaskIntensity.rows; ++i)
	{
		for (auto j = 0; j < subMaskIntensity.cols; ++j)
		{
			long long intensity = 0;
			for (auto k = 0; k < subWindowSize; ++k)
			{
				for (auto m = 0; m < subWindowSize; ++m)
				{
					intensity += srcImage.at<unsigned char>(subWindowSize * i + k, subWindowSize * j + m);
				}
			}
			subMaskIntensity.at<unsigned char>(i, j) = intensity / (subWindowSize * subWindowSize);
		}
	}

	for (auto i = 1; i < subMaskIntensity.rows - 1; ++i)
	{
		for (auto j = 1; j < subMaskIntensity.cols - 1; ++j)
		{
			if (subMaskIntensity.at<unsigned char>(i, j) < subMaskIntensity.at<unsigned char>(i - 1, j) &&
				subMaskIntensity.at<unsigned char>(i, j) < subMaskIntensity.at<unsigned char>(i + 1, j))
			{
				subMaskIntensity.at<unsigned char>(i, j) = (subMaskIntensity.at<unsigned char>(i - 1, j) + subMaskIntensity.at<unsigned char>(i + 1, j)) / 2;
			}
			if (subMaskIntensity.at<unsigned char>(i, j) < subMaskIntensity.at<unsigned char>(i, j - 1) &&
				subMaskIntensity.at<unsigned char>(i, j) < subMaskIntensity.at<unsigned char>(i, j + 1))
			{
				subMaskIntensity.at<unsigned char>(i, j) = (subMaskIntensity.at<unsigned char>(i, j - 1) + subMaskIntensity.at<unsigned char>(i, j + 1)) / 2;
			}
		}
	}

	long long imgIntensity = 0;			//全图灰度均值
	for (auto i = 0; i < srcImage.rows; ++i)
	{
		for (auto j = 0; j < srcImage.cols; ++j)
		{
			imgIntensity += srcImage.at<unsigned char>(i, j);
		}
	}
	imgIntensity /= (srcImage.cols * srcImage.rows);

	for (auto i = 0; i < subMaskIntensity.rows; ++i)		//消除背景光
	{
		for (auto j = 0; j < subMaskIntensity.cols; ++j)
		{
			double balanceFactor = double(imgIntensity) / subMaskIntensity.at<unsigned char>(i, j);
			for (auto k = 0; k < subWindowSize; ++k)
			{
				for (auto m = 0; m < subWindowSize; ++m)
				{
					srcImage.at<unsigned char>(subWindowSize * i + k, subWindowSize * j + m) *= balanceFactor;
				}
			}
		}
	}
	return srcImage;
}
Mat SilTest(Mat &img)
{
	Mat img_copy,img_1;
	img.copyTo(img_copy);
//	imshow("ori", img_copy);
	histogramequ(img_copy, img_copy);
//	imshow("htg", img_copy);



	threshold(img_copy, img_copy, 15, 255, THRESH_BINARY);                              //15            32
//	imshow("bin", img_copy);
	RemoveSmallRegion(img_copy, img_copy, 8, 0, 1);                       //15
//	imshow("remove", img_copy);
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	//morphologyEx(img_copy, img_copy, MORPH_CLOSE, element);
	erode(img_copy, img_1, element);                                          //3
	RemoveSmallRegion(img_1, img_1, 120, 0, 1);                                  //100
//	imshow("result", img_1);
	return img_1;
}
void ISilt(Mat &srcImage)
{
	vector<int> statistic;			//垂直窗口统计
	const int staWindowSize = 10;
	for (auto i = 0; i < srcImage.rows / staWindowSize; ++i)
	{
		int counter = 0;
		for (auto m = 0; m < staWindowSize; ++m)
		{
			for (auto j = 0; j < srcImage.cols; ++j)
			{
				if (srcImage.at<unsigned char>(i * staWindowSize + m, j) < 50)
				{
					++counter;
				}
			}
		}
		statistic.push_back(counter);
				cout << counter << ",";		//输出统计数据
	}
	cvtColor(srcImage, srcImage, CV_GRAY2BGR);
	for (auto i = 0; i < statistic.size(); ++i)
	{
		if (statistic[i] > 550)		//判断裂缝的阈值
		{
			auto j = i + 1;
			for (; j < statistic.size(); ++j)
			{
				if (statistic[j] < 155) break;		//裂缝结束阈值
			}
			if (j - i > 2)
			{
				cout << endl << "裂缝在" << i * 10 << "行到" << j * 10 << "行之间" << endl;
				line(srcImage, Point(0, j*10), Point(srcImage.cols, j*10), Scalar(0, 0, 255), 2, 4);
				line(srcImage, Point(0, i * 10), Point(srcImage.cols, i * 10), Scalar(0, 0, 255), 2, 4);
				break;
			}
		}
	}
	imshow("result", srcImage);
	
}
int main()
{
	int64 timestart = 0, timeend = 0;
	timestart = getTickCount();
	string imgpath = "E:\\Desktop\\defect-detection\\testImg\\";
	string imgname = imgpath + "11.1.bmp";
	Mat img = imread(imgname, 0);
	img = BackgroundEqu(img);
	img = SilTest(img);
	ISilt(img);
	timeend = getTickCount();
	cout << "\nhas passed: " << 1000.0*(timeend - timestart) / getTickFrequency() << " ms" << endl;
	waitKey(0);
	return 0;
}