#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	int RemoveCount = 0;       //��¼��ȥ�ĸ���  
	//��¼ÿ�����ص����״̬�ı�ǩ��0����δ��飬1�������ڼ��,2�����鲻�ϸ���Ҫ��ת��ɫ����3������ϸ������  
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
		//cout << "Mode: ȥ��С����. ";
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
		//cout << "Mode: ȥ���׶�. ";
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

	vector<Point> NeihborPos;  //��¼�����λ��  
	NeihborPos.push_back(Point(-1, 0));
	NeihborPos.push_back(Point(1, 0));
	NeihborPos.push_back(Point(0, -1));
	NeihborPos.push_back(Point(0, 1));
	if (NeihborMode == 1)
	{
		/*	cout << "Neighbor mode: 8����." << endl;*/
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	//else cout << "Neighbor mode: 4����." << endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//��ʼ���  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********��ʼ�õ㴦�ļ��**********  
				vector<Point2i> GrowBuffer;                                      //��ջ�����ڴ洢������  
				GrowBuffer.push_back(Point2i(j, i));
				Pointlabel.at<uchar>(i, j) = 1;
				int CheckResult = 0;                                               //�����жϽ�����Ƿ񳬳���С����0Ϊδ������1Ϊ����  

				for (int z = 0; z<GrowBuffer.size(); z++)
				{

					for (int q = 0; q<NeihborCount; q++)                                      //����ĸ������  
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX<Src.cols&&CurrY >= 0 && CurrY<Src.rows)  //��ֹԽ��  
						{
							if (Pointlabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //��������buffer  
								Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //���������ļ���ǩ�������ظ����  
							}
						}
					}

				}
				if (GrowBuffer.size()>AreaLimit) CheckResult = 2;                 //�жϽ�����Ƿ񳬳��޶��Ĵ�С����1Ϊδ������2Ϊ����  
				else { CheckResult = 1;   RemoveCount++; }
				for (int z = 0; z<GrowBuffer.size(); z++)                         //����Label��¼  
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
				}
				//********�����õ㴦�ļ��**********  


			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//��ʼ��ת�����С������  
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
Mat histcount(Mat &img, int *ptri, uchar *ptr, int &count)   //���Ҷ�ͼ��ֱ��ͼ,ptriΪ���ͼ��Ҷȼ����������Ķ�Ӧ��ϵ����������ptrָ��ͼ���һ�����ء�
{
	int maxheight = 256;
	for (int i = 0; i < img.rows*img.cols; i++)
	{
		if (ptr[i] != 255)
		{
			++ptri[ptr[i]];
			count++;
		}
	}	                                 //ͳ��ԭͼ���Ҷȼ����ص����
	ptri[255] = 0;
	int max = maxval(ptri, 256);
	Mat hist(256, 512, CV_8UC3);                         //ֱ��ͼ��512����256
	for (int i = 0; i < 256; i++)
	{
		double height = ptri[i] * maxheight / (1.0*max);
		rectangle(hist, Point(i * 2, 255), Point((i + 1) * 2 - 1, 255 - height), Scalar(0, 0, 255));
	}
	return hist;
}
void histogramequ(Mat &img, Mat img_htg)                       //ֱ��ͼ���⺯�����������ͼ���Сһ��
{
	int count = 0;
	uchar* ptr = img.ptr<uchar>(0);
	uchar* ptr_htg = img_htg.ptr<uchar>(0);                   //���Դ��ԭͼ���Ҷȼ����ص����
	int grayori[256] = { 0 };
	double  grayp[256] = { 0 };
	int graynew[256] = { 0 };                                //����ֱ��ͼ�������Ҷȼ�ӳ���ϵ
	Mat histori/*(256, 512, CV_8UC3)*/;
	histori = histcount(img, grayori, ptr, count);
	//	imshow("hist", histori);                                //��ԭͼ��ֱ��ͼ
	for (int i = 0; i < 256; i++)
		grayp[i] = grayori[i] / (1.0*count);               //ԭͼ���Ҷȼ����ص����ռ�����ظ�������
	double sum = 0;
	int sumi;
	for (int i = 0; i < 256; i++)
	{
		sum += grayp[i];
		sumi = int(255 * sum + 0.5);

		graynew[i] = sumi;                      //����ӳ���ϵ
	}
	Mat hist_htg/*(256, 512, CV_8UC3)*/;
	for (int i = 0; i < img.rows*img.cols; i++)        //ֱ��ͼ����
	{
		if (ptr_htg[i] != 255)
		{
			ptr_htg[i] = graynew[ptr[i]];
		}
	}
	count = 0;
	memset(graynew, 0, sizeof(int)* 256);               //���洢ӳ���ϵ�������������Դ�ž����ͼ���ֱ��ͼͳ������
	hist_htg = histcount(img_htg, graynew, ptr_htg, count);
	//	imshow("hist_htg", hist_htg);                     //�����ͼ��ֱ��ͼ
}
Mat BackgroundEqu(Mat &srcImage)
{
	const int subWindowSize = 8;	//�Ӵ��ڴ�С
	//�Ӵ��ڻҶȾ�ֵ
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

	long long imgIntensity = 0;			//ȫͼ�ҶȾ�ֵ
	for (auto i = 0; i < srcImage.rows; ++i)
	{
		for (auto j = 0; j < srcImage.cols; ++j)
		{
			imgIntensity += srcImage.at<unsigned char>(i, j);
		}
	}
	imgIntensity /= (srcImage.cols * srcImage.rows);

	for (auto i = 0; i < subMaskIntensity.rows; ++i)		//����������
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
	vector<int> statistic;			//��ֱ����ͳ��
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
				cout << counter << ",";		//���ͳ������
	}
	cvtColor(srcImage, srcImage, CV_GRAY2BGR);
	for (auto i = 0; i < statistic.size(); ++i)
	{
		if (statistic[i] > 550)		//�ж��ѷ����ֵ
		{
			auto j = i + 1;
			for (; j < statistic.size(); ++j)
			{
				if (statistic[j] < 155) break;		//�ѷ������ֵ
			}
			if (j - i > 2)
			{
				cout << endl << "�ѷ���" << i * 10 << "�е�" << j * 10 << "��֮��" << endl;
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