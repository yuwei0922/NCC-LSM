// NCC+LSM.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <opencv2/core/core.hpp>  
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>   
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include <stdlib.h>
#include <Windows.h>
#include <commdlg.h>
#include <atlstr.h>
#include <atlconv.h>
#include <time.h>
#include <iostream>   
#include <fstream>   

using namespace std;
using namespace cv; 

struct params_moravec {
	int winSize;//size of the window of interest value
	int threshold;//empirical threshold
	int restrainWinSize;//suppressing window
};
struct params_harris {
	int blockSize;//size of neighbor window 2*blocksize+1
	int apertureSize;//sobel window
	double rc;//harris responding cofficient-- 32-bit float
	double thHarrisRes;//threshold
};
struct params {
	params_moravec par_m;
	params_harris par_h;
};

struct params_ncc {
	int matchsize;//length of the square window
	int PreSearchRadius; //radius of the pre-search area for the right window
	float lowst_door; //threshold of correlation coefficient
	int dist_height;
	int dist_width;//relative distance between left and right photos(manually measured)
};

class CMatch {
public:
	string path_left, path_right;
	Mat result;
	vector <Point3f> featurePointLeft, featurePointRight;
	params par;
	params_ncc par2;
	CMatch(string pathLeft, string pathRight) {
		this->path_left = pathLeft;
		this->path_right = pathRight;

		//params setting
		//moravec params
		params_moravec par1_m;
		par1_m.winSize = 9;
		par1_m.threshold = 8000;
		par1_m.restrainWinSize = 80;
		//harris params
		params_harris par1_h;
		par1_h.blockSize = 4;
		par1_h.apertureSize = 5;
		par1_h.rc = 0.05;
		par1_h.thHarrisRes = 130;

		par.par_h = par1_h; par.par_m = par1_m;
		//ncc params
		par2.matchsize = 9;
		par2.PreSearchRadius = 15;
		par2.dist_width = 767;
		par2.dist_height = 0;
		par2.lowst_door = 0.85;
	};
	~CMatch() {};
	
	//Perspective transform
	Mat Transform(string right_path, string left_path);
	//Harris(image path，par(blockSize,apertureSize,rc,thHarrisRes),FeaturePoints)
	Mat Harris(string path, params_harris par, vector<Point3f> &featurePt);
	//Moravec(image，par(WinSize,Threshold，RestrainWinSize),FeaturePoints)
	Mat Moravec(string path, params_moravec par, vector <Point3f> &featurePt);
	//NCC & LSM feature images main
	void Get_featureimage(int kind, string srcImg_path, params par, vector <Point3f> &featurePt);
	//NCC & LSM utility
	float Get_coefficient(Mat matchLeftWindow, Mat imageRight, int x, int y);
	void Vector_Sort(vector < Point3f> &Temp_sort);
	Mat View(Mat imageLeftRGB, Mat imageRightRGB, vector<Point3f> featurePointLeft, vector<Point3f> featurePointRight);
	//NCC Match
	Mat NCCMatchingImg(string path_left, string path_right, params_ncc par, vector<Point3f> featurePointLeft);
	//LSM Match
	Mat LSMatchingImg(string path_left, string path_right, params_ncc par, vector<Point3f> featurePointLeft);
	void Get_matchresult(int kind, string path_left, string path_right, params_ncc par, vector<Point3f> featurePointLeft);
};

int main()
{
	cout << "Please choose left image:" << endl;
	OPENFILENAME ofn1;      //common dialog box structure    
	TCHAR szFile1[MAX_PATH]; //buffer for saving the retrieved file name              
	//initialize open file dialog     
	ZeroMemory(&ofn1, sizeof(OPENFILENAME));
	ofn1.lStructSize = sizeof(OPENFILENAME);
	ofn1.hwndOwner = NULL;
	ofn1.lpstrFile = szFile1;
	ofn1.lpstrFile[0] = '\0';
	ofn1.nMaxFile = sizeof(szFile1);
	ofn1.lpstrFilter = (LPCWSTR)"All(*.*)\0*.*\0\0";
	ofn1.nFilterIndex = 1;
	ofn1.lpstrFileTitle = NULL;
	ofn1.nMaxFileTitle = 0;
	ofn1.lpstrInitialDir = NULL;
	ofn1.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

	//display open file dialog
	if (GetOpenFileName(&ofn1))
	{
		//display selected file 
		OutputDebugString(szFile1);
		OutputDebugString((LPCWSTR)"\r\n");
	}

	wstring wstr1(szFile1);
	string pathLeft(wstr1.begin(), wstr1.end());
	cout << "File path:" << pathLeft << endl;

	cout << "Please choose right image:" << endl;
	OPENFILENAME ofn2;      //common dialog box structure      
	TCHAR szFile2[MAX_PATH]; //buffer for saving the retrieved file name               
	//initialize open file dialog          
	ZeroMemory(&ofn2, sizeof(OPENFILENAME));
	ofn2.lStructSize = sizeof(OPENFILENAME);
	ofn2.hwndOwner = NULL;
	ofn2.lpstrFile = szFile2;
	ofn2.lpstrFile[0] = '\0';
	ofn2.nMaxFile = sizeof(szFile2);
	ofn2.lpstrFilter = (LPCWSTR)"All(*.*)\0*.*\0\0";
	ofn2.nFilterIndex = 1;
	ofn2.lpstrFileTitle = NULL;
	ofn2.nMaxFileTitle = 0;
	ofn2.lpstrInitialDir = NULL;
	ofn2.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

	//display open file dialog
	if (GetOpenFileName(&ofn2))
	{
		//display selected file 
		OutputDebugString(szFile2);
		OutputDebugString((LPCWSTR)"\r\n");
	}

	wstring wstr2(szFile2);
	string pathRight(wstr2.begin(), wstr2.end());
	cout << "File path:" << pathRight << endl;
	int m, n;
	cout << "________________________________________________________________________________" << endl << "\n";

	CMatch match(pathLeft, pathRight);

	Mat trans = match.Transform(pathRight, pathLeft);
	pathRight = "Affine_transformed.jpg";//Affine_transformed
	
	//Get_feature_params：1——Moravec；2——Harris；3——SIFT
	cout << "Select feature extraction operators：(1—Moravec；2—Harris)" << endl;
	cin >> m;
	cout << "---------------------------------------" << endl;
	if (m == 1)
	{
		cout << "Parameter settings:" << endl;
		cout << "winSize:";
		cin >> match.par.par_m.winSize;
		cout << "threshold:";
		cin >> match.par.par_m.threshold;
		cout << "restrainWinSize:";
		cin >> match.par.par_m.restrainWinSize;
		cout << "Getting Moravec feature..." << endl;
	}
	else if (m == 2)
	{
		cout << "Parameter settings:" << endl;
		cout << "blockSize:";
		cin >> match.par.par_h.blockSize;
		cout << "apertureSize:";
		cin >> match.par.par_h.apertureSize;
		cout << "rc:";
		cin >> match.par.par_h.rc;
		cout << "thHarrisRes:";
		cin >> match.par.par_h.thHarrisRes;
		cout << "Getting Harris feature..." << endl;
	}
	else
	{
		cout << "No Such solution!" << endl;
		return 0;
	}
	match.Get_featureimage(m, pathLeft, match.par, match.featurePointLeft);
	
	cout << "Select match method：(1—NCC；2—LSM)" << endl;
	cin >> n;
	cout << "---------------------------------------" << endl;
	cout << "Parameter settings:" << endl;
	cout << "matchsize:";
	cin >> match.par2.matchsize;
	cout << "PreSearchRadius:";
	cin >> match.par2.PreSearchRadius;
	cout << "lowst_door:";
	cin >> match.par2.lowst_door;

	if (n == 1)
	{
		cout << "Getting NCC match result..." << endl;
	}
	else if (n == 2)
	{
		cout << "Getting LS match result..." << endl;
	}
	else
	{
		cout << "No Such solution!" << endl;
		return 0;
	}
	match.Get_matchresult(n, pathLeft, pathRight, match.par2, match.featurePointLeft);

}

Mat CMatch::Transform(string right_path, string left_path)
{
	// Read source image.
	Mat right_img = imread(right_path), left_img = imread(left_path);
	
	// Four corners of the book in source image:1
	vector <Point2f>pts1;
	// Four corners of the book in destination image.
	vector <Point2f>pts2;

		//299,141_179,111
		//267,776_151,742
		//713,626_590,592
		//575,997_455,959

		// Four corners of the book in source image:1
		pts1.push_back(Point2f(299, 141));
		pts1.push_back(Point2f(267, 776));
		pts1.push_back(Point2f(713, 626));
		pts1.push_back(Point2f(575, 997));

		// Four corners of the book in destination image.
		pts2.push_back(Point2f(179, 111));
		pts2.push_back(Point2f(151, 742));
		pts2.push_back(Point2f(590, 592));
		pts2.push_back(Point2f(455, 959));
	

	// Calculate Homography and set output image
	Mat H = findHomography(pts2, pts1, RANSAC);
	Mat r_transform = Mat::zeros(left_img.rows, left_img.cols, CV_8UC3);
	// Warp source image to destination based on homography
	warpPerspective(right_img, r_transform, H, left_img.size());

	imshow("Affine_transformed", r_transform);
	imwrite("Affine_transformed.jpg", r_transform);
	waitKey(10);
	return r_transform;

}

Mat CMatch::Moravec(string path, params_moravec par, vector <Point3f> &featurePt)
{
	Mat imageRGB = imread(path, cv::IMREAD_COLOR);
	if (imageRGB.empty())
	{
		cout << "Fail to read image:" << path << endl;
	}
	//creat grey-scale image for computation
	Mat srcImg;
	cvtColor(imageRGB, srcImg, cv::COLOR_RGB2GRAY);
	
	//Img size
	int rows = srcImg.rows; //y，rows
	int cols = srcImg.cols; //x，cols
	int k = par.winSize / 2;
	//Interests for Mats
	Mat valueMat = Mat::zeros(srcImg.rows, srcImg.cols, CV_32FC1);
	//the sum of squares of gray differences of adjacent pixels in four directions is calculated
	for (int c = k; c < srcImg.rows - k; c++)
	{//delete edges
		for (int r = k; r < srcImg.cols - k; r++)
		{
			int V1 = 0, V2 = 0, V3 = 0, V4 = 0;
			for (int i = -k; i <= k - 1; i++)
			{
				V1 += (srcImg.at<uchar>(c + i, r) - srcImg.at<uchar>(c + i + 1, r))*(srcImg.at<uchar>(c + i, r) - srcImg.at<uchar>(c + i + 1, r));
				V2 += (srcImg.at<uchar>(c + i, r + i) - srcImg.at<uchar>(c + i + 1, r + i + 1))*(srcImg.at<uchar>(c + i, r + i) - srcImg.at<uchar>(c + i + 1, r + i + 1));
				V3 += (srcImg.at<uchar>(c, r + i) - srcImg.at<uchar>(c, r + i + 1))*(srcImg.at<uchar>(c, r + i) - srcImg.at<uchar>(c, r + i + 1));
				V4 += (srcImg.at<uchar>(c + i, r - i) - srcImg.at<uchar>(c + i + 1, r - i - 1))*(srcImg.at<uchar>(c + i, r - i) - srcImg.at<uchar>(c + i + 1, r - i - 1));
			}
			//find min
			float IV = min(min(V1, V2), min(V3, V4));
			//save value
			valueMat.at<float>(c, r) = IV;
		}
	}
	//The points with local MAXIMUM and BIGGER threshold are limited as candidate points as the final feature points.
	int windowSize = par.restrainWinSize;
	int halfWindow = windowSize / 2;
	for (int y = halfWindow; y < valueMat.rows - halfWindow; y += windowSize)
	{//delete edges
		for (int x = halfWindow; x < valueMat.cols - halfWindow; x += windowSize)
		{
			//window interests max ori
			float max = 0;
			bool Flag = 0;
			//save pts tmp:(x,y,value)
			Point3f pt;
			pt.x = -1;
			pt.y = -1;
			pt.z = 0;
			//(y,x)as middle ,windowSize*windowSize,find maximum value
			for (int i = -halfWindow; i <= halfWindow; i++)
			{
				for (int j = -halfWindow; j <= halfWindow; j++)
				{
					float value;
					//get val
					value = valueMat.at<float>(y + i, x + j);

					if (value > max)
					{
						max = value;
						pt.x = x + j;
						pt.y = y + i;
						pt.z = value;
						Flag = 1;
					}
				}
			}
			//after cal
			if (Flag == 1 && max > par.threshold)
			{//save pts
				featurePt.push_back(pt);
			}
		}
	}
	//draw color img
	Mat img_source = imread(path, IMREAD_COLOR);
	//draw feature point
	int radius = 4;
	for (int i = 0; i < featurePt.size(); i++)
	{//read feature points
		int xx = featurePt.at(i).x;
		int yy = featurePt.at(i).y;
		//draw circle
		circle(img_source, Point(xx, yy), radius, Scalar(0, 255, 255), 1, CV_AA);
		//draw number
		putText(img_source, to_string(i), Point(xx + 2, yy - 2), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 0, 0), 1.8, CV_AA);
		//draw cross
		line(img_source, Point(xx - radius - 1, yy), Point(xx + radius + 1, yy), Scalar(0, 255, 255), 1, CV_AA);
		line(img_source, Point(xx, yy - radius - 1), Point(xx, yy + radius + 1), Scalar(0, 255, 255), 1, CV_AA);
	}
	return img_source;
}

Mat CMatch::Harris(string path, params_harris par, vector<Point3f> &featurePt)
{
	Mat imageRGB = imread(path, cv::IMREAD_COLOR);
	if (imageRGB.empty())
	{
		cout << "Fail to read image:" << path << endl;
	}

	//creat grey-scale image for computation
	Mat imageGray;
	cvtColor(imageRGB, imageGray, cv::COLOR_RGB2GRAY);

	//creat result matrix
	Mat result/*32-bit float*/, resultNorm/*0-255 32-bit float*/, resultNormUInt8/*0-255 uisigned char*/;
	result = Mat::zeros(imageGray.size(), CV_32FC1);

	//Define harris detector
	int blockSize = par.blockSize;//size of neighbor window 2*blocksize+1
	int apertureSize = par.apertureSize;//sobel window
	double k = par.rc;//harris responding cofficient-- 32-bit float
	cornerHarris(imageGray, result, blockSize, apertureSize, k, BORDER_DEFAULT);

	//Normalizing image to 0-255
	normalize(result, resultNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(resultNorm, resultNormUInt8);
	//drawing circles around corners
	bool bMarkCorners = true;
	if (bMarkCorners)
	{
		double thHarrisRes = par.thHarrisRes;
		int radius = 4;
		int q = 0;
		for (int j = 0; j < resultNorm.rows; j++)
		{
			for (int i = 0; i < resultNorm.cols; i++)
			{
				if ((int)resultNorm.at<float>(j, i) > thHarrisRes)
				{
					q++;
					circle(resultNormUInt8, Point(i, j), radius, Scalar(255), 1, 8, 0);
					circle(imageRGB, Point(i, j), radius, Scalar(0, 255, 255), 1, 4, 0);
					putText(imageRGB, to_string(q), Point(i + 2, j - 2), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 0, 0), 1.8, CV_AA);
					line(imageRGB, Point(i - radius - 2, j), Point(i + radius + 2, j), Scalar(0, 255, 255), 1, 8, 0);
					line(imageRGB, Point(i, j - radius - 2), Point(i, j + radius + 2), Scalar(0, 255, 255), 1, 8, 0);
					Point3f tempP;
					tempP.x = j;
					tempP.y = i;
					tempP.z = resultNorm.at<float>(j, i);
					featurePt.push_back(tempP);
				}
			}
		}
	}
	return imageRGB;
}

void CMatch::Get_featureimage(int kind, string srcImg_path, params par, vector <Point3f> &featurePt) {
	//start the clock
	clock_t start = clock();
	clock_t finish;
	Mat srcImg = imread(srcImg_path, IMREAD_GRAYSCALE);
	if (srcImg.empty())
	{
		cout << "Could not read the left image" << endl;
		system("pause");
		return;
	}

	if (kind == 1) {
		//Moravec(img_Left, par(winSize,threshold,restrainWinSize), featurePointLeft)
		Mat featureImg_Moravec = Moravec(srcImg_path, par.par_m, featurePt);
		//show and save MoravecFeature image
		imshow("FeatureLeft_Moravec", featureImg_Moravec);
		imwrite("FeatureLeft_Moravec.jpg", featureImg_Moravec);
		finish = clock();
		waitKey(10);
	}
	else if (kind == 2) {
		//Harris(img_Left, par(winSize,threshold,restrainWinSize), featurePointLeft)
		Mat featureImg_Harris;
		featureImg_Harris = Harris(srcImg_path, par.par_h, featurePt);
		//show and save MoravecFeature image
		imshow("FeatureLeft_Harris", featureImg_Harris);
		imwrite("FeatureLeft_Harris.jpg", featureImg_Harris);
		finish = clock();
		waitKey(10);
	}
	cout << "---------------------------------------" << endl;
	cout << "feature points Number:" << featurePt.size() << endl;
	cout << "feature points extraction time：" << (double)(finish - start) / CLOCKS_PER_SEC << "s" << endl;
	cout << "Feature points of image saved." << endl;
	cout << "__________________________________________________________" << endl;
	cout << "Feature points extraction processing finished!" << endl <<endl;
	return;
}

//Matching based on ncc
float CMatch::Get_coefficient(Mat matchLeftWindow, Mat imageRight, int x, int y)
{
	//根据左搜索窗口确定右搜索窗口的大小
	Mat Rmatchwindow;
	Rmatchwindow.create(matchLeftWindow.rows, matchLeftWindow.cols, CV_32FC1);
	float aveRImg = 0;
	for (int m = 0; m < matchLeftWindow.rows; m++)
	{
		for (int n = 0; n < matchLeftWindow.cols; n++)
		{
			aveRImg += imageRight.at<uchar>(x + m, y + n);
			Rmatchwindow.at<float>(m, n) = imageRight.at<uchar>(x + m, y + n);
		}
	}
	aveRImg = aveRImg / (matchLeftWindow.rows*matchLeftWindow.cols);
	for (int m = 0; m < matchLeftWindow.rows; m++)
	{
		for (int n = 0; n < matchLeftWindow.cols; n++)
		{
			Rmatchwindow.at<float>(m, n) -= aveRImg;
		}
	}
	//开始计算相关系数
	float cofficent1 = 0;
	float cofficent2 = 0;
	float cofficent3 = 0;
	for (int m = 0; m < matchLeftWindow.rows; m++)
	{
		for (int n = 0; n < matchLeftWindow.cols; n++)
		{
			cofficent1 += matchLeftWindow.at<float>(m, n)*Rmatchwindow.at<float>(m, n);
			cofficent2 += Rmatchwindow.at<float>(m, n)*Rmatchwindow.at<float>(m, n);
			cofficent3 += matchLeftWindow.at<float>(m, n)*matchLeftWindow.at<float>(m, n);
		}
	}
	double cofficent = cofficent1 / sqrt(cofficent2 * cofficent3);
	return cofficent;
}

void CMatch::Vector_Sort(vector <Point3f> &Temp_sort)
{
	for (int i = 0; i < Temp_sort.size() - 1; i++) {
		float tem = 0;
		float temx = 0;
		float temy = 0;
		for (int j = i + 1; j < Temp_sort.size(); j++) {
			if (Temp_sort.at(i).z < Temp_sort.at(j).z) {
				tem = Temp_sort.at(j).z;
				Temp_sort.at(j).z = Temp_sort.at(i).z;
				Temp_sort.at(i).z = tem;

				temx = Temp_sort.at(j).x;
				Temp_sort.at(j).x = Temp_sort.at(i).x;
				Temp_sort.at(i).x = temx;

				temy = Temp_sort.at(j).y;
				Temp_sort.at(j).y = Temp_sort.at(i).y;
				Temp_sort.at(i).y = temy;
			}
		}
	}
}

Mat CMatch::View(Mat imageLeftRGB, Mat imageRightRGB, vector<Point3f> featurePointLeft, vector<Point3f> featurePointRight)
{
	Mat bothview;//output image
	bothview.create(imageLeftRGB.rows, imageLeftRGB.cols + imageRightRGB.cols, imageLeftRGB.type());
	for (int i = 0; i < imageLeftRGB.rows; i++)
	{
		for (int j = 0; j < imageLeftRGB.cols; j++)
		{
			bothview.at<Vec3b>(i, j) = imageLeftRGB.at<Vec3b>(i, j);
		}
	}

	for (int i = 0; i < imageRightRGB.rows; i++)
	{
		for (int j = imageLeftRGB.cols; j < imageLeftRGB.cols + imageRightRGB.cols; j++)
		{
			bothview.at<Vec3b>(i, j) = imageRightRGB.at<Vec3b>(i, j - imageLeftRGB.cols);
		}
	}//combine
	for (int i = 0; i < featurePointRight.size(); i++)
	{
		int a = (rand() % 200);
		int b = (rand() % 200 + 99);
		int c = (rand() % 200) - 50;
		if (a > 100 || a < 0)
		{
			a = 255;
		}
		if (b > 255 || b < 0)
		{
			b = 88;
		}
		if (c > 255 || c < 0)
		{
			c = 188;
		}
		int radius = 5;
		//left
		int lm = int(featurePointLeft.at(i).y);
		int ln = int(featurePointLeft.at(i).x);

		circle(bothview, Point(lm, ln), radius, Scalar(0, 255, 255), 1, 4, 0);
		line(bothview, Point(lm - radius - 2, ln), Point(lm + radius + 2, ln), Scalar(0, 255, 255), 1, 8, 0);
		line(bothview, Point(lm, ln - radius - 2), Point(lm, ln + radius + 2), Scalar(0, 255, 255), 1, 8, 0);

		//right
		int rm = int(featurePointRight.at(i).y + imageLeftRGB.cols);
		int rn = int(featurePointRight.at(i).x);

		circle(bothview, Point(rm, rn), radius, Scalar(0, 255, 255), 1, 4, 0);
		line(bothview, Point(rm - radius - 2, rn), Point(rm + radius + 2, rn), Scalar(0, 255, 255), 1, 8, 0);
		line(bothview, Point(rm, rn - radius - 2), Point(rm, rn + radius + 2), Scalar(0, 255, 255), 1, 8, 0);
		//connect
		line(bothview, Point(lm, ln), Point(rm, rn), Scalar(a, b, c), 1, 8, 0);
	}

	return bothview;
}

Mat CMatch::NCCMatchingImg(string path_left, string path_right, params_ncc par, vector<Point3f> featurePointLeft)
{
	Mat imageLeft, imageLeftRGB = imread(path_left, IMREAD_COLOR);
	Mat imageRight, imageRightRGB = imread(path_right, IMREAD_COLOR);
	if (imageLeftRGB.empty())
	{
		cout << "Fail to read the left image:" << path_left << endl;
	}
	if (imageRightRGB.empty())
	{
		cout << "Fail to read the rihgt image:" << path_right << endl;
	}
	cvtColor(imageLeftRGB, imageLeft, COLOR_BGR2GRAY);
	cvtColor(imageRightRGB, imageRight, COLOR_BGR2GRAY);

	int matchsize = par.matchsize;//match window size
	int half_matchsize = matchsize / 2;

	vector<Point3f> featurePointRight;//match result on right page

	float lowst_door = par.lowst_door;
	int dist_height = par.dist_height;
	int dist_width = par.dist_width;

	//pre-processing, delet point that does not meet the specifications
	for (size_t i = 0; i < featurePointLeft.size(); i++)
	{
		if ((featurePointLeft.at(i).y + dist_width < imageLeft.cols) || (imageLeft.cols - featurePointLeft.at(i).y < 5))
		{
			featurePointLeft.erase(featurePointLeft.begin() + i);
			i--;
			continue;
		}
		if ((featurePointLeft.at(i).x - dist_height <5) || (imageLeft.rows - featurePointLeft.at(i).x < 5))//
		{
			featurePointLeft.erase(featurePointLeft.begin() + i);
			i--;
			continue;
		}
	}

	//create left window
	Mat matchLeftWindow;
	matchLeftWindow.create(matchsize, matchsize, CV_32FC1);
	for (size_t i = 0; i < featurePointLeft.size(); i++)
	{
		float aveLImg = 0;
		for (int m = 0; m < matchsize; m++)
		{
			for (int n = 0; n < matchsize; n++)
			{
				aveLImg += imageLeft.at<uchar>(featurePointLeft.at(i).x - half_matchsize + m, featurePointLeft.at(i).y - half_matchsize + n);
				matchLeftWindow.at<float>(m, n) = imageLeft.at<uchar>(featurePointLeft.at(i).x - half_matchsize + m, featurePointLeft.at(i).y - half_matchsize + n);
			}
		}
		aveLImg = aveLImg / (matchsize* matchsize);//left
		//norm for left
		for (int m = 0; m < matchsize; m++)
		{
			for (int n = 0; n < matchsize; n++)
			{
				matchLeftWindow.at<float>(m, n) = matchLeftWindow.at<float>(m, n) - aveLImg;
			}
		}
		//calculate right window
		//predict the position on the right window 
		int halflengthsize = par.PreSearchRadius;
		vector <Point3f> tempfeatureRightPoint;
		//delete the outranges
		for (int ii = -halflengthsize; ii <= halflengthsize; ii++)
		{
			for (int jj = -halflengthsize; jj <= halflengthsize; jj++)
			{
				if ((featurePointLeft.at(i).x - dist_height < (halflengthsize + 5)) || (imageRight.rows - featurePointLeft.at(i).x) < (halflengthsize + 5)
					|| (featurePointLeft.at(i).y + dist_width - imageLeft.cols) < (halflengthsize + 5))
				{
					Point3f temphalflengthsize;
					temphalflengthsize.x = 0;
					temphalflengthsize.y = 0;
					temphalflengthsize.z = 0;
					tempfeatureRightPoint.push_back(temphalflengthsize);
				}
				else
				{
					Point3f temphalflengthsize;
					int x = featurePointLeft.at(i).x - dist_height + ii - half_matchsize;//
					int y = featurePointLeft.at(i).y + dist_width - imageLeft.cols + jj - half_matchsize;
					float  coffee = Get_coefficient(matchLeftWindow, imageRight, x, y);
					temphalflengthsize.x = featurePointLeft.at(i).x - dist_height + ii;//
					temphalflengthsize.y = featurePointLeft.at(i).y + dist_width - imageLeft.cols + jj;
					temphalflengthsize.z = coffee;
					tempfeatureRightPoint.push_back(temphalflengthsize);
				}

			}
		}
		Vector_Sort(tempfeatureRightPoint);
		//compare with threshold
		if (tempfeatureRightPoint.at(0).z > lowst_door&&tempfeatureRightPoint.at(0).z < 1)
		{
			Point3f tempr;
			tempr.x = tempfeatureRightPoint.at(0).x;
			tempr.y = tempfeatureRightPoint.at(0).y;
			tempr.z = tempfeatureRightPoint.at(0).z;
			featurePointRight.push_back(tempr);
		}
		else
		{
			featurePointLeft.erase(featurePointLeft.begin() + i);
			i--;
			continue;
		}
	}
	//show
	Mat result = View(imageLeftRGB, imageRightRGB, featurePointLeft, featurePointRight);

	//Output pos
	ofstream outputfile;
	outputfile.open("FeturePointMatch_Output.txt");
	if (outputfile.is_open()) {
		outputfile << "ID \t Left_x \t Left_y \t NCCRight_x \t NCCRight_y \t Coef" << endl;
		for (size_t i = 0; i < featurePointRight.size(); i++)
		{
			outputfile << i << "," << featurePointLeft.at(i).x << ", " << featurePointLeft.at(i).y << ", " <<
				featurePointRight.at(i).x << ", " << featurePointRight.at(i).y << ", " << featurePointRight.at(i).z << "，" << endl;
		}
	}
	outputfile.close();
	cout << "---------------------------------------" << endl;
	cout << "Match points has saved in\"FeturePointMatch_Output.txt\"" << endl;

	return result;
}

Mat CMatch::LSMatchingImg(string path_left, string path_right, params_ncc par, vector<Point3f> featurePointLeft)
{
	Mat imageLeft, imageLeftRGB = imread(path_left, IMREAD_COLOR);
	Mat imageRight, imageRightRGB = imread(path_right, IMREAD_COLOR);
	if (imageLeftRGB.empty())
	{
		cout << "Fail to read the left image:" << path_left << endl;
	}
	if (imageRightRGB.empty())
	{
		cout << "Fail to read the rihgt image:" << path_right << endl;
	}
	cvtColor(imageLeftRGB, imageLeft, COLOR_BGR2GRAY);
	cvtColor(imageRightRGB, imageRight, COLOR_BGR2GRAY);

	int matchsize = par.matchsize;//match window size
	int half_matchsize = matchsize / 2;

	vector<Point3f> featurePointRight;//match result on right page

	float lowst_door = par.lowst_door;
	int dist_height = par.dist_height;
	int dist_width = par.dist_width;

	//pre-processing, delet point that does not meet the specifications
	for (size_t i = 0; i < featurePointLeft.size(); i++)
	{
		if ((featurePointLeft.at(i).y + dist_width < imageLeft.cols) || (imageLeft.cols - featurePointLeft.at(i).y < 5))
		{
			featurePointLeft.erase(featurePointLeft.begin() + i);
			i--;
			continue;
		}
		if ((featurePointLeft.at(i).x - dist_height < 5) || (imageLeft.rows - featurePointLeft.at(i).x < 5))//
		{
			featurePointLeft.erase(featurePointLeft.begin() + i);
			i--;
			continue;
		}
	}

	//create left window
	Mat matchLeftWindow;
	matchLeftWindow.create(matchsize, matchsize, CV_32FC1);
	for (size_t i = 0; i < featurePointLeft.size(); i++)
	{
		float aveLImg = 0;
		for (int m = 0; m < matchsize; m++)
		{
			for (int n = 0; n < matchsize; n++)
			{
				aveLImg += imageLeft.at<uchar>(featurePointLeft.at(i).x - half_matchsize + m, featurePointLeft.at(i).y - half_matchsize + n);
				matchLeftWindow.at<float>(m, n) = imageLeft.at<uchar>(featurePointLeft.at(i).x - half_matchsize + m, featurePointLeft.at(i).y - half_matchsize + n);
			}
		}
		aveLImg = aveLImg / (matchsize * matchsize);//left
		//norm for left
		for (int m = 0; m < matchsize; m++)
		{
			for (int n = 0; n < matchsize; n++)
			{
				matchLeftWindow.at<float>(m, n) = matchLeftWindow.at<float>(m, n) - aveLImg;
			}
		}
		//calculate right window
		//predict the position on the right window 
		int halflengthsize = par.PreSearchRadius;
		vector <Point3f> tempfeatureRightPoint;
		//delete the outranges
		for (int ii = -halflengthsize; ii <= halflengthsize; ii++)
		{
			for (int jj = -halflengthsize; jj <= halflengthsize; jj++)
			{
				if ((featurePointLeft.at(i).x - dist_height < (halflengthsize + 5)) || (imageRight.rows - featurePointLeft.at(i).x) < (halflengthsize + 5)
					|| (featurePointLeft.at(i).y + dist_width - imageLeft.cols) < (halflengthsize + 5))
				{
					Point3f temphalflengthsize;
					temphalflengthsize.x = 0;
					temphalflengthsize.y = 0;
					temphalflengthsize.z = 0;
					tempfeatureRightPoint.push_back(temphalflengthsize);
				}
				else
				{
					Point3f temphalflengthsize;
					int x = featurePointLeft.at(i).x - dist_height + ii - half_matchsize;//
					int y = featurePointLeft.at(i).y + dist_width - imageLeft.cols + jj - half_matchsize;
					float  coffee = Get_coefficient(matchLeftWindow, imageRight, x, y);
					temphalflengthsize.x = featurePointLeft.at(i).x - dist_height + ii;//
					temphalflengthsize.y = featurePointLeft.at(i).y + dist_width - imageLeft.cols + jj;
					temphalflengthsize.z = coffee;
					tempfeatureRightPoint.push_back(temphalflengthsize);
				}

			}
		}
		Vector_Sort(tempfeatureRightPoint);
		//compare with threshold
		if (tempfeatureRightPoint.at(0).z > lowst_door && tempfeatureRightPoint.at(0).z < 1)
		{
			Point3f tempr;
			tempr.x = tempfeatureRightPoint.at(0).x;
			tempr.y = tempfeatureRightPoint.at(0).y;
			tempr.z = tempfeatureRightPoint.at(0).z;
			featurePointRight.push_back(tempr);
		}
		else
		{
			featurePointLeft.erase(featurePointLeft.begin() + i);
			i--;
			continue;
		}
	}

	//start the lst matching
	vector <Point3f> featureRightPoint_LSM;

	//Initial
	Mat P0 = cv::Mat::eye(2 * featurePointLeft.size(), 2 * featurePointLeft.size(), CV_32F)/*P*/,
		L0 = cv::Mat::zeros(2 * featurePointLeft.size(), 1, CV_32F)/*L*/,
		A0 = cv::Mat::zeros(2 * featurePointLeft.size(), 6, CV_32F)/*A*/;

	for (int i = 0; i < featurePointLeft.size(); i++)
	{
		float x1 = featurePointLeft.at(i).x;
		float y1 = featurePointLeft.at(i).y;
		float x2 = featurePointRight.at(i).x;
		float y2 = featurePointRight.at(i).y;
		float coef = featurePointRight.at(i).z;//NCC result as coefficient
		P0.at<float>(2 * i, 2 * i) = coef;
		P0.at<float>(2 * i + 1, 2 * i + 1) = coef;
		L0.at<float>(2 * i, 0) = x2;
		L0.at<float>(2 * i + 1, 0) = y2;
		A0.at<float>(2 * i, 0) = 1;
		A0.at<float>(2 * i, 1) = x1;
		A0.at<float>(2 * i, 2) = y1;
		A0.at<float>(2 * i + 1, 3) = 1;
		A0.at<float>(2 * i + 1, 4) = x1;
		A0.at<float>(2 * i + 1, 5) = y1;
	}
	Mat Nbb = A0.t()*P0*A0, U = A0.t()*P0*L0;
	Mat R0 = Nbb.inv()*U;

	//iterative solution
	for (int i = 0; i < featurePointLeft.size(); i++)
	{
		//（x1_0,y1_0）（x2_0,y2_0）
		float x1 = featurePointLeft.at(i).x;
		float y1 = featurePointLeft.at(i).y;
		float x2 = featurePointRight.at(i).x;
		float y2 = featurePointRight.at(i).y;
		//geometric parameters Initial
		float a0 = R0.at<float>(0, 0);
		float a1 = R0.at<float>(1, 0);
		float a2 = R0.at<float>(2, 0);
		float b0 = R0.at<float>(3, 0);
		float b1 = R0.at<float>(4, 0);
		float b2 = R0.at<float>(5, 0);
		//Radiation parameters Initial
		float h0 = 0, h1 = 1;

		//Set the iteration end condition:the ncc latter < before
		float Coef_former = 0, Coef_Latter = 0;
		float xs = 0, ys = 0;

		while (Coef_former <= Coef_Latter)
		{
			Coef_former = Coef_Latter;
			Mat C = Mat::zeros(matchsize*matchsize, 8, CV_32F);//C，matchsize as the left
			Mat L = Mat::zeros(matchsize*matchsize, 1, CV_32F);//L
			Mat P = Mat::eye(matchsize*matchsize, matchsize*matchsize, CV_32F);//P
			float sumgxSquare = 0, sumgySquare = 0, sumXgxSquare = 0, sumYgySquare = 0;
			int dimension = 0;
			float sumLImg = 0, sumLImgSquare = 0, sumRImg = 0, sumRImgSquare = 0, sumLR = 0;

			for (int m = x1 - half_matchsize; m <= x1 + half_matchsize; m++)
			{
				for (int n = y1 - half_matchsize; n <= y1 + half_matchsize; n++)
				{
					float x2 = a0 + a1 * m + a2 * n;
					float y2 = b0 + b1 * m + b2 * n;
					int I = floor(x2); int J = floor(y2);//Maximum int < n
					if (I <= 1 || I >= imageRight.rows - 1 || J <= 1 || J >= imageRight.cols - 1)
					{
						I = 2; J = 2; P.at<float>((m - (y1 - 5) - 1)*(2 * 4 + 1) + n - (x1 - 5), (m - (y1 - 5) - 1)*(2 * 4 + 1) + n - (x1 - 5)) = 0;
					}

					//bilinear interpolation resampling
					float linerGray = (J + 1 - y2)*((I + 1 - x2)*imageRight.at<uchar>(I, J) + (x2 - I)*imageRight.at<uchar>(I + 1, J))
						+ (y2 - J)*((I + 1 - x2)*imageRight.at<uchar>(I, J + 1) + (x2 - I)*imageRight.at<uchar>(I + 1, J + 1));
					//radiometric correction
					float radioGray = h0 + h1 * linerGray;//f(x)=h_0+h_1*g(x)

					sumRImg += radioGray;
					sumRImgSquare += radioGray * radioGray;

					//coefficient matrix
					float gy = 0.5*(imageRight.at<uchar>(I, J + 1) - imageRight.at<uchar>(I, J - 1));
					float gx = 0.5*(imageRight.at<uchar>(I + 1, J) - imageRight.at<uchar>(I - 1, J));
					C.at<float>(dimension, 0) = 1; C.at<float>(dimension, 1) = linerGray;
					C.at<float>(dimension, 2) = gx; C.at<float>(dimension, 3) = x2 * gx;
					C.at<float>(dimension, 4) = y2 * gx; C.at<float>(dimension, 5) = gy;
					C.at<float>(dimension, 6) = x2 * gy; C.at<float>(dimension, 7) = y2 * gy;
					//constant matrix
					L.at<float>(dimension, 0) = imageLeft.at<uchar>(m, n) - radioGray;
					dimension = dimension + 1;
					//Weighted average for the left
					float gyLeft = 0.5*(imageLeft.at<uchar>(m, n + 1) - imageLeft.at<uchar>(m, n - 1));
					float gxLeft = 0.5*(imageLeft.at<uchar>(m + 1, n) - imageLeft.at<uchar>(m - 1, n));
					sumgxSquare += gxLeft * gxLeft;
					sumgySquare += gyLeft * gyLeft;
					sumXgxSquare += m * gxLeft*gxLeft;
					sumYgySquare += n * gyLeft*gyLeft;
					//Sum(&Sum of squares) of left
					sumLImg += imageLeft.at<uchar>(m, n);
					sumLImgSquare += imageLeft.at<uchar>(m, n)*imageLeft.at<uchar>(m, n);
					sumLR += radioGray * imageLeft.at<uchar>(m, n);
				}
			}
			//Get Coef
			float coefficent1 = sumLR - sumLImg * sumRImg / (matchsize*matchsize);
			float coefficent2 = sumLImgSquare - sumLImg * sumLImg / (matchsize*matchsize);
			float coefficent3 = sumRImgSquare - sumRImg * sumRImg / (matchsize*matchsize);
			Coef_Latter = coefficent1 / sqrt(coefficent2*coefficent3);
			//Get a0,a1,a2,b0,b1,b2 and h0,h1
			Mat Nb = C.t()*P*C, Ub = C.t()*P*L;
			Mat parameter = Nb.inv()*Ub;
			float dh0 = parameter.at<float>(0, 0); float dh1 = parameter.at<float>(1, 0);
			float da0 = parameter.at<float>(2, 0); float da1 = parameter.at<float>(3, 0); float da2 = parameter.at<float>(4, 0);
			float db0 = parameter.at<float>(5, 0); float db1 = parameter.at<float>(6, 0); float db2 = parameter.at<float>(7, 0);

			a0 = a0 + da0 + a0 * da1 + b0 * da2;
			a1 = a1 + a1 * da1 + b1 * da2;
			a2 = a2 + a2 * da1 + b2 * da2;
			b0 = b0 + db0 + a0 * db1 + b0 * db2;
			b1 = b1 + a1 * db1 + b1 * db2;
			b2 = b2 + a2 * db1 + b2 * db2;
			h0 = h0 + dh0 + h0 * dh1;
			h1 = h1 + h1 * dh1;

			float xt = sumXgxSquare / sumgxSquare;
			float yt = sumYgySquare / sumgySquare;
			xs = a0 + a1 * xt + a2 * yt;
			ys = b0 + b1 * xt + b2 * yt;
		}
		Point3f tempPoint;
		tempPoint.x = xs;
		tempPoint.y = ys;
		tempPoint.z = Coef_Latter;
		featureRightPoint_LSM.push_back(tempPoint);
	}
	//show
	Mat result = View(imageLeftRGB, imageRightRGB, featurePointLeft, featureRightPoint_LSM);

	//Output pos
	ofstream outputfile;
	outputfile.open("FeturePointMatch_Output.txt");
	if (outputfile.is_open()) {
		outputfile << "Left_x, Left_y, NCCRight_x, NCCRight_y, Coef, LSMRight_x, LSMRight_y, Coef" << endl;
		for (size_t i = 0; i < featurePointRight.size(); i++)
		{
			outputfile << featurePointLeft.at(i).x << ", " << featurePointLeft.at(i).y << ", "  <<
				featurePointRight.at(i).x << ", " << featurePointRight.at(i).y << ", " << featurePointRight.at(i).z << "，" <<
				featureRightPoint_LSM.at(i).x << ", " << featureRightPoint_LSM.at(i).y << ", " << featureRightPoint_LSM.at(i).z << endl;
		}
	}
	outputfile.close();
	cout << "---------------------------------------" << endl;

	return result;
}

void CMatch::Get_matchresult(int kind, string path_left, string path_right, params_ncc par, vector<Point3f> featurePointLeft) {

	//start the clock
	clock_t start = clock();
	clock_t finish;
	Mat result;

	if (kind == 1) {
		//NCC(pathLeft, pathRight,par(matchsize,PreSearchRadius,dist_width,dist_height,lowst_door), featurePointLeft)
		result = NCCMatchingImg(path_left, path_right, par, featurePointLeft);
		imshow("Result of NCC", result);
		imwrite("Result_NCC.jpg", result);
		finish = clock();
		waitKey(10);
	}
	else if (kind == 2) {
		//LSM(pathLeft, pathRight,par(matchsize,PreSearchRadius,dist_width,dist_height,lowst_door), featurePointLeft)
		result = LSMatchingImg(path_left, path_right, par, featurePointLeft);
		imshow("Result of LS", result);
		imwrite("Result_LSM.jpg", result);
		finish = clock();
		waitKey(10);
	}
	cout << "Matching time：" << (double)(finish - start) / CLOCKS_PER_SEC << "s" << endl;
	cout << "Match points has saved in\"FeturePointMatch_Output.txt\"" << endl;
	cout << "__________________________________________________________" << endl;
	cout << "Matching processing finished!" << endl;
	return;
}



// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
