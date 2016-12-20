#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>
#include <queue>
#include <intrin.h>
using namespace std;
using namespace cv;

class deHaze
{

public:
	int AR, AG, AB;
	Mat srcImg;
	Mat darkChannelImg;
	Mat roughtT;
	Mat findedT;
	Mat deHazeImg;
	deHaze();
	bool load(Mat img);
	void guidedFilter1Channel(Mat src, Mat guideImg, Mat& dst, int filterWindowSize, float eps = 0.001);
	void guidedFilter3Channel(Mat src, Mat floatB, Mat floatG,Mat floatR, Mat& dst, int filterWindowSize ,float eps = 0.001);
	Mat hazeRemoval(int darkWindowSize, int filterWindowSize);
	Mat practicalHazeRemoval(int darkWindowSize, int filterWindowSize);
	Mat fastHazeRemoval(int darkWindowSize, int filterWindowSize);
	Mat superFastHazeRemoval(int darkWindowSize, int filterWindowSize);
	~deHaze();
private:
	inline int scaleRGB(int a);
	void darkChannelProcess(Mat src, Mat& dst, int darkWindowSize);
	void streamMinFilter(Mat src, Mat& dst,int darkWindowSize);
	void picMinFilter(Mat src, Mat& dst, int darkWindowSize);
	void getGlobalAtmosphericLight(Mat darkChannelSrc,int& Ar, int& Ag, int& Ab);
	void getRoughT(Mat normB, Mat normG, Mat normR, Mat& dst, int Ar, int Ag, int Ab);
	void fastRoughtT(Mat src, Mat &dst, int Ar, int Ag, int Ab);
	void imgRecover(Mat src, Mat tFined, Mat& dst, int globalA);
};


inline int deHaze::scaleRGB(int a)
{
	if (a < 0)
	{
		return 0;
	}
	if (a > 255)
	{
		return 255;
	}
	return a;
}