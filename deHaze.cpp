#include "deHaze.h"



bool deHaze::load(Mat img)
{
	if (img.empty())
	{
		cerr << "empty image" << endl;
	}
	if ( CV_8UC3 != img.type() )
	{
		cerr << "wrong image type, should be CV_8UC3" << endl;
		return false;
	}
	this->srcImg = img;
	return true;
}

void deHaze::guidedFilter1Channel(Mat src, Mat floatGuideImg, Mat& dst, int filterWindowSize, float eps)
{
	Mat meanI, meanP, meanIP, covIP, meanII, varI, a, b;
	Size blurSize = Size(filterWindowSize, filterWindowSize);

	cv::blur(floatGuideImg, meanI, blurSize);
	cv::blur(src, meanP, blurSize);
	cv::blur(src.mul(floatGuideImg), meanIP, blurSize);
	covIP = meanIP - meanI.mul(meanP);
	cv::blur(floatGuideImg.mul(floatGuideImg), meanII, blurSize);
	varI = meanII - meanI.mul(meanI);
	a = covIP / (varI + eps);
	b = meanP - a.mul(meanI);
	cv::blur(a, a, blurSize);
	cv::blur(b, b, blurSize);
	dst = a.mul(floatGuideImg) + b;
}

void deHaze::guidedFilter3Channel(Mat src, Mat floatB, Mat floatG, Mat floatR, Mat& dst, int filterWindowSize, float eps)
{
	int i, j, Irow = src.rows, Icol = src.cols;
	Size blurSize = Size(filterWindowSize, filterWindowSize);
	Mat meanP, meanB, meanG, meanR, meanPB, meanPG, meanPR;
	Mat varRR, varRG, varRB, varGG, varGB, varBB;
	Mat covPB, covPG, covPR;
	vector<Mat> aChannels;
	Mat sigma(3, 3, CV_32FC1), covIp(1, 3, CV_32FC1), RGB(1, 3, CV_32FC1), a(Irow, Icol, CV_32FC3), b(Irow, Icol, CV_32FC1);
	
	cv::blur(src, meanP, blurSize);
#pragma omp parallel sections
	{
#pragma omp section
		{
			//process B channel
			cv::blur(floatB, meanB, blurSize);
			cv::blur(floatB.mul(src), meanPB, blurSize);
			covPB = meanPB - meanB.mul(meanP);
			cv::blur(floatB.mul(floatB), varBB, blurSize);
			varBB = varBB - meanB.mul(meanB);
		}
#pragma omp section
		{
			//process G channel
			cv::blur(floatG, meanG, blurSize);
			cv::blur(floatG.mul(src), meanPG, blurSize);
			covPG = meanPG - meanG.mul(meanP);
			cv::blur(floatG.mul(floatG), varGG, blurSize);
			varGG = varGG - meanG.mul(meanG);
		}
#pragma omp section
		{
			//process R channel
			cv::blur(floatR, meanR, blurSize);
			cv::blur(floatR.mul(src), meanPR, blurSize);
			covPR = meanPR - meanR.mul(meanP);
			cv::blur(floatR.mul(floatR), varRR, blurSize);
			varRR = varRR - meanR.mul(meanR);
		}
#pragma omp section
		{
			//rest variance
			cv::blur(floatR.mul(floatG), varRG, blurSize);
			varRG = varRG - meanR.mul(meanG);
			cv::blur(floatR.mul(floatB), varRB, blurSize);
			varRB = varRB - meanR.mul(meanB);
			cv::blur(floatG.mul(floatB), varGB, blurSize);
			varGB = varGB - meanG.mul(meanB);
		}
	}

	//calculate a
	const float* varRRData, *varRGData, *varRBData, *varGGData, *varGBData, *varBBData, *covPRData, *covPGData, *covPBData;
	float* aData, *covIPData, *sigmaData;
	for (i = 0; i < Irow; i++)
	{
		varRRData = varRR.ptr<float>(i);
		varRGData = varRG.ptr<float>(i);
		varRBData = varRB.ptr<float>(i);
		varGGData = varGG.ptr<float>(i);
		varGBData = varGB.ptr<float>(i);
		varBBData = varBB.ptr<float>(i);
		covPRData = covPR.ptr<float>(i);
		covPGData = covPG.ptr<float>(i);
		covPBData = covPB.ptr<float>(i);
		aData = a.ptr<float>(i);
		for (j = 0; j < Icol; j++)
		{
			//set covariance of rgb
			covIPData = covIp.ptr<float>(0);
			*covIPData++ = *covPRData++;
			*covIPData++ = *covPGData++;
			*covIPData = *covPBData++;
			//set the sigma mat
			sigmaData = sigma.ptr<float>(0);
			*sigmaData++ = *varRRData + eps;
			*sigmaData++ = *varRGData;
			*sigmaData = *varRBData;
			sigmaData = sigma.ptr<float>(1);
			*sigmaData++ = *varRGData;
			*sigmaData++ = *varGGData + eps;
			*sigmaData = *varGBData;
			sigmaData = sigma.ptr<float>(2);
			*sigmaData++ = *varRBData;
			*sigmaData++ = *varGBData;
			*sigmaData = *varBBData + eps;
			varRRData++;
			varRGData++;
			varGGData++;
			varRBData++;
			varGBData++;
			varBBData++;
			//calculate the rgb
			RGB = covIp * sigma.inv();
			*aData++ = RGB.ptr <float>(0)[0];
			*aData++ = RGB.ptr <float>(0)[1];
			*aData++ = RGB.ptr <float>(0)[2];
		}
	}
	//calculate b
	cv::split(a, aChannels);
	b = meanP - aChannels[0].mul(meanR) - aChannels[1].mul(meanG) - aChannels[2].mul(meanB);
	//caluculate q
	cv::blur(aChannels[0], aChannels[0], blurSize);
	cv::blur(aChannels[1], aChannels[1], blurSize);
	cv::blur(aChannels[2], aChannels[2], blurSize);
	cv::blur(b, b, blurSize);
	dst = aChannels[0].mul(floatR) + aChannels[1].mul(floatG) + aChannels[2].mul(floatB) + b;
}

Mat deHaze::hazeRemoval(int darkWindowSize, int filterWindowSize)
{
	//dark channel to get estimation t
	// 1/255 = 0.0038215686
	Mat NormB, NormG, NormR, floatB, floatG, floatR;
	vector<Mat> channels;
	cv::split(this->srcImg, channels);
#pragma omp parallel sections
	{
#pragma omp section
		{
			deHaze::darkChannelProcess(this->srcImg, this->darkChannelImg, darkWindowSize);
			deHaze::getGlobalAtmosphericLight(this->darkChannelImg, this->AR, this->AG, this->AB);
		}
		
#pragma omp section
		{
			deHaze::picMinFilter(channels[0], NormB, darkWindowSize);
			channels[0].convertTo(floatB, CV_32FC1, 0.0038215686);
		}
		
#pragma omp section
		{
			deHaze::picMinFilter(channels[1], NormG, darkWindowSize);
			channels[1].convertTo(floatG, CV_32FC1, 0.0038215686);
		}
#pragma omp section
		{
			deHaze::picMinFilter(channels[2], NormR, darkWindowSize);
			channels[2].convertTo(floatR, CV_32FC1, 0.0038215686);
		}
	}

	deHaze::getRoughT(NormB, NormG, NormR, this->roughtT, this->AR, this->AG, this->AB);
	//guide filter
	deHaze::guidedFilter3Channel(this->roughtT, floatB, floatG, floatR, this->findedT, filterWindowSize);

	int globalA = MAX(this->AR, this->AG);
	globalA = MAX(globalA, this->AB);

	deHaze::imgRecover(this->srcImg, this->findedT, this->deHazeImg, globalA);

	return this->deHazeImg;
}

Mat deHaze::practicalHazeRemoval(int darkWindowSize, int filterWindowSize)
{
	Mat NormB, NormG, NormR, floatB, floatG, floatR, floatGuideGray;
	vector<Mat> channels;
	cv::split(this->srcImg, channels);
#pragma omp parallel sections
	{
#pragma omp section
		{
			deHaze::darkChannelProcess(this->srcImg, this->darkChannelImg, darkWindowSize);
			deHaze::getGlobalAtmosphericLight(this->darkChannelImg, this->AR, this->AG, this->AB);
		}

#pragma omp section
		{
			deHaze::picMinFilter(channels[0], NormB, darkWindowSize);
			channels[0].convertTo(floatB, CV_32FC1, 0.0038215686);
		}

#pragma omp section
		{
			deHaze::picMinFilter(channels[1], NormG, darkWindowSize);
			channels[1].convertTo(floatG, CV_32FC1, 0.0038215686);
		}
#pragma omp section
		{
			deHaze::picMinFilter(channels[2], NormR, darkWindowSize);
			channels[2].convertTo(floatR, CV_32FC1, 0.0038215686);
		}
	}
	deHaze::getRoughT(NormB, NormG, NormR, this->roughtT, this->AR, this->AG, this->AB);

	this->srcImg.convertTo(floatGuideGray, CV_32FC1, 0.0038215686);
	cvtColor(floatGuideGray, floatGuideGray, COLOR_BGR2GRAY);
	deHaze::guidedFilter1Channel(this->roughtT, floatGuideGray, this->findedT, filterWindowSize);

	int globalA = MAX(this->AR, this->AG);
	globalA = MAX(globalA, this->AB);

	deHaze::imgRecover(this->srcImg, this->findedT, this->deHazeImg, globalA);
	return this->deHazeImg;
	return Mat();
}

Mat deHaze::fastHazeRemoval(int darkWindowSize, int filterWindowSize)
{
	Mat floatGuideGray;
	deHaze::darkChannelProcess(this->srcImg, this->darkChannelImg, darkWindowSize);
	deHaze::getGlobalAtmosphericLight(this->darkChannelImg, this->AR, this->AG, this->AB);

	this->srcImg.convertTo(floatGuideGray, CV_32FC1, 0.0038215686);
	cvtColor(floatGuideGray, floatGuideGray, COLOR_BGR2GRAY);

	deHaze::fastRoughtT(this->darkChannelImg, this->roughtT, this->AR, this->AG, this->AB);

	deHaze::guidedFilter1Channel(this->roughtT, floatGuideGray, this->findedT, filterWindowSize);

	int globalA = MAX(this->AR, this->AG);
	globalA = MAX(globalA, this->AB);

	deHaze::imgRecover(this->srcImg, this->findedT, this->deHazeImg, globalA);
	return this->deHazeImg;
}

Mat deHaze::superFastHazeRemoval(int darkWindowSize, int filterWindowSize)
{
	Mat floatGuideGray;
	deHaze::darkChannelProcess(this->srcImg, this->darkChannelImg, darkWindowSize);
	deHaze::getGlobalAtmosphericLight(this->darkChannelImg, this->AR, this->AG, this->AB);
	this->srcImg.convertTo(floatGuideGray, CV_32FC1, 0.0038215686);
	cvtColor(floatGuideGray, floatGuideGray, COLOR_BGR2GRAY);
	deHaze::fastRoughtT(this->darkChannelImg, this->roughtT, this->AR, this->AG, this->AB);

	int globalA = MAX(this->AR, this->AG);
	globalA = MAX(globalA, this->AB);

	deHaze::imgRecover(this->srcImg, this->roughtT, this->deHazeImg, globalA);
	return this->deHazeImg;
}

deHaze::deHaze()
{
}

deHaze::~deHaze()
{
}

/*int deHaze::scaleRGB(int a)
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
}*/

void deHaze::darkChannelProcess(Mat src, Mat& dst, int darkWindowSize)
{
	//get the minimum img
	int i, j, Irow = src.rows, Icol = src.cols;
	Mat minImg = Mat::zeros(Irow, Icol, CV_8UC1);
	if (src.isContinuous() && minImg.isContinuous())
	{
		Icol = Irow * Icol;
		Irow = 1;
	}

	const uchar* inData;
	uchar* outData;
	for ( i = 0; i < Irow; i++)
	{
		inData = src.ptr<uchar>(i);
		outData = minImg.ptr<uchar>(i);
		for ( j = 0; j < Icol; j++)
		{
			*outData = MIN(*inData, *(inData + 1));
			inData += 2;
			*outData = MIN(*outData, *inData);
			inData++;
			outData++;
		}
	}
	//pic min filter	
	deHaze::picMinFilter(minImg, dst, darkWindowSize);
}

void deHaze::streamMinFilter(Mat src, Mat& dst, int darkWindowSize)
{
	darkWindowSize = darkWindowSize / 2 + 1;
	int i, j, k, Irow = src.rows, Icol = src.cols + 1, minimum;
	deque<int> L;
	dst = 255 * Mat::ones(Irow, Icol, CV_8UC1);

	const uchar* inData;
	uchar* outData;
	for ( i = 0; i < Irow; i++)
	{
		inData = src.ptr<uchar>(i);
		outData = dst.ptr<uchar>(i);
		L.clear();
		L.push_back(0);
		for ( j = 1; j < Icol; j++)
		{
			outData++;
			if (j >= darkWindowSize)
			{
				minimum = *(inData + L.front());
				for ( k = 1; k < darkWindowSize + 1; k++)
				{
					if (minimum < *(outData - k))
					{
						*(outData - k) = minimum;
					}
				}
			}
			if (*(inData + j) < *(inData + j - 1))
			{
				L.pop_back();
				while (!L.empty() && (*(inData + j) < *(inData + L.back())))
				{
					L.pop_back();
				}
			}
			L.push_back(j);
			if (j == (darkWindowSize + L.front()))
			{
				L.pop_front();
			}
		}
	}
	dst = dst(Range(0, Irow), Range(0, Icol - 1));
}

void deHaze::picMinFilter(Mat src, Mat & dst, int darkWindowSize)
{
	Mat tmp;
	deHaze::streamMinFilter(src, tmp, darkWindowSize);
	deHaze::streamMinFilter(tmp.t(), dst, darkWindowSize);
	dst = dst.t();
}

void deHaze::getGlobalAtmosphericLight(Mat darkChannelSrc, int & Ar, int & Ag, int & Ab)
{
	//pick maximum of top 0.1% darkchannel
	int sum = 0, AThreshold, i, j, Irow = darkChannelSrc.rows, Icol = darkChannelSrc.cols, BucketSort[256] = { 0 };
	Ar = 0;
	Ag = 0;
	Ab = 0;
	if (darkChannelSrc.isContinuous() && this->srcImg.isContinuous())
	{
		Icol = Irow * Icol;
		Irow = 1;
	}

	const uchar* inData;
	for (i = 0; i < Irow; i++)
	{
		inData = darkChannelSrc.ptr<uchar>(i);
		for (j = 0; j < Icol; j++)
		{
			BucketSort[*inData]++;
			inData++;
		}
	}
	AThreshold = 0.001 * Irow * Icol;
	for (i = 255; i >= 0; i++)
	{
		sum += BucketSort[i];
		if (sum >= AThreshold)
		{
			AThreshold = i;
			break;
		}
	}

	const uchar* indexData;
	for (i = 0; i < Irow; i++)
	{
		inData = this->srcImg.ptr<uchar>(i);
		indexData = darkChannelSrc.ptr<uchar>(i);
		for (j = 0; j < Icol; j++)
		{
			if (*indexData < AThreshold)
			{
				if (*inData > Ab)
				{
					Ab = *inData;
				}
				inData++;
				if (*inData > Ag)
				{
					Ag = *inData;
				}
				inData++;
				if (*inData > Ar)
				{
					Ar = *inData;
				}
				inData++;
			}
		}
	}
}

void deHaze::getRoughT(Mat normB, Mat normG, Mat normR, Mat & dst, int Ar, int Ag, int Ab)
{
	int i, j, Irow = normB.rows, Icol = normB.cols;
	dst = Mat::zeros(Irow, Icol, CV_32FC1);
	float Arr = (float)(Ar);
	float Agg = (float)(Ag);
	float Abb = (float)(Ab);
	float tmpT;
	if (normB.isContinuous() && normG.isContinuous() && normR.isContinuous() && dst.isContinuous())
	{
		Icol = Irow * Icol;
		Irow = 1;
	}

	const uchar* bData, *gData, *rData;
	float* tData;
	for ( i = 0; i < Irow; i++)
	{
		bData = normB.ptr<uchar>(i);
		gData = normG.ptr<uchar>(i);
		rData = normR.ptr<uchar>(i);
		tData = dst.ptr<float>(i);
		for ( j = 0; j < Icol; j++)
		{
			tmpT = MIN(*bData / Abb, *gData / Agg);
			tmpT = MIN(*rData / Arr, tmpT);
			*tData = MAX(0, 1 - 0.95 * tmpT);
			bData++;
			gData++;
			rData++;
			tData++;
		}
	}
}

void deHaze::fastRoughtT(Mat src, Mat & dst, int Ar, int Ag, int Ab)
{
	int i, j, k, Irow = src.rows, Icol = src.cols;
	dst = Mat::zeros(Irow, Icol, CV_32FC1);
	float Ac = (Ar + Ag + Ab) / 3.0;
	if (src.isContinuous() && dst.isContinuous())
	{
		Icol = Irow * Icol;
		Irow = 1;
	}
	const uchar* inData;
	float* outData;
	for ( i = 0; i < Irow; i++)
	{
		inData = src.ptr<uchar>(i);
		outData = dst.ptr<float>(i);
		for ( j = 0; j < Icol; j++)
		{
			*outData = 1 - 0.95 * (*inData) / Ac;
			outData++;
			inData++;
		}
	}
}

void deHaze::imgRecover(Mat src, Mat tFined, Mat & dst, int globalA)
{
	int i, j, Irow = src.rows, Icol = src.cols;
	float tmp, t0 = 0.1;
	dst = Mat::zeros(Irow, Icol, CV_8UC3);
	if (src.isContinuous() && tFined.isContinuous())
	{
		Icol = Irow * Icol;
		Irow = 1;
	}

	const uchar* inData;
	const float* tData;
	uchar* outData;
	for (i = 0; i < Irow; i++)
	{
		inData = src.ptr<uchar>(i);
		tData = tFined.ptr<float>(i);
		outData = dst.ptr<uchar>(i);
		for (j = 0; j < Icol; j++)
		{
			tmp = MAX(*tData, t0);
			tData++;
			*outData++ = scaleRGB((*inData - globalA) / tmp + globalA);
			inData++;
			*outData++ = scaleRGB((*inData - globalA) / tmp + globalA);
			*inData++;
			*outData++ = scaleRGB((*inData - globalA) / tmp + globalA);
			*inData++;
		}
	}
}


