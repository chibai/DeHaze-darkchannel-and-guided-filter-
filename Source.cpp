#include "deHaze.h"
#include <iostream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	deHaze dhObj;
	Mat kk, tmp, img = imread("2.bmp");
	if (!dhObj.load(img))
	{
		cerr << "fail to load the image" << endl;
		exit(-1);
	}
	//initialize opencv3.0 functions
	blur(tmp, img, Size(5, 5));
	kk = img.t();
	//
	double t = (double)getTickCount();
	Mat c = dhObj.practicalHazeRemoval(15, 41);
	t = (double)getTickCount() - t;
	cout << 1000 * t / (getTickFrequency()) << "ms" << endl;
	imwrite("test.jpg", c);
	return 0;
}