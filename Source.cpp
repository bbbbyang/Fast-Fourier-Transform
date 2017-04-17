//-------------------------------------------------------------------
//				FAST FOURIER TRANSFORM
//				BINGYANG LIU
//				2015 - 07 - 23
//				OpenCV 3.0.0
//-------------------------------------------------------------------

//---------------- Head  File ---------------------------------------
#include "FastFourier.h"
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

//---------------- Name space ---------------------------------------
using namespace cv;
using namespace std;

int main(){
	// Read the image and resize it to fit the size of 2^n
	Mat Img = imread("woman_blonde.tif",  CV_LOAD_IMAGE_GRAYSCALE);
	resize(Img, Img, Size(256, 256), 0, 0, 1);
	
	namedWindow("The Original Image");
	imshow("The Original Image", Img);

	// Initialize FFT
	FourierTransform FFT(Img);
	FFT.ImageFourierTransform(Img);
	
	// Keep the window alive
	waitKey();
	return 1;
}