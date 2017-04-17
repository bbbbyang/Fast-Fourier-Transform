#include "FastFourier.h"
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

// Constructor
ComplexNumber::ComplexNumber(){
	Real = 0;
	Imaginary = 0;
}

// Print the complex number
void ComplexNumber::Print(){
	cout<<Real<<"  "<<Imaginary<<endl;
}

// Constructor
FourierTransform::FourierTransform(Mat Img){
	Number = Img.rows;
}

//----------------------- Perfect Shuffling ---------------------------------------
int FourierTransform::ReverseBit(int i){
	vector<int> BitNum;

	// Get the number of shuffle
	int Layers = log(Number) / log(2);

	// Get binary digit of the number of this pixel in the array
	int Layer_Of_I = log(i) / log(2) + 1;
	if(i == 0)
		Layer_Of_I = 1;
	int Zero_Num = Layers - Layer_Of_I;

	for(int k = 0; k < Layer_Of_I; k++){
		BitNum.push_back(i % 2);
		i /= 2;
	}
	
	for(int k = 0; k < Zero_Num; k++){
		BitNum.push_back(0);
	}

	for(int k = 0; k < Layers; k++){
		i += BitNum.at(k);
		i *= 2;
	}
	i /= 2;
	return i;
}
//------------------------------------------------------------------------------------

//-------------------------Get the coefficient --------------------------------------
float FourierTransform::CosArg(float u, int M){
	float Arg = cos(PI * u / M);
	return Arg;
}

float FourierTransform::SinArg(float u, int M){
	float Arg = sin(PI * u / M);
	return Arg;
}
//------------------------------------------------------------------------------------

void FourierTransform::OneFFT(ComplexNumber Img[], int Num){
	Number = Num;
	int M;
	int j;
	int Layer;

	ComplexNumber *F1 = new ComplexNumber[Number];
	ComplexNumber *F2 = new ComplexNumber[Number];

	// Reverse the array indices
	Layer = log(Number) / log(2);
	vector<int> NewIndex;
	for(int i = 0; i < Number; i++){
		NewIndex.push_back(ReverseBit(i));
	}

	// Reverse array elements
	for(int i = 0; i < Number; i++){
		F1[NewIndex[i]].Real	  = Img[i].Real;
		F1[NewIndex[i]].Imaginary = Img[i].Imaginary;
	}

	M = 1;					// The initial length of subgroups
	j = Number / 2;			// The number of pairs of subgroups
	
	// Successive mergings for n levels
	for(int i = 0; i < Layer; i++){
		// Merge pairs at the current group
		for(int k = 0; k < j; k++){
			int i1 = k * 2 * M;				// The start of the first group
			int i2 = (k * 2 + 1) * M;		// The start of the second group
			for(int u = 0; u < M; u++){

				F2[u].Real			= 0.5 * (F1[i1 + u].Real	  + (  F1[i2 + u].Real * CosArg(u, M) + F1[i2 + u].Imaginary * SinArg(u, M)));
				F2[u].Imaginary		= 0.5 * (F1[i1 + u].Imaginary + (- F1[i2 + u].Real * SinArg(u, M) + F1[i2 + u].Imaginary * CosArg(u, M)));
				F2[u + M].Real		= 0.5 * (F1[i1 + u].Real	  - (  F1[i2 + u].Real * CosArg(u, M) + F1[i2 + u].Imaginary * SinArg(u, M)));
				F2[u + M].Imaginary = 0.5 * (F1[i1 + u].Imaginary - (- F1[i2 + u].Real * SinArg(u, M) + F1[i2 + u].Imaginary * CosArg(u, M)));

			}

			for(int u = 0; u < M * 2; u++){
				F1[i1 + u] = F2[u];
			}
		}
		M *= 2;				// Double the subgroup length
		j /= 2;				// The number of groups is reduced by a half
	}

	for(int i = 0; i < Number; i++){
		Img[i].Real			= F1[i].Real;
		Img[i].Imaginary	= F1[i].Imaginary;
	}
		
	delete[] F1;
	delete[] F2;
}

void FourierTransform::InverseFFT(ComplexNumber Img[], int Num){
	Number = Num;
	int M;
	int j;
	int Layer;

	ComplexNumber *F1 = new ComplexNumber[Number];
	ComplexNumber *F2 = new ComplexNumber[Number];

	// Reverse the array indices
	Layer = log(Number) / log(2);
	vector<int> NewIndex;
	for(int i = 0; i < Number; i++){
		NewIndex.push_back(ReverseBit(i));
	}

	// Reverse array elements
	for(int i = 0; i < Number; i++){
		F1[NewIndex[i]].Real	  = Img[i].Real;
		F1[NewIndex[i]].Imaginary = Img[i].Imaginary;
	}

	M = 1;					// The initial length of subgroups
	j = Number / 2;			// The number of pairs of subgroups
	
	// Successive mergings for n levels
	for(int i = 0; i < Layer; i++){
		// Merge pairs at the current group
		for(int k = 0; k < j; k++){
			int i1 = k * 2 * M;					// The start of the first group
			int i2 = (k * 2 + 1) * M;			// The start of the second group
			for(int u = 0; u < M; u++){
	
				F2[u].Real			= (F1[i1 + u].Real	    + (F1[i2 + u].Real * CosArg(u, M) - F1[i2 + u].Imaginary * SinArg(u, M)));
				F2[u].Imaginary		= (F1[i1 + u].Imaginary + (F1[i2 + u].Real * SinArg(u, M) + F1[i2 + u].Imaginary * CosArg(u, M)));
				F2[u + M].Real		= (F1[i1 + u].Real	    - (F1[i2 + u].Real * CosArg(u, M) - F1[i2 + u].Imaginary * SinArg(u, M)));
				F2[u + M].Imaginary = (F1[i1 + u].Imaginary - (F1[i2 + u].Real * SinArg(u, M) + F1[i2 + u].Imaginary * CosArg(u, M)));

			}

			for(int u = 0; u < M * 2; u++){
				F1[i1 + u] = F2[u];
			}
		}
		M *= 2;				// Double the subgroup length
		j /= 2;				// The number of groups is reduced by a half
	}

	for(int i = 0; i < Number; i++){
		Img[i].Real			= F1[i].Real;
		Img[i].Imaginary	= F1[i].Imaginary;
	}
		
	delete[] F1;
	delete[] F2;
}

void FourierTransform::ImageFourierTransform(Mat& Img){

//----------------------- FFT for 2D Image  -----------------------------------------

	// Initialize the Complex Matrix
	ComplexNumber **Image = new ComplexNumber *[Img.rows];
	for(int i = 0; i < Img.rows; i++)
		Image[i] = new ComplexNumber[Img.cols];

	// Store the image data in the complex matrix
	for(int i = 0; i < Img.rows; i++){
		for(int j = 0; j < Img.cols; j++){
		//	Image[i][j].Real = log(1 + (double)Img.at<uchar>(i, j));		// This one is for Homomorphic Filtering
			Image[i][j].Real = (double)Img.at<uchar>(i, j);
		}
	}
	
	// Rows Fast Fourier Transform
	for(int i = 0; i < Img.rows; i++){
		OneFFT(Image[i], Img.cols);
	}
	
//	for(int i = 0; i < Img.cols; i++)
//		Image[0][i].Print();

	// Cols Fast Fourier Transform
	ComplexNumber *Cols = new ComplexNumber[Img.rows];
	for(int i = 0; i < Img.cols; i++){
		for(int j = 0; j < Img.rows; j++){
			Cols[j].Real	  = Image[j][i].Real;
			Cols[j].Imaginary = Image[j][i].Imaginary;
		}
		OneFFT(Cols, Img.rows);
		for(int j = 0; j < Img.rows; j++){
			Image[j][i].Real	  = Cols[j].Real;
			Image[j][i].Imaginary = Cols[j].Imaginary;
		}
	}

//------------------------------------------------------------------------------------

//		Four Quadrants
//		UL(2)		|		UR(1)   
//		------------------------
//		LL(3)		|		LR(4)

//------------------------------ Filtering  -----------------------------------------
	// Transform the quadrant
	for(int i = 0; i < Img.rows / 2; i++){
		for(int j = 0; j < Img.cols / 2; j++){
			ComplexNumber Temp;

			// Exchange Quadrant 2 and Quanrant 4
			Temp.Real	   = Image[i][j].Real;
			Temp.Imaginary = Image[i][j].Imaginary;

			Image[i][j].Real	  = Image[i + Img.cols / 2][j + Img.rows / 2].Real;
			Image[i][j].Imaginary = Image[i + Img.cols / 2][j + Img.rows / 2].Imaginary;

			Image[i + Img.cols / 2][j + Img.rows / 2].Real		= Temp.Real;
			Image[i + Img.cols / 2][j + Img.rows / 2].Imaginary = Temp.Imaginary;

			// Exchange Quadrant 1 and Quadrant 3
			Temp.Real	   = Image[i + Img.cols / 2][j].Real;
			Temp.Imaginary = Image[i + Img.cols / 2][j].Imaginary;

			Image[i + Img.cols / 2][j].Real		 = Image[i][j + Img.rows / 2].Real;
			Image[i + Img.cols / 2][j].Imaginary = Image[i][j + Img.rows / 2].Imaginary;

			Image[i][j + Img.rows / 2].Real		 = Temp.Real;
			Image[i][j + Img.rows / 2].Imaginary = Temp.Imaginary;
			
		}
	}

	/*
	for(int i = 0; i < Img.rows; i++){
		for(int j = 0; j < Img.cols; j++){
			Image[i][j].Print();
		}
	}*/
	// Store the FFT result in Mat form and scale the data
	Mat FFT(Img.rows, Img.cols, CV_32F);
	for(int i = 0; i < Img.rows; i++){
		for(int j = 0; j < Img.cols; j++){
	//		Img.at<uchar>(i, j) = sqrt(Image[i][j].Real * Image[i][j].Real + Image[i][j].Imaginary * Image[i][j].Imaginary) * Number;
			FFT.at<float>(i, j) = sqrt(Image[i][j].Real * Image[i][j].Real + Image[i][j].Imaginary * Image[i][j].Imaginary);
		}
	}
	
	double min, max;
	Point minloc, maxloc;
	log(FFT + 1, FFT);
	log(FFT + 1, FFT);
	log(FFT + 1, FFT);
	minMaxLoc(FFT, &min, &max, &minloc, &maxloc);
	double s = 255 / (max - min);
	FFT *= s;
	FFT.convertTo(FFT, CV_8U);
	
	namedWindow("FFT IMG");
	imshow("FFT IMG", FFT);

	imwrite("FFT.jpg", FFT);
	
//----------------------------- Do the Filtering  ------------------------------------
	Filtering ImageFilter(Img, 40);

	// Choose filter type
	// If you choose Homomorphic Filtering, you should change code that get image information
	// f(x, y)  ->  ln(f(x, y))
	// and change the code that get the final image
	// lnF(x, y)  ->  F(x, y)
	ImageFilter.IdealIowPassFiltering(Image);
//	ImageFilter.IdealHighPassFiltering(Image);
//	ImageFilter.ButterworthLowPassFiltering(Image);
//	ImageFilter.ButterworthHighPassFiltering(Image);
//	ImageFilter.HomomorphicFiltering(Image);
//-------------------------------Filtering Part --------------------------------------

	// Transform the quadrant
	for(int i = 0; i < Img.rows / 2; i++){
		for(int j = 0; j < Img.cols / 2; j++){
			ComplexNumber Temp;

			// Exchange Quadrant 2 and Quanrant 4
			Temp.Real	   = Image[i][j].Real;
			Temp.Imaginary = Image[i][j].Imaginary;

			Image[i][j].Real	  = Image[i + Img.cols / 2][j + Img.rows / 2].Real;
			Image[i][j].Imaginary = Image[i + Img.cols / 2][j + Img.rows / 2].Imaginary;

			Image[i + Img.cols / 2][j + Img.rows / 2].Real		= Temp.Real;
			Image[i + Img.cols / 2][j + Img.rows / 2].Imaginary = Temp.Imaginary;

			// Exchange Quadrant 1 and Quadrant 3
			Temp.Real	   = Image[i + Img.cols / 2][j].Real;
			Temp.Imaginary = Image[i + Img.cols / 2][j].Imaginary;

			Image[i + Img.cols / 2][j].Real		 = Image[i][j + Img.rows / 2].Real;
			Image[i + Img.cols / 2][j].Imaginary = Image[i][j + Img.rows / 2].Imaginary;

			Image[i][j + Img.rows / 2].Real		 = Temp.Real;
			Image[i][j + Img.rows / 2].Imaginary = Temp.Imaginary;
			
		}
	}
//------------------------------------------------------------------------------------
	
//----------------------------- Inverse FFT  -----------------------------------------
	// Rows Fast Fourier Transform
	for(int i = 0; i < Img.rows; i++){
		InverseFFT(Image[i], Img.cols);
	}

	// Cols Fast Fourier Transform
	for(int i = 0; i < Img.cols; i++){
		for(int j = 0; j < Img.rows; j++){
			Cols[j].Real	  = Image[j][i].Real;
			Cols[j].Imaginary = Image[j][i].Imaginary;
		}
		InverseFFT(Cols, Img.rows);
		for(int j = 0; j < Img.rows; j++){
			Image[j][i].Real	  = Cols[j].Real;
			Image[j][i].Imaginary = Cols[j].Imaginary;
		}
	}

	// Get the filtered image
	for(int i = 0; i < Img.rows; i++){
		for(int j = 0; j < Img.cols; j++){
	//		Img.at<uchar>(i, j) = exp(sqrt(Image[i][j].Real * Image[i][j].Real + Image[i][j].Imaginary * Image[i][j].Imaginary));	// This one if for Homomorphic Filtering
			if(sqrt(Image[i][j].Real * Image[i][j].Real + Image[i][j].Imaginary * Image[i][j].Imaginary) > 255){
				Img.at<uchar>(i, j) = 255;
			} else
				Img.at<uchar>(i, j) = sqrt(Image[i][j].Real * Image[i][j].Real + Image[i][j].Imaginary * Image[i][j].Imaginary);
		}
	}
	
	namedWindow("Inverse Image");
	imshow("Inverse Image", Img);
	imwrite("InverseFFT.tif", Img);
//------------------------------------------------------------------------------------

//-------------------- Release the memory applied before ----------------------------
	for(int i = 0; i < Img.rows; i++)
		delete[] Image[i];
	delete[] Image;
	delete[] Cols;
//------------------------------------------------------------------------------------
}

//------------------------ Constructor function  -------------------------------------
Filtering::Filtering(Mat const Img, int Dis){
	Rows = Img.rows;
	Cols = Img.cols;
	Distance = Dis;
	Filter = new double *[Rows];
	for(int i = 0; i < Rows; i++){
		Filter[i] = new double [Cols];
	}

	// Initialize the matrix
	for(int i = 0; i < Rows; i++){
		for(int j = 0; j < Cols; j++){
			Filter[i][j] = 0;
		}
	}
}
//------------------------------------------------------------------------------------

//------------------------- Destructor function  -------------------------------------
Filtering::~Filtering(){
	// Release the memory applied
	for(int i = 0; i < Rows; i++){
		delete[] Filter[i];
	}
	delete[] Filter;
}
//------------------------------------------------------------------------------------

//---------------------- Ideal Low-Pass Filter(ILPF) ---------------------------------
//
//							|	1,		if D(u, v)	<= Distance
//				H(u, v) =	|
//							|	0,		otherwise
//
void Filtering::IdealIowPassFiltering(ComplexNumber** Image){
	
	int MR = Rows / 2;
	int MC = Cols / 2;
	
	// Get ring
	Mat FFT(Rows, Cols, CV_32F);
	for(int i = 0; i < Rows; i++){
		for(int j = 0; j < Cols; j++){
	//		Img.at<uchar>(i, j) = sqrt(Image[i][j].Real * Image[i][j].Real + Image[i][j].Imaginary * Image[i][j].Imaginary) * Number;
			FFT.at<float>(i, j) = sqrt(Image[i][j].Real * Image[i][j].Real + Image[i][j].Imaginary * Image[i][j].Imaginary);
		}
	}
	



	// Keep the low pass information
	for(int i = 0; i < Rows; i ++){
		for(int j = 0; j < Cols; j++){
			int Dis = (i - MR) * (i - MR) + (j - MC) * (j -MC);
			if(Dis < Distance * Distance){
				Filter[i][j] = 1;
			}
		}
	}

	// Filter
	for(int i = 0; i < Rows; i++){
		for(int j = 0; j < Cols; j++){
			Image[i][j].Real	  *= Filter[i][j];
			Image[i][j].Imaginary *= Filter[i][j];
		}
	}

	// Get ring
	for(int i = 0; i < Rows; i++){
		for(int j = 0; j < Cols; j++){
			int Dis = sqrt((i - MR) * (i - MR) + (j - MC) * (j -MC));
			if(abs(Dis - Distance) < 1){
				FFT.at<float>(i, j) = 255;
			}
		}
	}

	double min, max;
	Point minloc, maxloc;
	log(FFT + 1, FFT);
	log(FFT + 1, FFT);
	log(FFT + 1, FFT);
	minMaxLoc(FFT, &min, &max, &minloc, &maxloc);
	double s = 255 / (max - min);
	FFT *= s;
	FFT.convertTo(FFT, CV_8U);
	
	imshow("IFFTFiltering", FFT);
	imwrite("RingFFT.tif", FFT);

}
//------------------------------------------------------------------------------------

//----------------------- High Low-Pass Filter(HLPF) ---------------------------------
//
//							|	0,		if D(u, v)	<= Distance
//				H(u, v) =	|
//							|	1,		otherwise
//
void Filtering::IdealHighPassFiltering(ComplexNumber** Image){
	int MR = Rows / 2;
	int MC = Cols / 2;
	// Keep the high pass information
	for(int i = 0; i < Rows; i ++){
		for(int j = 0; j < Cols; j++){
			int Dis = (i - MR) * (i - MR) + (j - MC) * (j -MC);
			if(Dis < Distance * Distance){
				Filter[i][j] = 0;
			}
			else
				Filter[i][j] = 1;
		}
	}

	// Filter
	for(int i = 0; i < Rows; i++){
		for(int j = 0; j < Cols; j++){
			Image[i][j].Real	  *= Filter[i][j];
			Image[i][j].Imaginary *= Filter[i][j];
		}
	}
}
//------------------------------------------------------------------------------------

//-------------------- Butterworth Low-Pass Filter(HLPF) -----------------------------
//
//											1
//				H(u, v) =	------------------------------------  (n == 1)
//							  1 + [	D(u,v) / Distance ] ^ (2n)
//
void Filtering::ButterworthLowPassFiltering(ComplexNumber** Image){
	int MR = Rows / 2;
	int MC = Cols / 2;
	// Butterworth Filter
	for(int i = 0; i < Rows; i ++){
		for(int j = 0; j < Cols; j++){
			int Dis =sqrt((i - MR) * (i - MR) + (j - MC) * (j -MC));
			Filter[i][j] = 1.0f / (1.0 + pow((Dis / Distance), 2.0));
		}
	}

	// Filter
	for(int i = 0; i < Rows; i++){
		for(int j = 0; j < Cols; j++){
			Image[i][j].Real	  *= Filter[i][j];
			Image[i][j].Imaginary *= Filter[i][j];
		}
	}
}
//------------------------------------------------------------------------------------

//------------------- Butterworth High-Pass Filter(HLPF) -----------------------------
//
//											1
//				H(u, v) =	------------------------------------  (n == 1)
//							  1 + [	Distance / D(u,v) ] ^ (2n)
//
void Filtering::ButterworthHighPassFiltering(ComplexNumber** Image){
	int MR = Rows / 2;
	int MC = Cols / 2;
	// Butterworth Filter
	for(int i = 0; i < Rows; i ++){
		for(int j = 0; j < Cols; j++){
			int Dis =sqrt((i - MR) * (i - MR) + (j - MC) * (j -MC));
			Filter[i][j] = 1.0f / (1.0 + pow((Distance / Dis), 2.0));
		}
	}

	// Filter
	for(int i = 0; i < Rows; i++){
		for(int j = 0; j < Cols; j++){
			Image[i][j].Real	  *= Filter[i][j];
			Image[i][j].Imaginary *= Filter[i][j];
		}
	}
}
//------------------------------------------------------------------------------------

//-------------------------- Homomorphic Filter(HLPF) --------------------------------
//
//		ln(f(x, y))  ->  FFT  ->  H(x, y)  ->  IFFT  -> exp(f(x, y))
//
//												  D(u, v) ^ 2
//			H(u, v) = (rH - rL) [ 1.0 - e ^ (c -----------------)] + rL
//												  Distance ^ 2
//			rL == 0.5		rH == 2.0		c == -1.5
//
void Filtering::HomomorphicFiltering(ComplexNumber** Image){
	int MR = Rows / 2;
	int MC = Cols / 2;
	// Homomorphic Filter
	for(int i = 0; i < Rows; i ++){
		for(int j = 0; j < Cols; j++){
			int Dis = (i - MR) * (i - MR) + (j - MC) * (j -MC);
			Filter[i][j] = (2.0 - 0.5) * (1.0 - exp(-1.5 * Dis / (Distance * Distance))) + 0.5;
		}
	}

	// Filter
	for(int i = 0; i < Rows; i++){
		for(int j = 0; j < Cols; j++){
			Image[i][j].Real	  *= Filter[i][j];
			Image[i][j].Imaginary *= Filter[i][j];
		}
	}
}
//------------------------------------------------------------------------------------