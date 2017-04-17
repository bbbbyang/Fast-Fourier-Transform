#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

const double PI = 3.1415926;

/*
typedef enum{
	ILPF = 1,
	IHPF = 2,
	BLPF = 3,
	BHPF = 4,
	HMPH = 5
}FilteringType;
*/

class ComplexNumber{
	public:
		double Real;								// Real part of complex number
		double Imaginary;							// Imaginary part of complex number
	public:
		ComplexNumber();							// Constructor
		void Print();								// Print the complex number
};

class FourierTransform{
	public:
		int Number;									// Number for 1D FFT
	public:
		FourierTransform(cv::Mat);					// Constructor
		int ReverseBit(int);						// Reverse the array indices
		float CosArg(float, int);					// Get cos angle
		float SinArg(float, int);					// Get sin angle
		void OneFFT(ComplexNumber[], int);			// 1D FFT
		void InverseFFT(ComplexNumber[], int);		// 1D Inverse FFT
		void ImageFourierTransform(cv::Mat&);		// Image Filtering
};

class Filtering{
	public:
		double **Filter;										// Filter window
		double Distance;										// Distance for filter
		int Rows;												// Image row number
		int Cols;												// Image column number
	public:
		Filtering(cv::Mat const, int);							// Constructor function
		~Filtering();											// Destructor function
		void IdealIowPassFiltering(ComplexNumber**);			// Ideal Iow-Pass Filtering
		void ButterworthLowPassFiltering(ComplexNumber**);		// Butterworth Low-Pass Filtering
		void IdealHighPassFiltering(ComplexNumber**);			// Ideal High-Pass Filtering
		void ButterworthHighPassFiltering(ComplexNumber**);		// Butterworth High-Pass Filtering
		void HomomorphicFiltering(ComplexNumber**);				// Homomorphic Filtering
};