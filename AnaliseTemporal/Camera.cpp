#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat discreteFourierTransform(String filename) {
	Mat I = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (I.empty()) {
		printf("imagem vazia");
	}

	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
	// viewable image form (float between values 0 and 1).

	//imshow("Input Image", I);    // Show the result
	imshow("spectrum magnitude", magI);

	return magI;
}

int turnIntoSquareImage(String filename) {
	Mat I = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (I.empty()) {
		printf("imagem vazia");
		return -1;
	}

	int max_side = 0;
	int min_side = 0;

	if (I.cols < I.rows){
		max_side = I.rows;
		min_side = I.cols;
		Mat imgPanel(max_side, max_side, CV_8UC1, Scalar(0));
		int dif_sides = max_side - min_side;
		Mat imgPanelRoi(imgPanel, Rect(dif_sides / 2, 0, I.cols, I.rows));
		I.copyTo(imgPanelRoi);
		//imshow("Squared Image - vertical", imgPanel);
		imwrite("frame.bmp", imgPanel);
	}
	else{
		max_side = I.cols;
		min_side = I.rows;
		Mat imgPanel(max_side, max_side, CV_8UC1, Scalar(0));
		int dif_sides = max_side - min_side;
		Mat imgPanelRoi(imgPanel, Rect(0, dif_sides/2, I.cols, I.rows));
		I.copyTo(imgPanelRoi);
		//imshow("Squared Image - horizontal", imgPanel);
		imwrite("frame.bmp", imgPanel);
	}

	waitKey();

	return 0;
}

static void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv,
	cv::Mat &X, cv::Mat &Y)
{
	cv::repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
	cv::repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);
}

// helper function (maybe that goes somehow easier)
static void meshgridTest(const cv::Range &xgv, const cv::Range &ygv,
	cv::Mat &X, cv::Mat &Y)
{
	std::vector<int> t_x, t_y;
	for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
	for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);
	meshgrid(cv::Mat(t_x), cv::Mat(t_y), X, Y);
}


int powerspectra(Mat in) {
	float value;
	Mat dftSquare;
	for (int i = 0; i<in.rows; i++)
	for (int j = 0; j < in.cols; j++) {
		// calcular o quadrado de cada pixel
		value = in.at<float>(i, j);	
		in.at<float>(i, j) = value*value;
	}

	imshow("dft square", in);

	// meshgrid 
	int imagesize = in.cols;
	Mat x, y;
	meshgridTest(cv::Range(-imagesize / 2, imagesize / (2 - 1)), cv::Range(-imagesize / 2, imagesize / (2 - 1)), x, y);
	
	// ver valores de x e y para ver se estamos a fazer bem
	/*std::cerr << x << std::endl;
	std::cerr << y << std::endl;*/

	Mat x2, y2;
	x.convertTo(x2, CV_32F);
	y.convertTo(y2, CV_32F);

	// converter as coordenadas de cartezianas para polares
	Mat rho, theta;

	cv::cartToPolar(x2, y2, rho, theta);

	for (int i = 0; i < rho.rows; i++)
	for (int j = 0; j < rho.cols; j++) {
		rho.at<float>(i, j) = cvRound(rho.at<float>(i, j));
	}

	//intervalo [1:(M/2)+1]
	int intervaloR = (imagesize/2) + 1;
	float r = 0;
	std::vector<float> myArray;
	for (int i = 0; i < rho.rows; i++)
	for (int j = 0; j < rho.cols; j++) {
		r = rho.at<float>(i, j);
		if ( (r >= 1) && (r <= intervaloR)) {
			if ( (i < in.cols) && (j < in.rows) ) {
				in.at<float>(i, j) += r;
				myArray.push_back(in.at<float>(i, j));
			}
		}
	}
	
	/*// imprimir o vector
	for (std::vector<float>::const_iterator i = myArray.begin(); i != myArray.end(); ++i)
		std::cout << *i << ' ';*/

	//média??

	return 0;
}

int captureVideo() {
	VideoCapture stream1("videos/approaching_lv_40ms_translate_approach.avi");   //0 is the id of video device.0 if you have only one camera.

	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	}

	Mat cameraFrame;
	//unconditional loop
	while (true) {
		stream1.read(cameraFrame);
		if (!cameraFrame.empty()) {
			imshow("cam", cameraFrame);
			imwrite("frame.bmp",cameraFrame);

			//tornar a imagem quadrada adicionando pixeis pretos nas margens que forem necessarias
			turnIntoSquareImage("frame.bmp");

			Mat in = discreteFourierTransform("frame.bmp");
			powerspectra(in);
			if (waitKey(30) >= 0)
				break;
		}
	}
	return 0;
}
