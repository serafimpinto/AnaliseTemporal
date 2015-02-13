#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "cvplot.h"
#include <fstream>

using namespace cv;
using namespace std;

vector<float> alphas;

Mat discreteFourierTransform(Mat I) {

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
	//imshow("spectrum magnitude", magI);

	return magI;
}

Mat turnIntoSquareImage(Mat I) {

	if (I.empty()) {
		printf("imagem vazia");
		return I;
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
		return imgPanel;
	}
	else{
		max_side = I.cols;
		min_side = I.rows;
		Mat imgPanel(max_side, max_side, CV_8UC1, Scalar(0));
		int dif_sides = max_side - min_side;
		Mat imgPanelRoi(imgPanel, Rect(0, dif_sides/2, I.cols, I.rows));
		I.copyTo(imgPanelRoi);
		//imshow("Squared Image - horizontal", imgPanel);
		return imgPanel;
	}
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

float calcMedia(vector<float> valoresDFT) {
	float sum = 0;
	for (std::vector<float>::const_iterator i = valoresDFT.begin(); i != valoresDFT.end(); ++i) {
		sum += *i;
	}
	sum = sum / (valoresDFT.size());
	return sum;
}


int powerspectra(Mat in) {
	float value;

	for (int i = 0; i<in.rows; i++)
	for (int j = 0; j < in.cols; j++) {
		// calcular o quadrado de cada pixel
		value = in.at<float>(i, j);	
		in.at<float>(i, j) = value*value;
	}

	//imshow("dft square", in);

	// meshgrid 
	int imagesize = in.cols;
	Mat x, y;
	meshgridTest(cv::Range(-imagesize / 2, imagesize / (2 - 1)), cv::Range(-imagesize / 2, imagesize / (2 - 1)), x, y);

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
	//intervalo [1:(M/2)]
	int intervaloR = imagesize/2;

	vector<float> medias;
	for (int i = 1; i <= intervaloR; i++) {
		std::vector<float> valoresDFT;
		for (int j = 0; j < rho.rows; j++)
		for (int k = 0; k < rho.cols; k++) {
			if (i == rho.at<float>(j, k)){
				if ((j < in.cols) && (k < in.rows)) {
					valoresDFT.push_back(in.at<float>(k, j));
				}
			}
		}
		float media = calcMedia(valoresDFT);
		medias.push_back(media);
	}

	// encontrar o alpha com a fitline
	vector<Point2f> points;
	for (int r = 1; r <= intervaloR; r++) {
		Point2f p;
		p.x = log(medias.at(r - 1));
		p.y = log(r);
		points.push_back(p);
	}

	Vec4f line;
	fitLine(points, line, CV_DIST_L2, 0, 0.01, 0.01);

	float alpha = line[1] / line[0];

	printf("\nalpha %f", alpha);
	alphas.push_back(alpha);

	/*//imprimir o vector
	for (std::vector<Point2f>::const_iterator i = points.begin(); i != points.end(); ++i)
		std::cout << *i << ' ';*/

	return 0;
}

int captureVideo() {
	VideoCapture stream1("videos/car2.avi");   //0 is the id of video device.0 if you have only one camera.
	//VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.

	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	}

	Mat cameraFrame;
	//unconditional loop
	while (true) {
		stream1.read(cameraFrame);
		if (!cameraFrame.empty()) {
			imshow("cam", cameraFrame);
			cvtColor(cameraFrame, cameraFrame, CV_BGR2GRAY);
			//tornar a imagem quadrada adicionando pixeis pretos nas margens que forem necessarias
			cameraFrame = turnIntoSquareImage(cameraFrame);

			cameraFrame = discreteFourierTransform(cameraFrame);

			powerspectra(cameraFrame);

			if (waitKey(30) >= 0)
				break;
		}
		ofstream grafico;
		grafico.open("grafico.txt");
		for (int i = 1; i <= alphas.size(); i++) {
			grafico << i << " " << alphas.at(i-1) << "\n";
		}
		grafico.close();
	}
	return 0;
}
