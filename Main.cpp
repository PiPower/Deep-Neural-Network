#include <iostream>
#include "DenseLayer.h"
#include "Functions.h"
#include "Network.h"
#include <algorithm>
#include "Mnist_Loader.h"
using namespace std;

int main()
{
	int NumberImg = 50000;
	int TestImg = 10000;

	DenseLayer lel{28*28,30,Sigmoid,DerivativeSigmoid,MatrixInit::RANDOM_INIT };
	DenseLayer lel3{ 30,10,Sigmoid,DerivativeSigmoid,MatrixInit::RANDOM_INIT };
	Network net;
	net.AddLayer(lel);
	net.AddLayer(lel3);
	net.SetCostFun(MSE, MSE_Der);
	//---------------------
	vector<Matrix<double>> TrainingData;
	vector<Matrix<double>> TrainingLabel;

	mnist_loader mln("Train_Img_Data.idx1-ubyte", "Train_Labels.idx1-ubyte", NumberImg);

	for (int i = 0;i < NumberImg; i++)
	{
		auto img = mln.images(i);
		Matrix<double> ImgMat(1, 28 * 28);
		for (int j = 0; j < 28 * 28; j++)
		{
			ImgMat[j] = img[j];
		}
		TrainingData.push_back(ImgMat);

		Matrix<double> Label(1, 10);
		Label.SetValue(0, mln.labels(i), 1);
		TrainingLabel.push_back(Label);
	}



	net.Train(TrainingData, TrainingLabel, 100, 10, 0.4);

	vector<Matrix<double>> TestData;
	vector<Matrix<double>> TestLabel;

	mnist_loader mln2("Train_Img_Data.idx1-ubyte", "Train_Labels.idx1-ubyte", TestImg);

	for (int i = 0; i < TestImg; i++)
	{
		auto img = mln2.images(i);
		Matrix<double> ImgMat(1, 28 * 28);
		for (int j = 0; j < 28 * 28; j++)
		{
			ImgMat[j] = img[j];
		}
		TestData.push_back(ImgMat);

		Matrix<double> Label(1, 10);
		Label.SetValue(0, mln2.labels(i), 1);
		TestLabel.push_back(Label);
	}

	auto t = net.Predict(TestData);
	int counter=0;
	for (int i = 0; i < TestImg; i++)
	{
		if (t[i].GetColumnMaxIndex(0) == TestLabel[i].GetColumnMaxIndex(0))
		{
			counter++;
		}
	}


	cout << "Accuracy: "<< counter << "/"<< TestImg;
}