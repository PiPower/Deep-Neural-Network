#include <iostream>
#include "DenseLayer.h"
#include "Network.h"
#include <algorithm>
#include "Mnist_Loader.h"
#include "CostFunction.h"
using namespace std;

int main()
{
	int NumberImg = 100;
	int TestImg = 10000;

	Network net;
	net.AddLayer(new DenseLayer{ 28 * 28,64,new RELU(),MatrixInit::RANDOM_INIT });
	net.AddLayer(new DenseLayer{ 64,32,new RELU(),MatrixInit::RANDOM_INIT });
	net.AddLayer(new DenseLayer{ 32,10,new TanH(),MatrixInit::RANDOM_INIT });
	net.SetCostFun(new QuadraticCost());
	//---------------------
	vector<Matrix<double>> TrainingData;
	vector<Matrix<double>> TrainingLabel;

	mnist_loader mln("train-images.idx3-ubyte", "train-labels.idx1-ubyte", NumberImg);

	
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



	net.Train(TrainingData, TrainingLabel, 10, 5 , 0.3);


	vector<Matrix<double>> TestData;
	vector<Matrix<double>> TestLabel;

	mnist_loader mln2("Test_Img_Data.idx1-ubyte", "Test_Labels.idx1-ubyte", TestImg);

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