#include <iostream>
#include "DenseLayer.h"
#include "Network.h"
#include <algorithm>
#include "Mnist_Loader.h"
#include "CostFunction.h"
#include "ConvLayer.h"
#include "Flatern.h"


using namespace std;

int main()
{
	int NumberImg = 60000;
	int TestImg = 10000;

	Network net;
	//net.AddLayer(new DenseLayer{ 28 * 28,64,new RELU(),MatrixInit::RANDOM_INIT,DenseLayer::WeightNormalization::floatRoI });
	//net.AddLayer(new DenseLayer{ 28*28,32,new Sigmoid(),MatrixInit::RANDOM_INIT,DenseLayer::WeightNormalization::floatRoI });
	//net.AddLayer(new ConvLayer(1, 1, Image_Dim{ 28,28 }, Image_Dim{ 5,5 }, new RELU(), MatrixInit::RANDOM_INIT, DenseLayer::WeightNormalization::floatRoI) );
	net.AddLayer(new ConvLayer(1, 2, Image_Dim{ 28,28 }, Image_Dim{ 5,5 }, new RELU(), MatrixInit::RANDOM_INIT, DenseLayer::WeightNormalization::floatRoI));
	net.AddLayer(new Flatern());
	net.AddLayer(new DenseLayer{ 24*24*2,10,new Sigmoid(),MatrixInit::RANDOM_INIT,DenseLayer::WeightNormalization::floatRoI }); //800
	net.SetCostFun(new QuadraticCost());
	//---------------------
	vector<Matrix<float>> TrainingData;
	vector<Matrix<float>> TrainingLabel;

	mnist_loader mln("train-images.idx3-ubyte", "train-labels.idx1-ubyte", NumberImg);

	
	for (int i = 0;i < NumberImg; i++)
	{
		auto img = mln.images(i);
		Matrix<float> ImgMat(28,28);
		//Matrix<float> ImgMat(1, 28*28);

		for (int j = 0; j < 28 * 28; j++)
		{
			ImgMat[j] = img[j];
		}
		TrainingData.push_back(ImgMat);

		Matrix<float> Label(1, 10);
		Label.SetValue(0, mln.labels(i), 1);
		TrainingLabel.push_back(Label);
	}



	net.Train(TrainingData, TrainingLabel, 100, 20 , 0.03);


	vector<Matrix<float>> TestData;
	vector<Matrix<float>> TestLabel;

	mnist_loader mln2("Test_Img_Data.idx1-ubyte", "Test_Labels.idx1-ubyte", TestImg);

	for (int i = 0; i < TestImg; i++)
	{
		auto img = mln2.images(i);
		Matrix<float> ImgMat(28, 28);
		//Matrix<float> ImgMat(1, 28 * 28);
		for (int j = 0; j < 28 * 28; j++)
		{
			ImgMat[j] = img[j];
		}
		TestData.push_back(ImgMat);

		Matrix<float> Label(1, 10);
		Label.SetValue(0, mln2.labels(i), 1);
		TestLabel.push_back(Label);
	}

	auto t = net.Predict(TestData);
	int counter=0;
	int cnt = 0;
	for (int i = 0; i < TestImg; i++)
	{
		if (t[i].GetColumnMaxIndex(0) == TestLabel[i].GetColumnMaxIndex(0))
		{
			counter++;
		}
	}


	cout << "Accuracy: "<< counter << "/"<< TestImg<<endl;
}