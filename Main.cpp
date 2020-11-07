#include <iostream>
#include "DenseLayer.h"
#include "Functions.h"
#include "Network.h"
#include <algorithm>
#include "Mnist_Loader.h"
using namespace std;

int main()
{
	DenseLayer lel{28*28,20,Sigmoid,DerivativeSigmoid,MatrixInit::RANDOM_INIT };
	DenseLayer lel2{ 20,15,Sigmoid,DerivativeSigmoid,MatrixInit::RANDOM_INIT };
	DenseLayer lel3{ 15,10,Sigmoid,DerivativeSigmoid,MatrixInit::RANDOM_INIT };
	Network net;
	net.AddLayer(lel);
	net.AddLayer(lel2);
	net.AddLayer(lel3);
	net.SetCostFun(MSE, MSE_Der);
	vector<Matrix<double>> lul;
	Matrix<double> lel0(1, 28 * 28);

	mnist_loader mln("Train_Img_Data.idx1-ubyte", "Train_Labels.idx1-ubyte", 1);
	auto img = mln.images(0);

	for (int i = 0; i < 28 * 28;i++) lel0.SetValue(0,i, img[i]);
	lul.push_back(lel0);

	Matrix<double> Label(1,10);
	Label.SetValue(0, mln.labels(0), 1);

	vector<Matrix<double>> lul2;
	lul2.push_back(Label);


	net.Train(lul, lul2, 1, 3, 0.3);
	cout << "hello world";

}