#include <iostream>
#include "DenseLayer.h"
#include "Functions.h"
#include "Network.h"
using namespace std;

int main()
{
	DenseLayer lel{10,20,Sigmoid,MatrixInit::RANDOM_INIT };
	DenseLayer lel2{ 20,7,Sigmoid,MatrixInit::RANDOM_INIT };
	DenseLayer lel3{ 7,2,Sigmoid,MatrixInit::RANDOM_INIT };
	Network net(2.90);
	net.AddLayer(lel);
	net.AddLayer(lel2);
	net.AddLayer(lel3);

	vector<Matrix<double>> lul;
	lul.emplace_back(Matrix<double>(1, 10));

	auto M = net.Predict(lul);
	cout << "hello world";
}