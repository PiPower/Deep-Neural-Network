#include <iostream>
#include "DenseLayer.h"
#include "Functions.h"
using namespace std;

int main()
{
	DenseLayer lel{10,20,Sigmoid};
	auto M = lel.Mul(Matrix<double>(1, 10));
	cout << "hello world";
}