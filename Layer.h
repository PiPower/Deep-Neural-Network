#pragma once
#include "Matrix.h"
#include "Functions.h"

class BaseLayer
{
protected:
	std::vector<Matrix<double>> Weights;
	Matrix<double> Biases;
	ActivationFunction* Func = nullptr;
};