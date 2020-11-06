#pragma once
#include <cmath>
const double M_E = 2.71828182845904523536;

double Sigmoid(double x)
{
	return  std::pow(M_E, x) / (std::pow(M_E, x) + 1);
}
double DerivativeSigmoid(double x)
{
	return Sigmoid(x) * (1 - Sigmoid(x));
}
