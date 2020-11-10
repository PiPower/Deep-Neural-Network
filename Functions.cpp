#include "Functions.h"

Matrix<double> ActivationFunction::Function(Matrix<double>& Z)
{
	return Matrix<double>();
}

Matrix<double> ActivationFunction::Function_Der(Matrix<double>& Z)
{
	return Matrix<double>();
}

ActivationFunction::~ActivationFunction()
{
}


Matrix<double> Sigmoid::Function(Matrix<double>& Z)
{
	for (int i = 0; i < Z.GetRows(); i++)
	{
		Z.SetValue(i, 0, SigmoidFunction(Z.GetAt(i, 0)));
	}
	return Z;
}

double Sigmoid::SigmoidFunction(double x)
{
	return  1 / (std::pow(M_E, -x) + 1);
}

Matrix<double> Sigmoid::Function_Der(Matrix<double>& z)
{
	for (int i = 0; i < z.GetRows(); i++)
	{
		auto x=z.GetAt(0, 0);
		z.SetValue(i, 0, SigmoidFunction(z.GetAt(i, 0)) * (1 - SigmoidFunction(z.GetAt(i, 0))));
	}
	return z;
}
