#pragma once

#include <cmath>
#include "Matrix.h"
const double M_E = 2.71828182845904523536;

class ActivationFunction
{
protected:
	ActivationFunction() = default;
public:
	virtual Matrix<double> Function(Matrix<double>& Z);
	virtual Matrix<double> Function_Der(Matrix<double>& Z);
	virtual ~ActivationFunction();
};

class Sigmoid : public ActivationFunction
{
public:
	Sigmoid() = default;

	Matrix<double> Function(Matrix<double>& Z);
	Matrix<double> Function_Der(Matrix<double>& Z);
	double  SigmoidFunction(double x);
};

class RELU : public ActivationFunction
{
public:
	RELU() = default;

	Matrix<double> Function(Matrix<double>& Z);
	Matrix<double> Function_Der(Matrix<double>& Z);
};

class TanH : public ActivationFunction
{
public:
	TanH() = default;
	double TanHFunction(double x);
	Matrix<double> Function(Matrix<double>& Z);
	Matrix<double> Function_Der(Matrix<double>& Z);
};


class Softmax : public ActivationFunction
{
public:
	Softmax() = default;
	Matrix<double> Function(Matrix<double>& Z);
	Matrix<double> Function_Der(Matrix<double>& Z);
};

/*

Matrix<double> Softmax( Matrix<double>& z)
{
	double denominator = 0;
	for (int i = 0; i < z.GetRows(); i++)
	{
		denominator += pow(M_E, z.GetAt(i, 0));
	}

	for (int i = 0; i < z.GetRows(); i++)
	{
		z.SetValue(i, 0, pow(M_E,z.GetAt(i,0))/denominator);
	}
	return z;
}

Matrix<double> DerivativeSoftmax(Matrix<double>& z)
{
	double denominator = 0;
	for (int i = 0; i < z.GetRows(); i++)
	{
		denominator += pow(M_E, z.GetAt(i, 0));
	}


	for (int i = 0; i < z.GetRows(); i++)
	{
		double Value = 0;
		double Z_i = pow(M_E, z.GetAt(i, 0));
		for (int j = 0; j < z.GetRows(); j++)
		{
			if (i == j) Value += Z_i / denominator * (1.0 - Z_i / denominator)* z.GetAt(j, 0);
			else Value += -(Z_i / denominator)* (pow(M_E, z.GetAt(j, 0)) / denominator)* z.GetAt(j, 0);
		}

		z.SetValue(i, 0, Value );
	}
	return z;
}*/
