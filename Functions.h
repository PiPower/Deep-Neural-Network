#pragma once

#include <cmath>
#include "Matrix.h"
const float M_E = 2.71828182845904523536;

class ActivationFunction
{
protected:
	ActivationFunction() = default;
public:
	virtual Matrix<float> Function(Matrix<float> Z);
	virtual Matrix<float> Function_Der(Matrix<float> Z);
	virtual ~ActivationFunction();
};

class Sigmoid : public ActivationFunction
{
public:
	Sigmoid() = default;

	Matrix<float> Function(Matrix<float> Z);
	Matrix<float> Function_Der(Matrix<float> Z);
	float  SigmoidFunction(float x);
};

class RELU : public ActivationFunction
{
public:
	RELU() = default;

	Matrix<float> Function(Matrix<float> Z);
	Matrix<float> Function_Der(Matrix<float> Z);
};

class TanH : public ActivationFunction
{
public:
	TanH() = default;
	float TanHFunction(float x);
	Matrix<float> Function(Matrix<float> Z);
	Matrix<float> Function_Der(Matrix<float> Z);
};


class Softmax : public ActivationFunction
{
public:
	Softmax() = default;
	Matrix<float> Function(Matrix<float> Z);
	Matrix<float> Function_Der(Matrix<float> Z);
};

/*

Matrix<float> Softmax( Matrix<float> z)
{
	float denominator = 0;
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

Matrix<float> DerivativeSoftmax(Matrix<float> z)
{
	float denominator = 0;
	for (int i = 0; i < z.GetRows(); i++)
	{
		denominator += pow(M_E, z.GetAt(i, 0));
	}


	for (int i = 0; i < z.GetRows(); i++)
	{
		float Value = 0;
		float Z_i = pow(M_E, z.GetAt(i, 0));
		for (int j = 0; j < z.GetRows(); j++)
		{
			if (i == j) Value += Z_i / denominator * (1.0 - Z_i / denominator)* z.GetAt(j, 0);
			else Value += -(Z_i / denominator)* (pow(M_E, z.GetAt(j, 0)) / denominator)* z.GetAt(j, 0);
		}

		z.SetValue(i, 0, Value );
	}
	return z;
}*/
