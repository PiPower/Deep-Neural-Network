#pragma once
#include <cmath>
const double M_E = 2.71828182845904523536;

double  Sigmoid(double x)
{
	return  1 / (std::pow(M_E, -x) + 1);
}


Matrix<double> Sigmoid(Matrix<double>& z)
{
	for (int i = 0; i < z.GetRows(); i++)
	{
		z.SetValue(i, 0, Sigmoid(z.GetAt(i, 0))  );
	}
	return z;
}
Matrix<double> DerivativeSigmoid(Matrix<double>& z)
{
	
	for (int i = 0; i < z.GetRows(); i++)
	{
		z.SetValue(i, 0,  Sigmoid(z.GetAt(i, 0)) * ( 1 - Sigmoid(z.GetAt(i, 0)) )   )    ;
	}
	return z;
}

Matrix<double> RELU(Matrix<double>& z)
{
	for (int i = 0; i < z.GetRows(); i++)
	{
		z.SetValue(i,0,std::max(0.0, z.GetAt(i,0)) );
	}
	return z;
}
Matrix<double> DerivativeRELU(Matrix<double>& z)
{
	for (int i = 0; i < z.GetRows(); i++)
	{
		z.SetValue(i, 0, ( z.GetAt(i, 0) > 0)  );
	}
	return z;
}

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
			if (i == j) Value += Z_i / denominator * (1.0 - Z_i / denominator);
			else Value += -(Z_i / denominator)* (pow(M_E, z.GetAt(j, 0)) / denominator);
		}

		z.SetValue(i, 0, Value );
	}
	return z;
}

Matrix<double> MSE(Matrix<double> A, Matrix<double> Y)
{
	return (A - Y) * (A - Y);
}

Matrix<double> MSE_Der(Matrix<double> A, Matrix<double> Y)
{
	return (A - Y);
}
