#include "Functions.h"

Matrix<double> ActivationFunction::Function(Matrix<double>& Z)
{
	return Z;
}

Matrix<double> ActivationFunction::Function_Der(Matrix<double>& Z)
{
	return Z;
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
		auto x = z.GetAt(i, 0);
		z.SetValue(i, 0, SigmoidFunction(z.GetAt(i, 0)) * (1 - SigmoidFunction(z.GetAt(i, 0))));
	}
	return z;
}

Matrix<double> RELU::Function(Matrix<double>& Z)
{
	for (int i = 0; i < Z.GetRows(); i++)
	{
		double x = Z.GetAt(i, 0) * (Z.GetAt(i, 0) > 0);
		Z.SetValue(i, 0, x);
	}
	return Z;
}

Matrix<double> RELU::Function_Der(Matrix<double>& Z)
{
	for (int i = 0; i < Z.GetRows(); i++)
	{
		double x = 1.0*(Z.GetAt(i, 0) > 0);
		Z.SetValue(i, 0, x);
	}
	return Z;
}

double TanH::TanHFunction(double x)
{
	return (std::pow(M_E, 2.0*x)-1) / (std::pow(M_E, 2.0*x) + 1);
}

Matrix<double> TanH::Function(Matrix<double>& Z)
{
	for (int i = 0; i < Z.GetRows(); i++)
	{
		Z.SetValue(i, 0, TanHFunction(Z.GetAt(i,0) ));
	}
	return Z;
}

Matrix<double> TanH::Function_Der(Matrix<double>& Z)
{
	for (int i = 0; i < Z.GetRows(); i++)
	{
		Z.SetValue(i, 0, 1.0-std::pow(TanHFunction(Z.GetAt(i,0)),2.0)  );
	}
	return Z;
}

Matrix<double> Softmax::Function(Matrix<double>& Z)
{
	double denominator = 0;
	for (int i = 0; i < Z.GetRows(); i++)
	{
		denominator += pow(M_E, Z.GetAt(i, 0));
	}

	for (int i = 0; i < Z.GetRows(); i++)
	{
		Z.SetValue(i, 0, pow(M_E, Z.GetAt(i, 0)) / denominator);
	}
	return Z;
}

Matrix<double> Softmax::Function_Der(Matrix<double>& Z)
{
	double denominator = 0;
	for (int i = 0; i < Z.GetRows(); i++)
	{
		denominator += pow(M_E, Z.GetAt(i, 0));
	}


	for (int i = 0; i < Z.GetRows(); i++)
	{
		double Value = 0;
		double Z_i = pow(M_E, Z.GetAt(i, 0));
		for (int j = 0; j < Z.GetRows(); j++)
		{
			if (i == j) Value += Z_i / denominator * (1.0 - Z_i / denominator);
			else Value += -(Z_i / denominator) * (pow(M_E, Z.GetAt(j, 0)) / denominator);
		}

		Z.SetValue(i, 0, Value);
	}
	return Z;
}
