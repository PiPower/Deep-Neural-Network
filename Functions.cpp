#include "Functions.h"

Matrix<double> ActivationFunction::Function(Matrix<double> Z)
{
	return Z;
}

Matrix<double> ActivationFunction::Function_Der(Matrix<double> Z)
{
	return Z;
}

ActivationFunction::~ActivationFunction()
{
}


Matrix<double> Sigmoid::Function(Matrix<double> Z)
{
	for (int j = 0; j < Z.GetColumns(); j++)
	{
		for (int i = 0; i < Z.GetRows(); i++)
		{
			Z.SetValue(i, 0, SigmoidFunction(Z.GetAt(i, j)));
		}
	}
	return Z;
}

double Sigmoid::SigmoidFunction(double x)
{
	return  1 / (std::pow(M_E, -x) + 1);
}

Matrix<double> Sigmoid::Function_Der(Matrix<double> z)
{
	for (int j = 0; j < z.GetColumns(); j++)
	{
		for (int i = 0; i < z.GetRows(); i++)
		{
			auto x = z.GetAt(i, j);
			z.SetValue(i, j, SigmoidFunction(z.GetAt(i, j)) * (1 - SigmoidFunction(z.GetAt(i, j))));
		}
	}
	return z;
}

Matrix<double> RELU::Function(Matrix<double> Z)

{
	for (int j = 0; j < Z.GetColumns(); j++)
	{
		for (int i = 0; i < Z.GetRows(); i++)
		{
			double x = Z.GetAt(i, j) * (Z.GetAt(i, j) > 0);
			Z.SetValue(i, j, x);
		}
	}
	return Z;
}

Matrix<double> RELU::Function_Der(Matrix<double> Z)
{
	for (int j = 0; j < Z.GetColumns(); j++)
	{
		for (int i = 0; i < Z.GetRows(); i++)
		{
			double x = 1.0 * (Z.GetAt(i, j) > 0);
			Z.SetValue(i, j, x);
		}
	}
	return Z;
}

double TanH::TanHFunction(double x)
{
	return (std::pow(M_E, 2.0*x)-1) / (std::pow(M_E, 2.0*x) + 1);
}

Matrix<double> TanH::Function(Matrix<double> Z)
{
	for (int j = 0; j < Z.GetColumns(); j++)
	{
		for (int i = 0; i < Z.GetRows(); i++)
		{
			Z.SetValue(i, j, TanHFunction(Z.GetAt(i, j)));
		}
	}
	return Z;
}

Matrix<double> TanH::Function_Der(Matrix<double> Z)
{
	for (int j = 0; j < Z.GetColumns(); j++)
	{
		for (int i = 0; i < Z.GetRows(); i++)
		{
			Z.SetValue(i, j, 1.0 - std::pow(TanHFunction(Z.GetAt(i, j)), 2.0));
		}
	}
	return Z;
}

// Its not gonna work with Conv Layer 
Matrix<double> Softmax::Function(Matrix<double> Z)
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

Matrix<double> Softmax::Function_Der(Matrix<double> Z)
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
