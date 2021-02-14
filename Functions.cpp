#include "Functions.h"

Matrix<float> ActivationFunction::Function(Matrix<float> Z)
{
	return Z;
}

Matrix<float> ActivationFunction::Function_Der(Matrix<float> Z)
{
	return Z;
}

ActivationFunction::~ActivationFunction()
{
}


Matrix<float> Sigmoid::Function(Matrix<float> Z)
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

float Sigmoid::SigmoidFunction(float x)
{
	return  1 / (std::pow(M_E, -x) + 1);
}

Matrix<float> Sigmoid::Function_Der(Matrix<float> z)
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

Matrix<float> RELU::Function(Matrix<float> Z)

{
	for (int j = 0; j < Z.GetColumns(); j++)
	{
		for (int i = 0; i < Z.GetRows(); i++)
		{
			float x = Z.GetAt(i, j) * (Z.GetAt(i, j) > 0);
			Z.SetValue(i, j, x);
		}
	}
	return Z;
}

Matrix<float> RELU::Function_Der(Matrix<float> Z)
{
	for (int j = 0; j < Z.GetColumns(); j++)
	{
		for (int i = 0; i < Z.GetRows(); i++)
		{
			float x = 1.0 * (Z.GetAt(i, j) > 0);
			Z.SetValue(i, j, x);
		}
	}
	return Z;
}

float TanH::TanHFunction(float x)
{
	return (std::pow(M_E, 2.0*x)-1) / (std::pow(M_E, 2.0*x) + 1);
}

Matrix<float> TanH::Function(Matrix<float> Z)
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

Matrix<float> TanH::Function_Der(Matrix<float> Z)
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
Matrix<float> Softmax::Function(Matrix<float> Z)
{
	float denominator = 0;
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

Matrix<float> Softmax::Function_Der(Matrix<float> Z)
{
	float denominator = 0;
	for (int i = 0; i < Z.GetRows(); i++)
	{
		denominator += pow(M_E, Z.GetAt(i, 0));
	}


	for (int i = 0; i < Z.GetRows(); i++)
	{
		float Value = 0;
		float Z_i = pow(M_E, Z.GetAt(i, 0));
		for (int j = 0; j < Z.GetRows(); j++)
		{
			if (i == j) Value += Z_i / denominator * (1.0 - Z_i / denominator);
			else Value += -(Z_i / denominator) * (pow(M_E, Z.GetAt(j, 0)) / denominator);
		}

		Z.SetValue(i, 0, Value);
	}
	return Z;
}
