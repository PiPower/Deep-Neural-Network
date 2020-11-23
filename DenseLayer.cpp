#include "DenseLayer.h"

DenseLayer::DenseLayer(int input_dim, int output_dim, ActivationFunction* Func_,MatrixInit init, WeightNormalization W_Init)
	:
Output_Dim(output_dim), Input_Dim(input_dim)
{
	std::normal_distribution<double> unif(0, 1);
	std::random_device rd;
	std::mt19937_64 gen(rd());
	Weights = Matrix( input_dim, output_dim, init, &gen, &unif );
	switch (W_Init)
	{
	case WeightNormalization::RoI:
		Weights = Weights * (1.0 / sqrt(input_dim));
		break;
	case WeightNormalization::DoubleRoI:
		Weights = Weights * (2.0 / sqrt(input_dim));
		break;
	default:
		break;
	}
	Biases = Matrix{ 1, output_dim, init, &gen, &unif };
	Func = Func_;
}


void DenseLayer::UpdateWeights(const Matrix<double>& Weights_)
{
	Weights += Weights_;
}

void DenseLayer::UpdateBiases(const Matrix<double>& Biases_)
{
	Biases += Biases_;
}

std::vector<Matrix<double>> DenseLayer::ActivationPrime(std::vector<Matrix<double>> Z)
{
	std::vector<Matrix<double>> F{ Func->Function_Der(Z.back()) };
	 return F;
}

std::vector<Matrix<double>> DenseLayer::ApplyActivation(std::vector<Matrix<double>> Z)
{
	std::vector<Matrix<double>> F{ Func->Function(Z.back()) };
	return F;
}

std::vector<Matrix<double>> DenseLayer::Mul(std::vector<Matrix<double>>& A)
{
	std::vector<Matrix<double>> Z{ Weights * A.back() + Biases };
	return Z;
}

DenseLayer::~DenseLayer()
{
	if (Func != nullptr)
	{
		delete Func;
		Func = nullptr;
	}
}

