#include "DenseLayer.h"

DenseLayer::DenseLayer(int input_dim, int output_dim, ActivationFunction* Func_,MatrixInit init, WeightNormalization W_Init)
	:
Output_Dim(output_dim), Input_Dim(input_dim)
{
	std::normal_distribution<float> unif(0, 1);
	std::random_device rd;
	std::mt19937_64 gen(rd());
	Weights = Matrix( input_dim, output_dim, init, &gen, &unif );
	switch (W_Init)
	{
	case WeightNormalization::RoI:
		Weights = Weights * (1.0 / sqrt(input_dim));
		break;
	case WeightNormalization::floatRoI:
		Weights = Weights * (2.0 / sqrt(input_dim));
		break;
	default:
		break;
	}
	Biases = Matrix( 1, output_dim, init, &gen, &unif );
	Func = Func_;
}


void DenseLayer::UpdateWeights(const std::vector<Matrix<float>>& Weights_, const float& Eta)
{
	Weights += Weights_[0]* Eta;
}

void DenseLayer::UpdateBiases(const std::vector<Matrix<float>>& Biases_, const float& Eta)
{
	Biases += Biases_[0] * Eta;
}

std::vector<Matrix<float>> DenseLayer::ActivationPrime(std::vector<Matrix<float>>& Z)
{
	std::vector<Matrix<float>> F{ Func->Function_Der(Z.back()) };
	 return F;
}

std::vector<Matrix<float>> DenseLayer::ApplyActivation(std::vector<Matrix<float>>& Z)
{
	std::vector<Matrix<float>> F{ Func->Function(Z.back()) };
	return F;
}

std::vector<Matrix<float>> DenseLayer::Mul(std::vector<Matrix<float>>& A)
{
	std::vector<Matrix<float>> Z{ Weights * A.back() + Biases };
	return Z;
}

std::vector<Matrix<float>> DenseLayer::GetNablaWeight()
{
	return std::vector<Matrix<float>>{Matrix<float>(Input_Dim, Output_Dim)};
}

std::vector<Matrix<float>> DenseLayer::GetNablaBias()
{
	return std::vector<Matrix<float>>{Matrix<float>(1, Output_Dim)};
}

Tensor1D DenseLayer::CalculateNablaWeight( Tensor1D& Delta,  Tensor1D& Activation)
{
	return Delta*Activation;
}

Tensor1D DenseLayer::CalculateNablaBias( Tensor1D& Delta,  Tensor1D& Activation)
{
	return Delta;
}

DenseLayer::~DenseLayer()
{
	if (Func != nullptr)
	{
		delete Func;
		Func = nullptr;
	}
}

