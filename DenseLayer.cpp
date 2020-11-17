#include "DenseLayer.h"

DenseLayer::DenseLayer(int input_dim, int output_dim, ActivationFunction* Func_,MatrixInit init)
	:
Output_Dim(output_dim), Input_Dim(input_dim)
{
	std::normal_distribution<double> unif(0, 1);
	std::random_device rd;
	std::mt19937_64 gen(rd());
	Weights = Matrix( input_dim, output_dim, init, &gen, &unif );
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

Matrix<double> DenseLayer::ActivationPrime(Matrix<double> Z)
{
	 auto F = Func->Function_Der(Z);
	 return F;
}

Matrix<double> DenseLayer::ApplyActivation(Matrix<double> Z)
{
	auto F  = Func->Function(Z);
	return F;
}

Matrix<double> DenseLayer::Mul(Matrix<double>& A)
{
	Matrix<double> Z = Weights*A + Biases;
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

