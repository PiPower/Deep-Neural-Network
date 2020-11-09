#include "DenseLayer.h"


DenseLayer::DenseLayer(int input_dim, int output_dim, Matrix<double>(*activation)(Matrix<double>& Z), Matrix<double>(*activationDer)(Matrix<double>& Z),MatrixInit init)
	:
Output_Dim(output_dim), Input_Dim(input_dim)
{
	std::normal_distribution<double> unif(0, 1);
	std::random_device rd;
	std::mt19937_64 gen(rd());
	Weights = Matrix{ input_dim, output_dim, init, &gen, &unif };
	Biases = Matrix{ 1, output_dim, init, &gen, &unif };
	Activation = activation;
	ActivationDer = activationDer;
}


Matrix<double> DenseLayer::ActivationPrime(Matrix<double> Z)
{
	 Z = ActivationDer(Z);
	 return Z;
}

Matrix<double> DenseLayer::ApplyActivation(Matrix<double> Z)
{
	Z = Activation(Z);
	return Z;
}

Matrix<double> DenseLayer::Mul(Matrix<double>& A)
{
	Matrix<double> Z = Weights*A + Biases;
	return Z;
}

