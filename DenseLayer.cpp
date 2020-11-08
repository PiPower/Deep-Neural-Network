#include "DenseLayer.h"


DenseLayer::DenseLayer(int input_dim, int output_dim, Matrix<double>(*activation)(Matrix<double>& Z), Matrix<double>(*activationDer)(Matrix<double>& Z),MatrixInit init)
	:
	Weights(input_dim,output_dim,init),Biases(1,output_dim,init),Output_Dim(output_dim), Input_Dim(input_dim)
{
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

