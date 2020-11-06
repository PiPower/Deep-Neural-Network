
#include "DenseLayer.h"


DenseLayer::DenseLayer(int input_dim, int output_dim, double  (*activation)(double Z), MatrixInit init)
	:
	Weights(input_dim,output_dim,init),Biases(1,output_dim,init),Output_Dim(output_dim), Input_Dim(input_dim)
{
	Activation = activation;
}


Matrix<double> DenseLayer::Mul(Matrix<double> A)
{
	Matrix<double> Z = Weights*A + Biases;
	Z.ApplyFunction( Activation);
	return Z;
}
