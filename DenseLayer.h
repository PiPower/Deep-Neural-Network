#pragma once
#include "Matrix.h"

class Network;
class DenseLayer
{
	friend Network;
public:
	DenseLayer(int input_dim, int output_dim, Matrix<double>(*activation)(Matrix<double>& Z) , Matrix<double>(*activationDer)(Matrix<double>& Z),MatrixInit init= MatrixInit::ZERO_INIT);
	int GetOutDim() { return Output_Dim; }
	int GetInDim() { return Input_Dim; }
	Matrix<double> ActivationPrime(Matrix<double> Z);
	Matrix<double> ApplyActivation(Matrix<double> Z);
	Matrix<double> Mul(Matrix<double>& A);
private:
	int Output_Dim;
	int Input_Dim;
	Matrix<double> Weights;
	Matrix<double> Biases;
	Matrix<double> (* Activation)(Matrix<double>& ) = nullptr;
	Matrix<double> (*ActivationDer)(Matrix<double>& ) = nullptr;
};

