#pragma once
#include "Matrix.h"

class DenseLayer
{
public:
	DenseLayer(int input_dim, int output_dim,double  (*activation)(double Z) ,MatrixInit init= MatrixInit::ZERO_INIT);
	Matrix<double> Mul(Matrix<double> A);
private:
	Matrix<double> Weights;
	Matrix<double> Biases;
	double  (* Activation)(double Z) = nullptr;
};

