#pragma once
#include "Matrix.h"

class DenseLayer
{
public:
	DenseLayer(int input_dim, int output_dim,double  (*activation)(double Z) ,MatrixInit init= MatrixInit::ZERO_INIT);
	int GetOutDim() { return Output_Dim; }
	int GetInDim() { return Input_Dim; }
	Matrix<double> Mul(Matrix<double> A);
private:
	int Output_Dim;
	int Input_Dim;
	Matrix<double> Weights;
	Matrix<double> Biases;
	double  (* Activation)(double Z) = nullptr;
};

