 #pragma once
#include "Matrix.h"
#include "Functions.h"
#include "Layer.h"
class DenseLayer : public BaseLayer
{
public:
	DenseLayer(int input_dim, int output_dim, ActivationFunction* Func_, MatrixInit init = MatrixInit::ZERO_INIT, WeightNormalization W_Init= WeightNormalization::None);
	int GetOutDim() { return Output_Dim; }
	int GetInDim() { return Input_Dim; }
	const Matrix<double>& GetWeights(){return Weights;}
	const Matrix<double>& GetBiases() { return Biases; }
	void UpdateWeights(const Matrix<double>& Weights_);
	void UpdateBiases(const Matrix<double>& Biases_);
	Matrix<double> ActivationPrime(Matrix<double> Z);
	Matrix<double> ApplyActivation(Matrix<double> Z);
	Matrix<double> Mul(Matrix<double>& A);
	~DenseLayer();
private:
	int Output_Dim;
	int Input_Dim;
	Matrix<double> Weights;
	Matrix<double> Biases;
	ActivationFunction* Func = nullptr;
};

