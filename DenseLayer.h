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
	Tensor1D GetWeights(){return Tensor1D(Weights);}
	Tensor1D GetBiases() { return Tensor1D(Biases); }
	void UpdateWeights(const std::vector<Matrix<float>>& Weights_, const float& Eta);
	void UpdateBiases(const std::vector<Matrix<float>>& Biases_, const float& Eta);
	std::vector<Matrix<float>> ActivationPrime(std::vector<Matrix<float>>& Z);
	std::vector<Matrix<float>> ApplyActivation(std::vector<Matrix<float>>& Z);
	std::vector<Matrix<float>> Mul(std::vector<Matrix<float>>& A);
	std::vector<Matrix<float>> GetNablaWeight();
	std::vector<Matrix<float>> GetNablaBias();
	Tensor1D CalculateNablaWeight(Tensor1D& Delta, Tensor1D& Activation);
	Tensor1D CalculateNablaBias(Tensor1D& Delta, Tensor1D& Activation);
	~DenseLayer();
private:
	int Output_Dim;
	int Input_Dim;
	Matrix<float> Weights;
	Matrix<float> Biases;
	ActivationFunction* Func = nullptr;
};

