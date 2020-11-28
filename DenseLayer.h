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
	void UpdateWeights(const std::vector<Matrix<double>>& Weights_, const double& Eta);
	void UpdateBiases(const std::vector<Matrix<double>>& Biases_, const double& Eta);
	std::vector<Matrix<double>> ActivationPrime(std::vector<Matrix<double>>& Z);
	std::vector<Matrix<double>> ApplyActivation(std::vector<Matrix<double>>& Z);
	std::vector<Matrix<double>> Mul(std::vector<Matrix<double>>& A);
	std::vector<Matrix<double>> GetNablaWeight();
	std::vector<Matrix<double>> GetNablaBias();
	Tensor1D CalculateNablaWeight(const Tensor1D& Delta, const Tensor1D& Activation);
	Tensor1D CalculateNablaBias(const Tensor1D& Delta, const Tensor1D& Activation);
	~DenseLayer();
private:
	int Output_Dim;
	int Input_Dim;
	Matrix<double> Weights;
	Matrix<double> Biases;
	ActivationFunction* Func = nullptr;
};

