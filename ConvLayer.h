#pragma once
#include "Layer.h"

typedef std::vector<Matrix<float>>  Kernel;
typedef std::vector<Matrix<float>>  Images; // every matrix in vector is one Channel


struct Image_Dim
{
	int Height;
	int Width;
};

class ConvLayer : public BaseLayer
{
public:
	ConvLayer(int Input_dim, int Output_dim, Image_Dim Image_dim, Image_Dim Kernel_Dim,ActivationFunction* Func_, MatrixInit init = MatrixInit::ZERO_INIT, WeightNormalization W_Init = WeightNormalization::None);
	Tensor1D GetWeights();
	Tensor1D GetBiases();
	std::vector<Matrix<float>> ActivationPrime(Images& Z) override;
	std::vector<Matrix<float>> ApplyActivation(Images& Z) override;
    std::vector<Matrix<float>> Mul(Images& A) override;
	std::vector<Matrix<float>> GetNablaWeight() override;
	std::vector<Matrix<float>> GetNablaBias() override;
	Tensor1D CalculateNablaWeight(const Tensor1D& Delta, const Tensor1D& Activation) override;
	Tensor1D CalculateNablaBias(const Tensor1D& Delta, const Tensor1D& Activation) override;
private:
	int Output_Dim;
	int Input_Dim;
	ActivationFunction* Func = nullptr;
	std::vector<Kernel> Kernels;
	Matrix<float> Bias;
	Image_Dim img_dim;
};

