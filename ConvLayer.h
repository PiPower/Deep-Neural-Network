#pragma once
#include "Layer.h"

typedef std::vector<Matrix<float>>  Kernel;
typedef std::vector<Matrix<float>>  Images; // every matrix in vector is one Channel


struct Image_Dim
{
	int Height;
	int Width;
};

typedef Image_Dim Kernel_Dim;

class ConvLayer : public BaseLayer
{
public:
	ConvLayer(int Input_dim, int Output_dim, Image_Dim Image_dim, Image_Dim Kernel_Dim,ActivationFunction* Func_, MatrixInit init = MatrixInit::ZERO_INIT, WeightNormalization W_Init = WeightNormalization::None);
	Tensor1D GetWeights();
	std::vector<Kernel> GetWeights_v2();
	Tensor1D GetBiases();
	std::vector<Matrix<float>> ActivationPrime(Images& Z) override;
	std::vector<Matrix<float>> ApplyActivation(Images& Z) override;
    std::vector<Matrix<float>> Mul(Images& A) override;
	std::vector<Matrix<float>> GetNablaWeight() override;
	std::vector<Matrix<float>> GetNablaBias() override;
	int GetOutDim() override;
	Kernel_Dim GetKernelDim() const;
	Tensor1D CalculateNablaWeight(Tensor1D& Delta, Tensor1D& Activation) override;
	Tensor1D CalculateNablaBias( Tensor1D& Delta, Tensor1D& Activation) override;
	Tensor1D static Convolve_Backprop(Images& A, std::vector<Kernel>& Kernel, int OutChannels, Tensor1D Z_Out);
private:
	int Output_Dim;
	int Input_Dim;
	ActivationFunction* Func = nullptr;
	std::vector<Kernel> Kernels;
	Matrix<float> Bias;
	Kernel_Dim krn_dim;
	Image_Dim img_dim;
};

