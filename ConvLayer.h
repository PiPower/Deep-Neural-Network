#pragma once
#include "Layer.h"

typedef std::vector<Matrix<double>>  Kernel;
typedef std::vector<Matrix<double>>  Images; // every matrix in vector is one Channel


struct Image_Dim
{
	int Height;
	int Width;
};
class ConvLayer : public BaseLayer
{
public:
	ConvLayer(int Input_dim, int Output_dim, Image_Dim Image_dim, ActivationFunction* Func_, MatrixInit init = MatrixInit::ZERO_INIT, WeightNormalization W_Init = WeightNormalization::None);
	std::vector<Matrix<double>> ActivationPrime(Images Z) override;
	std::vector<Matrix<double>> ApplyActivation(Images Z) override;
    std::vector<Matrix<double>> Mul(Images& A) override;
private:
	int Output_Dim;
	int Input_Dim;
	ActivationFunction* Func = nullptr;
	std::vector<Kernel> Kernels;
	Matrix<double> Bias;
	Image_Dim img_dim;
};

