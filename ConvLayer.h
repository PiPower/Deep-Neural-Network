#pragma once
#include "Layer.h"

typedef std::vector<Matrix<double>>  Kernel;
typedef std::vector<Matrix<double>>  Images;


struct Image_Dim
{
	int Height;
	int Width;
	int Nr_Channels;
};
class ConvLayer : public BaseLayer
{
public:
	ConvLayer(int Input_dim, int Output_dim, Image_Dim Image_dim, ActivationFunction* Func_, MatrixInit init = MatrixInit::ZERO_INIT, WeightNormalization W_Init = WeightNormalization::None);
	virtual std::vector<Matrix<double>> ActivationPrime(std::vector<Images > Z);
	virtual std::vector<Matrix<double>> ApplyActivation(std::vector<Images > Z);
	virtual std::vector<Matrix<double>> Mul(std::vector<Images>& A);
private:
	int Output_Dim;
	int Input_Dim;
	ActivationFunction* Func = nullptr;
	std::vector<Kernel> Kernels;
	Matrix<double> Bias;
	Image_Dim img_dim;
};

