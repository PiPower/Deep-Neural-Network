#include "ConvLayer.h"
#include <vector>

using namespace std;

ConvLayer::ConvLayer(int Input_dim, int Output_dim, Image_Dim Image_dim, ActivationFunction* Func_, MatrixInit init, WeightNormalization W_Init)
	:
Input_Dim(Input_dim),Output_Dim(Output_dim),Func(Func_), img_dim(Image_dim)
{
	int index = 0;
	std::normal_distribution<double> unif(0, 1);
	std::random_device rd;
	std::mt19937_64 gen(rd());
	Bias = Matrix<double>(1, Output_Dim, init, &gen, &unif);


	for (int i = 0; i < Output_Dim;i++)
	{
		Kernel Ker;
		for (int j = 0; j < Input_Dim; j++)
		{

			Matrix<double> Weights(Image_dim.Width, Image_dim.Height,init, &gen, &unif);

			switch (W_Init)
			{
			case WeightNormalization::RoI:
				Weights = Weights * (1.0 / sqrt(Input_dim));
				break;
			case WeightNormalization::DoubleRoI:
				Weights = Weights * (2.0 / sqrt(Input_dim));
				break;
			default:
				break;
			}

			Ker.push_back(Weights);
		}
		Kernels.push_back(Ker);
    }

}

std::vector<Matrix<double>> ConvLayer::ActivationPrime(Images Z)
{
	return std::vector<Matrix<double>>();
}

std::vector<Matrix<double>> ConvLayer::ApplyActivation(Images Z)
{
	return std::vector<Matrix<double>>();
}

std::vector<Matrix<double>> ConvLayer::Mul(Images& A)
{
	Images img;
	img.resize(Output_Dim);
	for (int i = 0; i < Output_Dim;  i++)
	{
		img[i] = Matrix<double>(img_dim.Width, img_dim.Height);
		for (int j = 0; j < Input_Dim; j++)
		{
			img[i] += Matrix<double>::Convolution(A[j], Kernels[i][j]);
		}

	}
	return img;

}
