#include "ConvLayer.h"
#include <vector>

using namespace std;

ConvLayer::ConvLayer(int Input_dim, int Output_dim, Image_Dim Image_dim, Image_Dim Kernel_Dim,ActivationFunction* Func_, MatrixInit init, WeightNormalization W_Init)
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

			Matrix<double> Weights(Kernel_Dim.Width, Kernel_Dim.Height,init, &gen, &unif);

			switch (W_Init)
			{
			case WeightNormalization::RoI:
				Weights = Weights * (1.0 / sqrt(Input_dim + Output_Dim));
				break;
			case WeightNormalization::DoubleRoI:
				Weights = Weights * (2.0 / sqrt(Input_dim+ Output_Dim));
				break;
			default:
				break;
			}

			Ker.push_back(Weights);
		}
		Kernels.push_back(Ker);
    }

}

Tensor1D ConvLayer::GetWeights()
{
	Tensor1D out;
	for (int i = 0; i < Kernels.size(); i++)
	{
		for (int j = 0; j < Kernels[i].size(); j++)
		{
			out.Append(Kernels[i][j]);
		}
	}
	return out;
}

Tensor1D ConvLayer::GetBiases()
{
	return Tensor1D(Bias);
}

std::vector<Matrix<double>> ConvLayer::ActivationPrime(Images& Z)
{
	return std::vector<Matrix<double>>();
}

std::vector<Matrix<double>> ConvLayer::ApplyActivation(Images& Z)
{
	std::vector<Matrix<double>> Out;
	Out.resize(Z.size());
	for (int i = 0;i< Z.size(); i++)
	{
		Out[i] = Func->Function(Z[i]);
	}

	return Out;
}

std::vector<Matrix<double>> ConvLayer::Mul(Images& A)
{
	Images img;
	img.resize(Output_Dim);
	for (int i = 0; i < Output_Dim;  i++)
	{
		img[i] = Matrix<double>::Convolution(A[0], Kernels[i][0]);;
		for (int j = 1; j < Input_Dim; j++)
		{
			img[i] += Matrix<double>::Convolution(A[j], Kernels[i][j]);
		}
		img[i] += Bias.GetAt(i,0);
	}
	return img;

}

std::vector<Matrix<double>> ConvLayer::GetNablaWeight()
{
	std::vector<Matrix<double>> out;
	for (int i = 0; i < Kernels.size(); i++)
	{
		for (int j = 0; j < Kernels[i].size(); j++)
		{
			out.push_back(Kernels[i][j]);
			out.back().Clear();
		}
	}
	return out;
}

std::vector<Matrix<double>> ConvLayer::GetNablaBias()
{
	Matrix<double> out = Bias;
	out.Clear();
	return vector<Matrix<double>>{out};
}

Tensor1D ConvLayer::CalculateNablaWeight(const Tensor1D& Delta, const Tensor1D& Activation)
{
	Tensor1D out;
	
	return Tensor1D::Convolution(Activation, Delta);
}

Tensor1D ConvLayer::CalculateNablaBias(const Tensor1D& Delta, const Tensor1D& Activation)
{
	return Tensor1D(Matrix<double>(1, Output_Dim));
}
