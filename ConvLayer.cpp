#include "ConvLayer.h"
#include <vector>

using namespace std;

ConvLayer::ConvLayer(int Input_dim, int Output_dim, Image_Dim Image_dim, Image_Dim Kernel_Dim,ActivationFunction* Func_, MatrixInit init, WeightNormalization W_Init)
	:
Input_Dim(Input_dim),Output_Dim(Output_dim),Func(Func_), img_dim(Image_dim)
{
	int index = 0;
	std::normal_distribution<float> unif(0, 1);
	std::random_device rd;
	std::mt19937_64 gen(rd());
	Bias = Matrix<float>(1, Output_Dim, init, &gen, &unif);


	for (int i = 0; i < Output_Dim;i++)
	{
		krn_dim = Kernel_Dim;
		Kernel Ker;
		for (int j = 0; j < Input_Dim; j++)
		{

			Matrix<float> Weights(Kernel_Dim.Width, Kernel_Dim.Height,init, &gen, &unif);

			switch (W_Init)
			{
			case WeightNormalization::RoI:
				Weights = Weights * (1.0 / sqrt(Input_dim + Output_Dim));
				break;
			case WeightNormalization::floatRoI:
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

std::vector<Kernel> ConvLayer::GetWeights_v2()
{
	return Kernels;
}

Tensor1D ConvLayer::GetBiases()
{
	return Tensor1D(Bias);
}

std::vector<Matrix<float>> ConvLayer::ActivationPrime(Images& Z)
{
	return std::vector<Matrix<float>>();
}

std::vector<Matrix<float>> ConvLayer::ApplyActivation(Images& Z)
{
	std::vector<Matrix<float>> Out;
	Out.resize(Z.size());
	for (int i = 0;i< Z.size(); i++)
	{
		Out[i] = Func->Function(Z[i]);
	}

	return Out;
}

std::vector<Matrix<float>> ConvLayer::Mul(Images& A)
{
	Images img;
	img.resize(Output_Dim);
	for (int i = 0; i < Output_Dim;  i++)
	{
		img[i] = Matrix<float>::Convolution(A[0], Kernels[i][0]);;
		for (int j = 1; j < Input_Dim; j++)
		{
			img[i] += Matrix<float>::Convolution(A[j], Kernels[i][j]);
		}
		img[i] += Bias.GetAt(i,0);
	}
	return img;

}

std::vector<Matrix<float>> ConvLayer::GetNablaWeight()
{
	std::vector<Matrix<float>> out;
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

std::vector<Matrix<float>> ConvLayer::GetNablaBias()
{
	Matrix<float> out = Bias;
	out.Clear();
	return vector<Matrix<float>>{out};
}

int ConvLayer::GetOutDim()
{
	return Output_Dim;
}

Kernel_Dim ConvLayer::GetKernelDim() const
{
	return krn_dim;
}

Tensor1D ConvLayer::CalculateNablaWeight( Tensor1D& Delta,  Tensor1D& Activation)
{
	Tensor1D Out;

	for (int i = 0; i < Delta.GetTensor().size(); i++)
	{
		for (int j = 0; j < Activation.GetTensor().size(); j++)
			Out.Append(Matrix<float>::Convolution( Activation.GetTensor()[j], Delta.GetTensor()[i]));
	}

	return Out;
}

Tensor1D ConvLayer::CalculateNablaBias( Tensor1D& Delta, Tensor1D& Activation)
{
	return Tensor1D(Matrix<float>(1, Output_Dim));
}

Tensor1D ConvLayer::Convolve_Backprop(Images& A, std::vector<Kernel>& Kernel, int OutChannels, Tensor1D Z_Out)
{
	Tensor1D Out;
	Matrix<float> img;
	Matrix<float> Z;

	for (int i = 0; i < OutChannels; i++)
	{
		img = Matrix<float>::Convolution(A[0], Kernel[0][i]);;
		for (int y = 1; y < Kernel.size(); y++)
		{
			img  += Matrix<float>::Convolution(A[y], Kernel[y][i]);
		}
		Out.Append(Hadamard(img, Z_Out[i]));
	}
	return Out;
}
