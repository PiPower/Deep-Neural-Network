#include "Layer.h"

int BaseLayer::GetOutDim()
{
    return 0;
}

int BaseLayer::GetInDim()
{
    return 0;
}

 Tensor1D BaseLayer::GetWeights()
{
    return Tensor1D( Matrix<double>());
}

 Tensor1D BaseLayer::GetBiases()
{
    return Tensor1D(Matrix<double>());
}

void BaseLayer::UpdateWeights(const std::vector<Matrix<double>>& Weights_, const double& Eta)
{
}

void BaseLayer::UpdateBiases(const std::vector<Matrix<double>>& Biases_, const double& Eta)
{
}


std::vector<Matrix<double>> BaseLayer::ActivationPrime(std::vector<Matrix<double>>& Z)
{
    return Z;
}

std::vector<Matrix<double>> BaseLayer::ApplyActivation(std::vector<Matrix<double>>& Z)
{
    return Z;
}

std::vector<Matrix<double>> BaseLayer::Mul(std::vector<Matrix<double>>& A)
{
    return A;
}

std::vector<Matrix<double>> BaseLayer::GetNablaWeight()
{
    return std::vector<Matrix<double>>();
}

std::vector<Matrix<double>> BaseLayer::GetNablaBias()
{
    return std::vector<Matrix<double>>();
}
