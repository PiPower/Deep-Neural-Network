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
    return Tensor1D( Matrix<float>());
}

 Tensor1D BaseLayer::GetBiases()
{
    return Tensor1D(Matrix<float>());
}

void BaseLayer::UpdateWeights(const std::vector<Matrix<float>>& Weights_, const float& Eta)
{
}

void BaseLayer::UpdateBiases(const std::vector<Matrix<float>>& Biases_, const float& Eta)
{
}


std::vector<Matrix<float>> BaseLayer::ActivationPrime(std::vector<Matrix<float>>& Z)
{
    return Z;
}

std::vector<Matrix<float>> BaseLayer::ApplyActivation(std::vector<Matrix<float>>& Z)
{
    return Z;
}

std::vector<Matrix<float>> BaseLayer::Mul(std::vector<Matrix<float>>& A)
{
    return A;
}

std::vector<Matrix<float>> BaseLayer::GetNablaWeight()
{
    return std::vector<Matrix<float>>();
}

std::vector<Matrix<float>> BaseLayer::GetNablaBias()
{
    return std::vector<Matrix<float>>();
}

Tensor1D BaseLayer::CalculateNablaWeight(const Tensor1D& Delta, const Tensor1D& Activation)
{
    return Tensor1D();
}

Tensor1D BaseLayer::CalculateNablaBias(const Tensor1D& Delta, const Tensor1D& Activation)
{
    return Tensor1D();
}


