#include "Layer.h"

int BaseLayer::GetOutDim()
{
    return 0;
}

int BaseLayer::GetInDim()
{
    return 0;
}

const Matrix<double>& BaseLayer::GetWeights()
{
    return Matrix<double>();
}

const Matrix<double>& BaseLayer::GetBiases()
{
    return Matrix<double>();
}

void BaseLayer::UpdateWeights(const Matrix<double>& Weights_)
{
}

void BaseLayer::UpdateBiases(const Matrix<double>& Biases_)
{
}


std::vector<Matrix<double>> BaseLayer::ActivationPrime(std::vector<Matrix<double>> Z)
{
    return Z;
}

std::vector<Matrix<double>> BaseLayer::ApplyActivation(std::vector<Matrix<double>> Z)
{
    return Z;
}

std::vector<Matrix<double>> BaseLayer::Mul(std::vector<Matrix<double>>& A)
{
    return A;
}
