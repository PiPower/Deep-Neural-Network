#pragma once
#include "Matrix.h"
#include "Functions.h"
#include "Tensor1D.h"

class BaseLayer
{
public: 
	enum class WeightNormalization
	{
		None,
		RoI, // Root of input 
		floatRoI // float root of input
	};
protected:
	BaseLayer() = default;
public:
	virtual int GetOutDim();
	virtual int GetInDim();
	virtual  Tensor1D GetWeights();
	virtual  Tensor1D GetBiases();
	virtual void UpdateWeights(const std::vector<Matrix<float>>& Weights_,const float& Eta);
	virtual void UpdateBiases(const std::vector<Matrix<float>>& Biases_, const float& Eta);
	virtual std::vector<Matrix<float>> ActivationPrime(std::vector<Matrix<float> >& Z);
	virtual std::vector<Matrix<float>> ApplyActivation(std::vector<Matrix<float> >& Z);
	virtual std::vector<Matrix<float>> Mul(std::vector<Matrix<float>>& A);
	virtual  std::vector<Matrix<float>> GetNablaWeight();
	virtual  std::vector<Matrix<float>> GetNablaBias();
	virtual Tensor1D CalculateNablaWeight(const Tensor1D& Delta, const Tensor1D& Activation);
	virtual Tensor1D CalculateNablaBias(const Tensor1D& Delta, const Tensor1D& Activation);
};