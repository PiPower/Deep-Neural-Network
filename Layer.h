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
		DoubleRoI // Double root of input
	};
protected:
	BaseLayer() = default;
public:
	virtual int GetOutDim();
	virtual int GetInDim();
	virtual  Tensor1D GetWeights();
	virtual  Tensor1D GetBiases();
	virtual void UpdateWeights(const std::vector<Matrix<double>>& Weights_,const double& Eta);
	virtual void UpdateBiases(const std::vector<Matrix<double>>& Biases_, const double& Eta);
	virtual std::vector<Matrix<double>> ActivationPrime(std::vector<Matrix<double> >& Z);
	virtual std::vector<Matrix<double>> ApplyActivation(std::vector<Matrix<double> >& Z);
	virtual std::vector<Matrix<double>> Mul(std::vector<Matrix<double>>& A);
	virtual  std::vector<Matrix<double>> GetNablaWeight();
	virtual  std::vector<Matrix<double>> GetNablaBias();
	virtual Tensor1D CalculateNablaWeight(const Tensor1D& Delta, const Tensor1D& Activation);
	virtual Tensor1D CalculateNablaBias(const Tensor1D& Delta, const Tensor1D& Activation);
};