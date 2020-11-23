#pragma once
#include "Matrix.h"
#include "Functions.h"

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
	virtual const Matrix<double>& GetWeights();
	virtual const Matrix<double>& GetBiases();
	virtual void UpdateWeights(const Matrix<double>& Weights_);
	virtual void UpdateBiases(const Matrix<double>& Biases_);
	// --------------------------------------------------------
	//virtual Matrix<double> ActivationPrime(Matrix<double> Z);
	//virtual Matrix<double> ApplyActivation(Matrix<double> Z);
	//virtual Matrix<double> Mul(Matrix<double>& A);
	// --------------------------------------------------------
	virtual std::vector<Matrix<double>> ActivationPrime(std::vector<Matrix<double> > Z);
	virtual std::vector<Matrix<double>> ApplyActivation(std::vector<Matrix<double> > Z);
	virtual std::vector<Matrix<double>> Mul(std::vector<Matrix<double>>& A);
};