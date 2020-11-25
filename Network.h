#pragma once
#include "DenseLayer.h"
#include "CostFunction.h"
#include "Tensor2D.h"

typedef std::vector< Matrix<double>> MatrixD_Array;

class Network
{
	const double M_E = 2.71828182845904523536;
public:
	Network();
	void AddLayer(BaseLayer* layer);
	MatrixD_Array Predict(const std::vector<Matrix<double>>& Input);
	void SetCostFun(CostFunction* CostFunc_);
	void Train(MatrixD_Array& TrainingData, MatrixD_Array& TrainingLabels, int BatchSize,int epochs, double LearningRate );
	~Network();
private:
	std::pair<Tensor2D, Tensor2D >  BackPropagation(Matrix<double>& Training_Data, Matrix<double>& label);
private:
	std::vector<BaseLayer*> Layers;
	CostFunction* CostFunc = nullptr;
};
