#pragma once
#include "DenseLayer.h"
#include "CostFunction.h"
#include "Tensor2D.h"
#include "ConvLayer.h"

typedef std::vector< Matrix<float>> MatrixD_Array;

class Network
{
	const float M_E = 2.71828182845904523536;
public:
	Network();
	void AddLayer(BaseLayer* layer);
	MatrixD_Array Predict(const std::vector<Matrix<float>>& Input);
	void SetCostFun(CostFunction* CostFunc_);
	void Train(MatrixD_Array& TrainingData, MatrixD_Array& TrainingLabels, int BatchSize,int epochs, float LearningRate );
	~Network();
private:
	std::pair<Tensor2D, Tensor2D >  BackPropagation(Matrix<float>& Training_Data, Matrix<float>& label);
private:
	std::vector<BaseLayer*> Layers;
	CostFunction* CostFunc = nullptr;
};
