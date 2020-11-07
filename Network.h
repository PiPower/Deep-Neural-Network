#include "DenseLayer.h"

class Network
{
public:
	Network();
	void AddLayer(DenseLayer layer);
	std::vector< Matrix<double>> Predict(std::vector<Matrix<double>> Input);
	void SetCostFun(Matrix<double> (*CostFunc_)(Matrix<double> A, Matrix<double> Y), Matrix<double>(*CostFuncDer_)(Matrix<double> A, Matrix<double> Y));
	void Train(std::vector<Matrix<double>> TrainingData, std::vector<Matrix<double>> TrainingLabels, int BatchSize,int epochs, double LearningRate );
private:
	std::vector<DenseLayer> Layers;
	Matrix<double>(*CostFunc)(Matrix<double> A, Matrix<double> Y);
	Matrix<double>(*CostFuncDer)(Matrix<double> A, Matrix<double> Y);
};
