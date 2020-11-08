#include "DenseLayer.h"

typedef std::vector< Matrix<double>> MatrixD_Array;
class Network
{
public:
	Network();
	void AddLayer(DenseLayer layer);
	MatrixD_Array Predict(std::vector<Matrix<double>> Input);
	void SetCostFun(Matrix<double> (*CostFunc_)(Matrix<double> A, Matrix<double> Y), Matrix<double>(*CostFuncDer_)(Matrix<double> A, Matrix<double> Y));
	void Train(MatrixD_Array& TrainingData, MatrixD_Array& TrainingLabels, int BatchSize,int epochs, double LearningRate );
private:
	std::pair<MatrixD_Array, MatrixD_Array> BackPropagation( MatrixD_Array& ActivationOutput, MatrixD_Array& Z_Output, Matrix<double>& label);
private:
	std::vector<DenseLayer> Layers;
	Matrix<double>(*CostFunc)(Matrix<double> A, Matrix<double> Y);
	Matrix<double>(*CostFuncDer)(Matrix<double> A, Matrix<double> Y);
};
