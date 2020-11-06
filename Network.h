#include "DenseLayer.h"

class Network
{
public:
	Network(double LearningRate);
	void AddLayer(DenseLayer layer);
	std::vector< Matrix<double>> Predict(std::vector<Matrix<double>> Input);
private:
	float Learning_Rate;
	std::vector<DenseLayer> Layers;
};
