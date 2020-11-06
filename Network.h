#include "DenseLayer.h"

class Network
{
public:
	Network(double LearningRate);
	void AddLayer(DenseLayer layer);
private:
	std::vector<DenseLayer> Layers
};
