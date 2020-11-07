#include "Network.h"

using namespace std;

Network::Network()
{
}

void Network::AddLayer(DenseLayer layer)
{
	if(Layers.size() > 0 ) assert(Layers[Layers.size() - 1].GetOutDim() == layer.GetInDim());
	Layers.push_back(layer);
}

std::vector<Matrix<double>>  Network::Predict(std::vector<Matrix<double>> Input)
{
	for (int i = 0; i < Input.size(); i++)
	{
		for (auto layer : Layers)
		{
			Input[i] = layer.Mul(Input[i]);
		}
	}
	return Input;
}

void Network::SetCostFun(Matrix<double>(*CostFunc_)(Matrix<double> A, Matrix<double> Y), Matrix<double>(*CostFuncDer_)(Matrix<double> A, Matrix<double> Y))
{
	CostFunc = CostFunc_;
	CostFuncDer = CostFuncDer_;
}



/*std::vector<Matrix<double>> Network::Predict_Training(std::vector<Matrix<double>> Input, std::vector<long unsigned int> indexes,)
{
	for (int i = 0; i < Input.size(); i++)
	{
		for (auto layer : Layers)
		{
			Input[i] = layer.Mul(Input[i]);
		}
	}
	return Input;
}*/


void Network::Train(std::vector<Matrix<double>> TrainingData, std::vector<Matrix<double>> TrainingLabels, int BatchSize,int epochs, double LearningRate)
{
	assert(TrainingData.size() == TrainingLabels.size() && CostFunc != nullptr && CostFuncDer != nullptr);
	std::vector<long unsigned int > indexes;
	for (int i = 0; i < TrainingData.size(); i++)  indexes.push_back(i);

	std::random_device rd;
	std::mt19937 g(rd());

	vector<pair<Matrix<double>, Matrix<double> > > Gradient;

	for (int i = 0; i < Layers.size(); i++)
	{
		Matrix<double> GradientWeight(Layers[i].GetInDim(), Layers[i].GetOutDim());
		Matrix<double> GradientBias(1, Layers[i].GetOutDim());
		Gradient.push_back(make_pair(GradientWeight, GradientBias));
	}

	for (int i = 0; i < epochs; i++)
	{
		shuffle(indexes.begin(), indexes.end(), g);
		memset(Gradient.data(), 0, sizeof(pair< Matrix<double>, Matrix<double> >) * Gradient.size());

		// Batch iterating 
		for (int batch = 0; batch < TrainingData.size(); batch = batch + BatchSize)
		{
			// Per Batch iterating

			for (int i = 0; i < BatchSize; i++)
			{
				std::vector<Matrix<double>> OutputsA;
				std::vector<Matrix<double>> OutputsZ;
				OutputsA.resize(Layers.size() + 1);
				OutputsZ.resize(Layers.size());

				OutputsA[i] = TrainingData[i + batch];
				int index = 0;
				for (auto layer : Layers)
				{
					OutputsZ[index] = layer.Mul(OutputsA[index]);
					OutputsA[index + 1] = layer.ApplyActivation(OutputsZ[index]);
					index++;
				}
                
				Matrix<double> Delta_L = Hadamard(CostFuncDer(OutputsA[OutputsA.size() - 1], TrainingLabels[i + batch]), Layers[Layers.size() - 1].ActivationPrime(OutputsZ[OutputsZ.size() - 1]));
			}


		}
	}
}
