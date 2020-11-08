#include "Network.h"
#include <algorithm>
#include <iostream>
#include <stack>
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
			Input[i]= layer.ApplyActivation(Input[i]);
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


void Network::Train(MatrixD_Array& TrainingData, MatrixD_Array& TrainingLabels, int BatchSize,int epochs, double LearningRate)
{
	assert(TrainingData.size() == TrainingLabels.size() && CostFunc != nullptr && CostFuncDer != nullptr && LearningRate > 0);
	std::vector<long unsigned int > indexes;
	for (int i = 0; i < TrainingData.size(); i++)  indexes.push_back(i);

	std::random_device rd;
	std::mt19937 g(rd());


	//---- Epoch iterating
	for (int i = 0; i < epochs; i++)
	{
		cout << "Epoch number: " << i << endl;
		shuffle(indexes.begin(), indexes.end(), g);

		// Batch iterating 
		for (int batch = 0; batch < TrainingData.size(); batch = batch + BatchSize)
		{
			std::vector<Matrix<double>> NablaWeights;
			std::vector<Matrix<double>> NablaBiases;

			for (int i = 0; i < Layers.size(); i++)
			{
				NablaWeights.push_back( Matrix<double>(Layers[i].GetInDim(), Layers[i].GetOutDim()) );
				NablaBiases.push_back( Matrix<double>(1, Layers[i].GetOutDim()) );
			}

			int i = 0;
			//Per Batch iterating
			while( i < BatchSize && (i+batch)< TrainingData.size())
			{
				
                // ---- Calculating Nabla that will be later devided by batch size element wise 
				auto DeltaNabla = BackPropagation(TrainingData[indexes[i + batch]], TrainingLabels[indexes[i + batch]]);

				for (int j = 0; j < Layers.size(); j++)
				{
					NablaWeights[j] = NablaWeights[j] + DeltaNabla.first[j];
					NablaBiases[j] = NablaBiases[j] + DeltaNabla.second[j];
				}
			    i++;
			}

			for (int g = 0; g < Layers.size(); g++)
			{
				
				Layers[g].Weights = Layers[g].Weights - NablaWeights[g] * LearningRate;
				Layers[g].Biases = Layers[g].Biases - NablaBiases[g] * LearningRate;
			}

		}
	}
}

std::pair<MatrixD_Array, MatrixD_Array> Network::BackPropagation(Matrix<double>& Training_Data,Matrix<double>& label)
{
	MatrixD_Array NablaWeight;
	MatrixD_Array NablaBias;

	for (int i = 0; i < Layers.size(); i++)
	{
		NablaWeight.push_back(Matrix<double>(Layers[i].GetInDim(), Layers[i].GetOutDim()));
		NablaBias.push_back(Matrix<double>(1, Layers[i].GetOutDim()));
	}
	

	std::stack<Matrix<double>> ActivationOutput;
	std::stack<Matrix<double>> Z_Output;

	ActivationOutput.push( Training_Data);
	// Pushing values through
	for (int index =0 ; index<Layers.size();index++)
	{
		Z_Output.push( Layers[index].Mul(ActivationOutput.top()) );
		ActivationOutput.push( Layers[index].ApplyActivation(Z_Output.top() )  );
	}

	auto zs = Layers[Layers.size() - 1].ActivationPrime(Z_Output.top());
	Z_Output.pop();
	Matrix<double> Delta_L = Hadamard(CostFuncDer(ActivationOutput.top(),label), zs);
	ActivationOutput.pop();


	NablaWeight[Layers.size() - 1] = Delta_L * ActivationOutput.top().Transpose();
	ActivationOutput.pop();
	NablaBias[Layers.size() - 1] = Delta_L;

	for (int j = 2; j <= Layers.size() ; j++)
	{
		auto sp = Layers[Layers.size() - j].ActivationPrime(Z_Output.top());
		Z_Output.pop();
		Delta_L = Hadamard(Layers[Layers.size() -  j+1 ].Weights.Transpose() * Delta_L, sp);

		NablaWeight[Layers.size() - j] = Delta_L * ActivationOutput.top().Transpose();
		ActivationOutput.pop();
		NablaBias[Layers.size() - j] = Delta_L;
	}

	return make_pair(NablaWeight, NablaBias);
}
