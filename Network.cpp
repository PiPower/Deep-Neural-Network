#include "Network.h"
#include <algorithm>
#include <iostream>
#include <stack>
using namespace std;

Network::Network()
{
}


void Network::AddLayer(DenseLayer* layer)
{
	if(Layers.size() > 0 ) assert(Layers[Layers.size() - 1]->GetOutDim() == layer->GetInDim());
	Layers.push_back(layer);
}

std::vector<Matrix<double>>  Network::Predict(const std::vector<Matrix<double>>& Input)
{
	std::vector<Matrix<double>> Predictions;
	for (int i = 0; i < Input.size(); i++)
	{
		Predictions.push_back(Input[i]);
		for (auto layer : Layers)
		{
			Predictions[i] = layer->Mul(Predictions[i]);
			Predictions[i] = layer->ApplyActivation(Predictions[i]);
		}
	}
	return Predictions;
}

void Network::SetCostFun(CostFunction* CostFunc_)
{
	CostFunc = CostFunc_;
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
	assert(TrainingData.size() == TrainingLabels.size() && CostFunc != nullptr  && LearningRate > 0);
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
				NablaWeights.push_back( Matrix<double>(Layers[i]->GetInDim(), Layers[i]->GetOutDim()) );
				NablaBiases.push_back( Matrix<double>(1, Layers[i]->GetOutDim()) );
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
				
				Layers[g]->UpdateWeights( NablaWeights[g] * (-LearningRate/BatchSize)  );
				Layers[g]->UpdateBiases( NablaBiases[g] * (-LearningRate/BatchSize) );
			}

		}
	}
}

Network::~Network()
{
	if (CostFunc != nullptr)
	{
		delete CostFunc;
		CostFunc = nullptr;
	}
	for (auto layer : Layers)
	{
		delete layer;
		layer = nullptr;
	}
	Layers.clear();
}

std::pair<MatrixD_Array, MatrixD_Array> Network::BackPropagation(Matrix<double>& Training_Data,Matrix<double>& label)
{
	// typedef vector<Matrix<double>>
	MatrixD_Array NablaWeight;
	MatrixD_Array NablaBias;

	for (int i = 0; i < Layers.size(); i++)
	{
		NablaWeight.push_back(Matrix<double>(Layers[i]->GetInDim(), Layers[i]->GetOutDim()));
		NablaBias.push_back(Matrix<double>(1, Layers[i]->GetOutDim()));
	}
	

	std::stack<Matrix<double>> ActivationOutput; // output after activation function 
	std::stack<Matrix<double>> Z_Output;// output after matrix operation 

	ActivationOutput.push( Training_Data);
	// Pushing values through
	for (int index =0 ; index<Layers.size();index++)
	{
		Z_Output.push( Layers[index]->Mul(ActivationOutput.top()) ); // matrix operation w*a+b
		ActivationOutput.push( Layers[index]->ApplyActivation(Z_Output.top() )  ); // aplaying activation function 
	}	

	auto zs = Layers[Layers.size() - 1]->ActivationPrime(Z_Output.top());
	Z_Output.pop();
	Matrix<double> Delta_L = Hadamard(CostFunc->Function_Der(ActivationOutput.top(),label), zs); // Delta of last layer 
	ActivationOutput.pop();


	NablaWeight[Layers.size() - 1] = Delta_L * ActivationOutput.top().Transpose();
	ActivationOutput.pop();
	NablaBias[Layers.size() - 1] = Delta_L;

	// loop of calculating delta based on earlier delta and setting weights 
	for (int j = 2; j <= Layers.size() ; j++)
	{
		auto sp = Layers[Layers.size() - j]->ActivationPrime(Z_Output.top());
		Z_Output.pop();
		Delta_L = Hadamard(Layers[Layers.size() -  j+1 ]->GetWeights().Transpose() * Delta_L, sp);

		NablaWeight[Layers.size() - j] = Delta_L * ActivationOutput.top().Transpose();
		ActivationOutput.pop();
		NablaBias[Layers.size() - j] = Delta_L;
	}

	return make_pair(NablaWeight, NablaBias);
}
