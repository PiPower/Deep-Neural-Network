#include "Network.h"
#include <algorithm>
#include <iostream>
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


void Network::Train(MatrixD_Array TrainingData, MatrixD_Array TrainingLabels, int BatchSize,int epochs, double LearningRate)
{
	assert(TrainingData.size() == TrainingLabels.size() && CostFunc != nullptr && CostFuncDer != nullptr && LearningRate > 0);
	std::vector<long unsigned int > indexes;
	for (int i = 0; i < TrainingData.size(); i++)  indexes.push_back(i);

	std::random_device rd;
	std::mt19937 g(rd());

	std::vector<Matrix<double>> NablaWeights;
	std::vector<Matrix<double>> NablaBiases;


	NablaWeights.resize(Layers.size());
	NablaBiases.resize(Layers.size());


	for (int i = 0; i < Layers.size(); i++)
	{
		NablaWeights[i] = Matrix<double>(Layers[i].GetInDim(), Layers[i].GetOutDim());
		NablaBiases[i] = Matrix<double>(1, Layers[i].GetOutDim());
	}

	//---- Epoch iterating
	for (int i = 0; i < epochs; i++)
	{
		cout << "Epoch number: " << i << endl;
		shuffle(indexes.begin(), indexes.end(), g);

		// Batch iterating 
		for (int batch = 0; batch < TrainingData.size(); batch = batch + BatchSize)
		{
			for (int i = 0; i < Layers.size(); i++)
			{
				NablaWeights[i].Clear();
				NablaBiases[i].Clear();
			}


			int i = 0;
			while( i < BatchSize && (i+batch)< TrainingData.size())
			{
				
				std::vector<Matrix<double>> OutputsA;
				std::vector<Matrix<double>> OutputsZ;
				OutputsA.resize(Layers.size() + 1);
				OutputsZ.resize(Layers.size());

				OutputsA[0] = TrainingData[indexes[i + batch]];
				int index = 0;
				for (auto layer : Layers)
				{
					OutputsZ[index] = layer.Mul(OutputsA[index]);
					OutputsA[index + 1] = layer.ApplyActivation(OutputsZ[index]);
					index++;
				}
                // ---- Calculating Nabla that will be later devided by batch size element wise 
				auto DeltaNabla = BackPropagation(OutputsA, OutputsZ, TrainingLabels[indexes[i + batch]]);

				for (int i = 0; i < Layers.size(); i++)
				{
					NablaWeights[i] = NablaWeights[i] + DeltaNabla.first[i];
					NablaBiases[i] = NablaBiases[i] + DeltaNabla.second[i];
				}
			i++;
			}

			for (int g = 0; g < Layers.size(); g++)
			{
				
				Layers[g].Weights = Layers[g].Weights - NablaWeights[g] * LearningRate;
				Layers[g].Biases = Layers[g].Biases - NablaBiases[g] * LearningRate;
			}
		   // std::transform(NablaWeights.begin(), NablaWeights.end(), NablaWeights.begin(),[BatchSize, LearningRate](Matrix<double>& Weight) {return Weight * (LearningRate / BatchSize);});
			//std::transform(NablaBiases.begin(), NablaBiases.end(), NablaBiases.begin(), [BatchSize, LearningRate](Matrix<double>& Bias) {return Bias * (LearningRate / BatchSize); });

		}
	}
}

std::pair<MatrixD_Array, MatrixD_Array> Network::BackPropagation( MatrixD_Array& OutputsA, MatrixD_Array& OutputsZ,Matrix<double> label)
{
	MatrixD_Array NablaWeight;
	MatrixD_Array NablaBias;

	NablaWeight.resize(Layers.size());
	NablaBias.resize(Layers.size());

	auto zs = Layers[Layers.size() - 1].ActivationPrime(OutputsZ[OutputsZ.size() - 1]);
	Matrix<double> Delta_L = Hadamard(CostFuncDer(OutputsA[OutputsA.size() - 1],label), zs);

	NablaWeight[Layers.size() - 1] = Delta_L * OutputsA[OutputsA.size() - 2].Transpose();
	NablaBias[Layers.size() - 1] = Delta_L;


	/*for (int j = 0; j < Layers.size() - 1; j++)
	{
		auto sp = Layers[Layers.size() - 2 - j].ActivationPrime(OutputsZ[Layers.size() - 2 - j]);
		Delta_L = Hadamard(Layers[Layers.size() - 1 - j].Weights.Transpose() * Delta_L, sp );
		
		NablaWeight[Layers.size() - 2 - j] = Delta_L * OutputsA[OutputsA.size() - 3 - j].Transpose();
		NablaBias[Layers.size() - 2 - j] = Delta_L;
	}*/

	for (int j = 2; j <= Layers.size() ; j++)
	{
		auto sp = Layers[Layers.size() - j].ActivationPrime(OutputsZ[Layers.size() -j]);
		Delta_L = Hadamard(Layers[Layers.size() -  j+1 ].Weights.Transpose() * Delta_L, sp);

		NablaWeight[Layers.size() - j] = Delta_L * OutputsA[OutputsA.size() -j-1].Transpose();
		NablaBias[Layers.size() - j] = Delta_L;
	}

	return make_pair(NablaWeight, NablaBias);
}
