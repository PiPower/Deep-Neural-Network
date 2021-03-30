#include "Network.h"
#include <algorithm>
#include <iostream>
#include <stack>
#include <future>
#include "Flatern.h"
#include <thread>




using namespace std;


Network::Network()
{
}


void Network::AddLayer(BaseLayer* layer)
{
	//if(Layers.size() > 0 ) assert(Layers[Layers.size() - 1]->GetOutDim() == layer->GetInDim());
	Layers.push_back(layer);
}

std::vector<Matrix<float>>  Network::Predict(const std::vector<Matrix<float>>& Input)
{
	std::vector<Matrix<float>> Predictions;
	std::vector<Matrix<float>> Buffor;
	for (int i = 0; i < Input.size(); i++)
	{
		std::vector<Matrix<float>> Buffor;
		Buffor.push_back( Input[i] );
		for (auto layer : Layers)
		{
			Buffor = layer->Mul(Buffor) ;
			Buffor = layer->ApplyActivation(Buffor);
		}

		Predictions.push_back(Buffor.back());
	}
	return Predictions;
}

void Network::SetCostFun(CostFunction* CostFunc_)
{
	CostFunc = CostFunc_;
}



/*std::vector<Matrix<float>> Network::Predict_Training(std::vector<Matrix<float>> Input, std::vector<long unsigned int> indexes,)
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


void Network::Train(MatrixD_Array& TrainingData, MatrixD_Array& TrainingLabels, int BatchSize,int epochs, float LearningRate)
{
	assert(TrainingData.size() == TrainingLabels.size() && CostFunc != nullptr  && LearningRate > 0);
	std::vector<long unsigned int > indexes;
	for (int i = 0; i < TrainingData.size(); i++)  indexes.push_back(i);

	std::random_device rd;
	std::mt19937 g(rd());


	Tensor2D NablaWeights;
	Tensor2D NablaBiases;

	for (int i = 0; i < Layers.size(); i++)
	{
		NablaWeights.Append(Layers[i]->GetNablaWeight());
		NablaBiases.Append(Layers[i]->GetNablaBias());
	}


	//---- Epoch iterating
	for (int i = 0; i < epochs; i++)
	{
		cout << "Epoch number: " << i << endl;
		shuffle(indexes.begin(), indexes.end(), g);

		// Batch iterating 
		for (int batch = 0; batch < TrainingData.size(); batch = batch + BatchSize)
		{
           if((batch / BatchSize )%50==0)	cout <<"Batch nr :" <<batch/ BatchSize << endl;

			NablaWeights.Clear();
			NablaBiases.Clear();

			int i = 0;
			//Per Batch iterating
			while( i < BatchSize && (i+batch)< TrainingData.size())
			{
					
				//auto T_Data = TrainingData[indexes[i + batch + 1]];
				//auto T_Label = TrainingLabels[indexes[i + batch + 1]];
				//future<std::pair<Tensor2D, Tensor2D >> f1 = async([this, &T_Data, &T_Label]
				//() {return this->BackPropagation(T_Data, T_Label); });

                // ---- Calculating Nabla that will be later devided by batch size element wise  batch* BatchSize
				auto DeltaNabla = BackPropagation(TrainingData[indexes[i + batch ]], TrainingLabels[indexes[i + batch]]);

				NablaWeights = NablaWeights + DeltaNabla.first;
				NablaBiases = NablaBiases + DeltaNabla.second;

				i++;
			}

			int x = 0;
			for (int g = 0; g < Layers.size(); g++)
			{
				
				Layers[g]->UpdateWeights( NablaWeights[g].GetTensor() ,-LearningRate/BatchSize  );
				Layers[g]->UpdateBiases( NablaBiases[g].GetTensor() ,-LearningRate/BatchSize );
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

std::pair<Tensor2D, Tensor2D > Network::BackPropagation(Matrix<float>& Training_Data,Matrix<float>& label)
{
	std::stack<Tensor1D> ActivationOutput; // output after activation function 
	std::stack<Tensor1D> Z_Output;// output after matrix operation 


	//---------------------- Forward Pass
	ActivationOutput.push( vector<Matrix<float>>{Training_Data} );
	
	for (int index =0 ; index<Layers.size();index++)
	{
		Z_Output.push( Layers[index]->Mul(ActivationOutput.top().GetTensor() ) ); // matrix operation w*a+b
		ActivationOutput.push( Layers[index]->ApplyActivation(Z_Output.top().GetTensor() )  ); // aplaying activation function 
	}	

	//-------------------------------- Creation of gradient
	Tensor2D NablaWeight;
	Tensor2D NablaBias;

	for (int i = 0; i < Layers.size(); i++)
	{
		NablaWeight.Append(Layers[i]->GetNablaWeight());
		NablaBias.Append(Layers[i]->GetNablaBias());
	}

	//------------------------------ Calculating gradient
	auto zs = Layers[Layers.size() - 1]->ActivationPrime(Z_Output.top().GetTensor());

	Tensor1D Derr{ CostFunc->Function_Der(ActivationOutput.top().GetTensor()[0],label) };
	Z_Output.pop();

	Tensor1D Delta_L = Tensor1D::Tensor1DHadamard(Derr, zs); // Delta of last layer 
	ActivationOutput.pop();

	Tensor1D Activation = ActivationOutput.top();
	Activation.Transpose();
	NablaWeight[Layers.size() - 1] = Delta_L * Activation;
	ActivationOutput.pop();

	NablaBias[Layers.size() - 1] = Delta_L;

	// loop of calculating delta based on earlier delta and setting weights 
	for (int j = 2; j <= Layers.size() ; j++)
	{
		// ----------- gradient for Flatern 
		if (dynamic_cast<Flatern*>(Layers[Layers.size() - j]))
		{
			Tensor1D sp = Layers[Layers.size() - j]->ActivationPrime(Z_Output.top().GetTensor());
			Z_Output.pop();
			Tensor1D Weights = Layers[Layers.size() - j + 1]->GetWeights();
			Weights.Transpose();


			Delta_L = Tensor1D::Tensor1DHadamard(Weights * Delta_L, sp);
			auto NablaTensor = NablaWeight[Layers.size() - j].GetTensor()[0];
			NablaWeight[Layers.size() - j] = Tensor1D();
			NablaBias[Layers.size() - j] = Tensor1D();
			Delta_L = Delta_L.ReshapeFlat(NablaTensor.GetColumns(), NablaTensor.GetRows());
			ActivationOutput.pop();
		}
		// ----------- gradient for Dense 
		if (dynamic_cast<DenseLayer*>(Layers[Layers.size() - j]))
		{
			Tensor1D sp = Layers[Layers.size() - j]->ActivationPrime(Z_Output.top().GetTensor());
			Z_Output.pop();
			Tensor1D Weights = Layers[Layers.size() - j + 1]->GetWeights();
			Weights.Transpose();

			Delta_L = Tensor1D::Tensor1DHadamard(Weights * Delta_L, sp);
			Activation = ActivationOutput.top();
			Activation.Transpose();

			NablaWeight[Layers.size() - j] = Layers[Layers.size() - j]->CalculateNablaWeight(Delta_L ,Activation);
			ActivationOutput.pop();
			NablaBias[Layers.size() - j] = Layers[Layers.size() - j]->CalculateNablaBias(Delta_L, Activation);
		}
		// ----------- gradient for Conv 
		if (auto Layer = dynamic_cast<ConvLayer*>(Layers[Layers.size() - j]))
		{

			//Tensor1D sp = Layers[Layers.size() - j]->ActivationPrime(Z_Output.top().GetTensor());
			
			//Tensor1D Weights = Layers[Layers.size() - j + 1]->GetWeights();
			//Weights.Transpose();

			/*if (!dynamic_cast<Flatern*>(Layers[Layers.size() - j + 1]))
			{
				Delta_L = Tensor1D::Tensor1DHadamard(Weights * Delta_L, sp);
			}*/

			if (auto Layer_Next = dynamic_cast<ConvLayer*>(Layers[Layers.size() - j + 1]))
			{
				Kernel_Dim krn_dim = Layer_Next->GetKernelDim();
				Delta_L.AddPadding(krn_dim.Height-1, krn_dim.Width - 1);
				std::vector<Kernel> Kernel = Layer_Next->GetWeights_v2();
				Delta_L = ConvLayer::Convolve_Backprop(Delta_L.GetTensor() , Kernel,Layer->GetOutDim(), Z_Output.top() );
			}

			Z_Output.pop();
			Activation = ActivationOutput.top();

			NablaWeight[Layers.size() - j] = Layer->CalculateNablaWeight(Delta_L, Activation);
			ActivationOutput.pop();
			NablaBias[Layers.size() - j] = Layer->CalculateNablaBias(Delta_L, Activation);
		}
	}

	return make_pair(NablaWeight, NablaBias);
}
