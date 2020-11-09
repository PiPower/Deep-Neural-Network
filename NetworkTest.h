#include <vector>
#include "Matrix.h"
#include <stack>
#include <fstream>
#include <string>
using namespace std;
class NetworkTest
{
	double  Sigmoid(double x)
	{
		return  1 / (std::pow(M_E, -x) + 1);
	}


	Matrix<double> Sigmoid(Matrix<double>& z)
	{
		for (int i = 0; i < z.GetRows(); i++)
		{
			z.SetValue(i, 0, Sigmoid(z.GetAt(i, 0)));
		}
		return z;
	}
	Matrix<double> DerivativeSigmoid(Matrix<double>& z)
	{

		for (int i = 0; i < z.GetRows(); i++)
		{
			z.SetValue(i, 0, Sigmoid(z.GetAt(i, 0)) * (1 - Sigmoid(z.GetAt(i, 0))));
		}
		return z;
	}
	Matrix<double> MSE(Matrix<double> A, Matrix<double> Y)
	{
		return (A - Y) * (A - Y);
	}

	Matrix<double> MSE_Der(Matrix<double> A, Matrix<double> Y)
	{
		return (A - Y);
	}
public:
	NetworkTest(vector<Matrix<double>> W, vector<Matrix<double>> B)
		:
		Weights(W),Biases(B)
	{
		fstream file("Weights.txt", ios::in);
		string tekst;
		vector<double> numbers;
		
			for (int i = 0; i < 2; i++)
			{

				for (int y = 0; y < Weights[i].GetRows(); y++)
				{
					for (int x = 0; x < Weights[i].GetColumns(); x++)
					{
						getline(file, tekst);
						Weights[i].SetValue(y, x, stod(tekst));
					}
				}

			}


			fstream file2("Biases.txt", ios::in);
			for (int i = 0; i < 2; i++)
			{

				for (int y = 0; y < Biases[i].GetRows(); y++)
				{
					for (int x = 0; x < Biases[i].GetColumns(); x++)
					{
						getline(file2, tekst);
						Biases[i].SetValue(y, x, stod(tekst));
					}
				}

			}
			file.close();
			file2.close();
		
	
	}
	std::vector<Matrix<double>>  Predict(std::vector<Matrix<double>> Input)
	{
		for (int i = 0; i < Input.size(); i++)
		{
			for (int k=0; k < Weights.size() ;k++)
			{
				Input[i] = Weights[k]*Input[i] + Biases[k];
				Input[i] = Sigmoid(Input[i]);
			}
		}
		return Input;
	}
	void Train(MatrixD_Array& TrainingData, MatrixD_Array& TrainingLabels, int BatchSize, int epochs, double LearningRate)
	{
		std::vector<long unsigned int > indexes;
		for (int i = 0; i < TrainingData.size(); i++)  indexes.push_back(i);

		std::random_device rd;
		std::mt19937 g(rd());


		//---- Epoch iterating
		for (int i = 0; i < epochs; i++)
		{
			cout << "Epoch number: " << i << endl;
			//shuffle(indexes.begin(), indexes.end(), g);

			// Batch iterating 
			for (int batch = 0; batch < TrainingData.size(); batch = batch + BatchSize)
			{
				std::vector<Matrix<double>> NablaWeights;
				std::vector<Matrix<double>> NablaBiases;

				for (int i = 0; i < Weights.size(); i++)
				{
					NablaWeights.push_back(Matrix<double>(Weights[i].GetColumns(), Weights[i].GetRows()));
					NablaBiases.push_back(Matrix<double>(1, Biases[i].GetRows() ));
				}

				int i = 0;
				//Per Batch iterating
				while (i < BatchSize && (i + batch) < TrainingData.size())
				{

					// ---- Calculating Nabla that will be later devided by batch size element wise 
					auto DeltaNabla = BackPropagation(TrainingData[indexes[i + batch]], TrainingLabels[indexes[i + batch]]);

					for (int j = 0; j < Weights.size(); j++)
					{
						NablaWeights[j] = NablaWeights[j] + DeltaNabla.first[j];
						NablaBiases[j] = NablaBiases[j] + DeltaNabla.second[j];
					}
					i++;
				}

				for (int g = 0; g < Weights.size(); g++)
				{

					Weights[g] = Weights[g] - NablaWeights[g] * LearningRate;
					Biases[g] = Biases[g] - NablaBiases[g] * LearningRate;
				}

			}
		}
	}

	std::pair<MatrixD_Array, MatrixD_Array> BackPropagation(Matrix<double>& Training_Data, Matrix<double>& label)
	{
		MatrixD_Array NablaWeight;
		MatrixD_Array NablaBias;

		for (int i = 0; i < Weights.size(); i++)
		{
			NablaWeight.push_back(Matrix<double>(Weights[i].GetColumns(), Weights[i].GetRows()));
			NablaBias.push_back(Matrix<double>(1, Biases[i].GetRows()));
		}


		std::stack<Matrix<double>> ActivationOutput;
		std::stack<Matrix<double>> Z_Output;

		ActivationOutput.push(Training_Data);
		// Pushing values through
		for (int index = 0; index < NablaWeight.size(); index++)
		{
			Z_Output.push(Weights[index]*ActivationOutput.top()+Biases[index]);
			ActivationOutput.push(Sigmoid(Z_Output.top()));
		}

		auto zs = DerivativeSigmoid(Z_Output.top());
		Z_Output.pop();
		auto lul = MSE_Der(ActivationOutput.top(), label);
		Matrix<double> Delta_L = Hadamard(lul, zs);
		ActivationOutput.pop();


		NablaWeight[Weights.size() - 1] = Delta_L * ActivationOutput.top().Transpose();
		ActivationOutput.pop();
		NablaBias[Weights.size() - 1] = Delta_L;

		for (int j = 2; j <= Weights.size(); j++)
		{
			auto sp = DerivativeSigmoid(Z_Output.top());
			Z_Output.pop();
			Delta_L = Hadamard(Weights[Weights.size()-j+1].Transpose() * Delta_L, sp);

			NablaWeight[Weights.size() - j] = Delta_L * ActivationOutput.top().Transpose();
			ActivationOutput.pop();
			NablaBias[Weights.size() - j] = Delta_L;
		}

		return make_pair(NablaWeight, NablaBias);
	}
private:
	vector<Matrix<double>> Weights;
	vector<Matrix<double>> Biases;
};