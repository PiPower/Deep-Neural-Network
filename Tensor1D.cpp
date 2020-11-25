#include "Tensor1D.h"
Tensor1D::Tensor1D(std::vector<Matrix<double>> original)
{
	Tensor = original;
}

Tensor1D::Tensor1D(Matrix<double> original)
{
	Tensor.push_back(original);
}

Tensor1D::Tensor1D()
{
}

void Tensor1D::Append(const Matrix<double> matrix)
{
	Tensor.push_back(matrix);
}

std::vector<Matrix<double>>& Tensor1D::GetTensor()
{
	return Tensor; 
}



Tensor1D Tensor1D::operator*(const Tensor1D& rhs)
{
	assert(Tensor.size() == rhs.Tensor.size());
	Tensor1D out;
	out.Tensor.resize(Tensor.size());
	for (int i = 0; i < Tensor.size(); i++)
	{
	 out.Tensor[i] = Tensor[i] * rhs.Tensor[i];
	}
	return out;
}

Tensor1D Tensor1D::Tensor1DHadamard(const Tensor1D& Lhs, const Tensor1D& Rhs)
{
	assert(Lhs.Tensor.size() == Rhs.Tensor.size());
	Tensor1D Out;
	Out.Tensor.resize(Lhs.Tensor.size());
	for (int i = 0; i < Lhs.Tensor.size(); i++)
	{
		Out.Tensor[i] = Hadamard(Lhs.Tensor[i], Rhs.Tensor[i]);
	}
	return Out;
}

std::vector<Matrix<double>>& Tensor1D::operator[](int i)
{
	return Tensor;
}

void Tensor1D::Transpose()
{
	for (int i = 0; i < Tensor.size(); i++)
	{
		Tensor[i] = Tensor[i].Transpose();
	}
}

unsigned int Tensor1D::GetSize()
{
	return Tensor.size();
}

void Tensor1D::operator=(const Tensor1D source)
{
	Tensor = source.Tensor;
}

Tensor1D Tensor1D::operator+(const Tensor1D source)
{
	assert(Tensor.size() == source.Tensor.size());
	Tensor1D out;
	out.Tensor.resize(Tensor.size());
	for (int i = 0; i < Tensor.size(); i++)
	{
		out.Tensor[i] = Tensor[i] + source.Tensor[i];
	}
	return out;
}
