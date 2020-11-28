#include "Tensor2D.h"

Tensor2D::Tensor2D(const std::vector<std::vector<Matrix<double>>> original)
{
	for (int i = 0; i < original.size(); i++)
	{
		Tensor.push_back(original[i]);
	}
}

Tensor2D::Tensor2D()
{
}

void Tensor2D::Append(const Matrix<double> matrix)
{
	Tensor.back().Append(matrix);
}

void Tensor2D::Append(const std::vector<Matrix<double>> MatrixList)
{
	Tensor.push_back(MatrixList);
}

Tensor2D Tensor2D::operator*(const Tensor2D& rhs)
{
	assert(Tensor.size() == rhs.Tensor.size());
	Tensor2D out;
	out.Tensor.resize(Tensor.size());

	for (int i = 0; i < Tensor.size(); i++)
	{
		auto res = Tensor[i] * rhs.Tensor[i];
		out[i] = res;
	}
	return out;
}

Tensor2D Tensor2D::operator+(const Tensor2D& rhs)
{
	assert(Tensor.size() == rhs.Tensor.size());
	Tensor2D out;
	out.Tensor.resize(Tensor.size());
	for (int i = 0; i < Tensor.size(); i++)
	{
		out.Tensor[i] = Tensor[i] + rhs.Tensor[i];
	}

	return out;
}


Tensor1D& Tensor2D::operator[](int i)
{
	return Tensor[i];
}

void Tensor2D::Clear()
{
	for (auto& tensor_ : Tensor)
	{
		tensor_.Clear();
	}
}

void Tensor2D::TransposeAt(unsigned int index)
{
	Tensor[index].Transpose();
}

unsigned int Tensor2D::GetSize()
{
	return Tensor.size();
}
