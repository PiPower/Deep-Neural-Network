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



Tensor1D Tensor1D::operator*(const Tensor1D& rhs) const
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

// Flat means that the tensor is 1 dim with matrix(1,n) or matrix(n,1)
Tensor1D Tensor1D::ReshapeFlat(int nrColumns, int nrRows)
{
	assert(Tensor.size()==1);
	Tensor1D out ( Tensor[0].ReshapeFlatMatrix(nrColumns, nrRows));
	return out;
}

void Tensor1D::Transpose()
{
	for (int i = 0; i < Tensor.size(); i++)
	{
		Tensor[i] = Tensor[i].Transpose();
	}
}

void Tensor1D::Clear()
{
	for (int i = 0; i < Tensor.size(); i++)
	{
		Tensor[i].Clear();
	}
}

Tensor1D Tensor1D::Convolution(const Tensor1D& Img, const Tensor1D& Kernel)
{
	Tensor1D out;
	assert(Img.Tensor.size() >0 && Kernel.Tensor.size() > 0);
	out.Tensor.resize(Kernel.Tensor.size());
	for (int i = 0; i < Kernel.Tensor.size(); i++)
	{
		out.Tensor[i] = Matrix<double>::Convolution(Img.Tensor[0], Kernel.Tensor[i]);
	}

	return out;
}

unsigned int Tensor1D::GetSize() const
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
