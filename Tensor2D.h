#pragma once
#include "Tensor1D.h"

class Tensor2D
{
public:
	Tensor2D(const std::vector<std::vector<Matrix<float>>>& original);
	Tensor2D();
	void Append(const Matrix<float>& matrix);
	void Append(const std::vector<Matrix<float>>& MatrixList);
	std::vector<Tensor1D > GetTensor() { return Tensor; }
	Tensor2D operator*(const Tensor2D& rhs);
	Tensor2D operator+(const Tensor2D& rhs);
	Tensor1D& operator[](int i);
	void Clear();
	void TransposeAt(unsigned int index);
	unsigned int GetSize();

private:
	std::vector<Tensor1D> Tensor;
};
