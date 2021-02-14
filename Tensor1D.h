#pragma once
#include "Matrix.h"


class Tensor1D
{
	public:
		Tensor1D(const std::vector<Matrix<float>>& original);
		Tensor1D(const Matrix<float>& original);
		Tensor1D();
		void Append(const Matrix<float>& matrix);
		std::vector<Matrix<float>>& GetTensor();
		Tensor1D operator*(const Tensor1D& rhs) const;
		static Tensor1D  Tensor1DHadamard(const Tensor1D& Lhs, const Tensor1D& Rhs);
		std::vector < Matrix<float> >& operator[](int i);
		Tensor1D ReshapeFlat(int nrColumns, int nrRows);
		void Transpose();
		void Clear();
		static Tensor1D Convolution(const Tensor1D& Lhs, const Tensor1D& Rhs);
		unsigned int GetSize() const;
		void operator=(const Tensor1D& source);
		Tensor1D operator+(const Tensor1D& source);
	private:
		std::vector<Matrix<float>>  Tensor;
};