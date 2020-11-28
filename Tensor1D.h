#pragma once
#include "Matrix.h"


class Tensor1D
{
	public:
		Tensor1D(const std::vector<Matrix<double>>& original);
		Tensor1D(const Matrix<double>& original);
		Tensor1D();
		void Append(const Matrix<double>& matrix);
		std::vector<Matrix<double>>& GetTensor();
		Tensor1D operator*(const Tensor1D& rhs) const;
		static Tensor1D  Tensor1DHadamard(const Tensor1D& Lhs, const Tensor1D& Rhs);
		std::vector < Matrix<double> >& operator[](int i);
		Tensor1D ReshapeFlat(int nrColumns, int nrRows);
		void Transpose();
		void Clear();
		static Tensor1D Convolution(const Tensor1D& Lhs, const Tensor1D& Rhs);
		unsigned int GetSize() const;
		void operator=(const Tensor1D& source);
		Tensor1D operator+(const Tensor1D& source);
	private:
		std::vector<Matrix<double>>  Tensor;
};