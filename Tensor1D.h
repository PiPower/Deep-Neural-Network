#pragma once
#include "Matrix.h"


class Tensor1D
{
	public:
		Tensor1D(std::vector<Matrix<double>> original);
		Tensor1D(Matrix<double> original);
		Tensor1D();
		void Append(const Matrix<double> matrix);
		std::vector<Matrix<double>>& GetTensor();
		Tensor1D operator*(const Tensor1D& rhs);
		static Tensor1D  Tensor1DHadamard(const Tensor1D& Lhs, const Tensor1D& Rhs);
		std::vector < Matrix<double> >& operator[](int i);
		void Transpose();
		unsigned int GetSize();
		void operator=(const Tensor1D source);
		Tensor1D operator+(const Tensor1D source);
	private:
		std::vector<Matrix<double>>  Tensor;
};