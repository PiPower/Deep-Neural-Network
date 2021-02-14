#include "CostFunction.h"
#include <limits>

Matrix<float> CostFunction::Function(const Matrix<float>& A, const Matrix<float>& Y)
{
	return Matrix<float>();
}

Matrix<float> CostFunction::Function_Der(const Matrix<float>& A, const Matrix<float>& Y)
{
	return Matrix<float>();
}

CostFunction::~CostFunction()
{
}


Matrix<float> QuadraticCost::Function(const Matrix<float>& A, const Matrix<float>& Y)
{
	return (A - Y) * (A - Y);
}

Matrix<float> QuadraticCost::Function_Der(const Matrix<float>& A, const Matrix<float>& Y)
{
	return (A - Y);
}