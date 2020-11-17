#include "CostFunction.h"
#include <limits>

Matrix<double> CostFunction::Function(const Matrix<double>& A, const Matrix<double>& Y)
{
	return Matrix<double>();
}

Matrix<double> CostFunction::Function_Der(const Matrix<double>& A, const Matrix<double>& Y)
{
	return Matrix<double>();
}

CostFunction::~CostFunction()
{
}


Matrix<double> QuadraticCost::Function(const Matrix<double>& A, const Matrix<double>& Y)
{
	return (A - Y) * (A - Y);
}

Matrix<double> QuadraticCost::Function_Der(const Matrix<double>& A, const Matrix<double>& Y)
{
	return (A - Y);
}