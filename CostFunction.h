#pragma once
#include "Matrix.h"
class CostFunction
{
protected:
	CostFunction() = default;
public:
	virtual Matrix<double> Function(const Matrix<double>& A, const Matrix<double>& Y);
	virtual Matrix<double> Function_Der(const Matrix<double>& A, const Matrix<double>& Y);
	virtual ~CostFunction() ;
};


class QuadraticCost : public CostFunction
{
public:
	QuadraticCost() = default;
	Matrix<double> Function(const Matrix<double>& A, const Matrix<double>& Y);
    Matrix<double> Function_Der(const Matrix<double>& A, const Matrix<double>& Y);
};
