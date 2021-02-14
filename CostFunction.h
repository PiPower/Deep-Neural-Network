#pragma once
#include "Matrix.h"
class CostFunction
{
protected:
	CostFunction() = default;
public:
	virtual Matrix<float> Function(const Matrix<float>& A, const Matrix<float>& Y);
	virtual Matrix<float> Function_Der(const Matrix<float>& A, const Matrix<float>& Y);
	virtual ~CostFunction() ;
};


class QuadraticCost : public CostFunction
{
public:
	QuadraticCost() = default;
	Matrix<float> Function(const Matrix<float>& A, const Matrix<float>& Y);
    Matrix<float> Function_Der(const Matrix<float>& A, const Matrix<float>& Y);
};



