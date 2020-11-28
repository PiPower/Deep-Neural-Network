#pragma once
#include "Layer.h"

class Flatern : public BaseLayer
{
public:
	Flatern() = default;
	std::vector<Matrix<double>> Mul(std::vector<Matrix<double>>& A);
	std::vector<Matrix<double>> GetNablaWeight();
	std::vector<Matrix<double>> GetNablaBias();
private:
	std::vector<Matrix<double>> ShapeWeight;
	std::vector<Matrix<double>> ShapeBias;
};

