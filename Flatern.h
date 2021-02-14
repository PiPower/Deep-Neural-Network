#pragma once
#include "Layer.h"

class Flatern : public BaseLayer
{
public:
	Flatern() = default;
	std::vector<Matrix<float>> Mul(std::vector<Matrix<float>>& A);
	std::vector<Matrix<float>> GetNablaWeight();
	std::vector<Matrix<float>> GetNablaBias();
private:
	std::vector<Matrix<float>> ShapeWeight;
	std::vector<Matrix<float>> ShapeBias;
};

