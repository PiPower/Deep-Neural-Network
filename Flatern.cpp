#include "Flatern.h"

std::vector<Matrix<float>> Flatern::Mul(std::vector<Matrix<float>>& A)
{
	int nrRows = A[0].GetRows();
	int nrColumns = A[0].GetColumns();

	for (int i = 0; i < A.size(); i++)
	{
		ShapeWeight.push_back(A[i]);
		ShapeWeight.back().Clear();
	}

	ShapeBias.push_back(Matrix<float>(1, A.size()));
	std::vector<Matrix<float>>OutVec{Matrix<float>::CopyFromVector(A)};
	return OutVec;
}

std::vector<Matrix<float>> Flatern::GetNablaWeight()
{
	return ShapeWeight;
}

std::vector<Matrix<float>> Flatern::GetNablaBias()
{
	return ShapeBias;
}
