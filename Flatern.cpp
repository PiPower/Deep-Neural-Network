#include "Flatern.h"

std::vector<Matrix<double>> Flatern::Mul(std::vector<Matrix<double>>& A)
{
	int nrRows = A[0].GetRows();
	int nrColumns = A[0].GetColumns();

	for (int i = 0; i < A.size(); i++)
	{
		ShapeWeight.push_back(A[i]);
		ShapeWeight.back().Clear();
	}

	ShapeBias.push_back(Matrix<double>(1, A.size()));
	std::vector<Matrix<double>>OutVec{Matrix<double>::CopyFromVector(A)};
	return OutVec;
}

std::vector<Matrix<double>> Flatern::GetNablaWeight()
{
	return ShapeWeight;
}

std::vector<Matrix<double>> Flatern::GetNablaBias()
{
	return ShapeBias;
}
