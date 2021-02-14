#include "Matrix.h"

Matrix<float> Hadamard(const Matrix<float>& a1, const Matrix<float>& a2)
{
	assert(a1.Columns == a2.Columns && a1.Rows == a2.Rows);
	Matrix<float> Out(a1.Columns, a1.Rows);

	for (int y = 0; y < a1.Rows; y++)
	{
		for (int x = 0; x < a1.Columns; x++)
		{
			Out[y * a1.Columns + x] = a1.MatPtr[y * a1.Columns + x] * a2.MatPtr[y * a1.Columns + x];
		}
	}
	return Out;
}
