#pragma once
#include <assert.h>
#include <vector>
template <typename Type>

class Matrix
{
public:
	enum class Init_Type
	{
		ZERO_INIT = 0,
		RANDOM_INIT = 1,
	};
public:
	Matrix( int Columns = 1, int Rows = 1, Init_Type init= Init_Type::ZERO_INIT)
		:
		Columns(Columns), Rows(Rows)
	{
		MatPtr.resize(Columns * Rows);

		switch (init)
		{
		case Init_Type::ZERO_INIT:
			memset(MatPtr.data(), 0, sizeof(Type) * Rows * Columns);
		}
	}

	void ApplyFunction(Type  (* FcnPtr)(Type Z) )
	{
		for (int i = 0; i < Columns * Rows; i++) MatPtr[i] = FcnPtr(MatPtr[i]);
	}
	// kopiowanie Matrixy po prawej stronie
	void operator =(const Matrix rhs)
	{
		Columns = rhs.Columns;
		Rows = rhs.Rows;

		MatPtr.resize(Columns * Rows);
		for (int y = 0; y < Rows; y++)
		{
			for (int x = 0; x < Columns; x++)
			{
				MatPtr[y * Columns + x] = rhs.MatPtr[y * Columns + x];
			}
		}
	}

	double& operator[](int index)
	{
		return MatPtr[index];
	}
	// dodawanie Matrixy i zwracanie wyniku w postaci Matrixy
	Matrix operator+(const Matrix& rhs)
	{
		assert(Columns == rhs.Columns && Rows == rhs.Rows);
		
			Matrix out(Columns, Rows);
			for (int i = 0; i < Rows * Columns; i++)
			{
				out[i] = MatPtr[i] + rhs.MatPtr[i];
			}
			return out;
	}
	// dodawanie Matrixy I zapisywanie dzialania w obiekcie wywolujacym
	void operator+=(const Matrix& rhs)
	{
		assert(Columns == rhs.Columns && Rows == rhs.Rows);
		
			for (int i = 0; i < Rows * Columns; i++)
			{
				MatPtr[i] += rhs.MatPtr[i];
			}

	}
	// odejmowanie Matrixy i zwracanie wyniku w postaci Matrixy
	Matrix operator-(const Matrix& rhs)
	{
		assert(Columns == rhs.Columns && Rows == rhs.Rows);
		
			Matrix out(Columns, Rows);
			for (int i = 0; i < Rows * Columns; i++)
			{
				out[i] = MatPtr[i] - rhs.MatPtr[i];
			}
			return out;
	;
	}
	// odejmowanie Matrixy I zapisywanie dzialania w obiekcie wywolujacym
	void operator-=(const Matrix& rhs)
	{
		assert(Columns == rhs.Columns && Rows == rhs.Rows);
		
			for (int i = 0; i < Rows * Columns; i++)
			{
				MatPtr[i] -= rhs.MatPtr[i];
			}
		
	}
	// zmniejszenie wszystkich wartosci Matrixy o 1
	Matrix operator--(int)
	{
		for (int i = 0; i < Rows * Columns; i++)
		{
			MatPtr[i]--;
		}
		return *this;
	}
	Matrix operator++(int)
	{
		for (int i = 0; i < Rows * Columns; i++)
		{
			MatPtr[i]++;
		}
		return *this;
	}
	Matrix operator*(const Matrix& rhs)
	{
		assert(Columns == rhs.Rows);
	
			Matrix out(rhs.Columns, Rows);
			for (int line1 = 0; line1 < Rows; line1++)
			{
				for (int line2 = 0; line2 < rhs.Columns; line2++)
				{
					for (int i = 0; i < rhs.Rows; i++)
					{
						out[line1 * rhs.Columns + line2] += MatPtr[line1 * Columns + i] * rhs.MatPtr[i * rhs.Columns + line2];
					}
				}
			}
			return out;
		
	}
	bool operator==(const Matrix& rhs)
	{
		assert(Columns == rhs.Columns && Rows == rhs.Rows);
	
			for (int i = 0; i < Rows * Columns; i++)
			{
				if (MatPtr[i] != rhs.MatPtr[i]) return false;
			}
			return true;
		
	}
	/*void ShowValues()
	{
		cout << "--------------------------------------------------------- \n";
		for (int i = 0; i < Rows; i++)
		{
			cout << "|";
			for (int g = 0; g < Columns; g++)
			{
				int index = i * Columns + g;
				double wartosc = MatPtr[index];
				cout << "   " << MatPtr[index];
			}
			cout << "|\n";
		}
	}*/
private:
	std::vector<Type> MatPtr;
	int Columns;
	int Rows;
};

typedef Matrix<double>::Init_Type  MatrixInit;