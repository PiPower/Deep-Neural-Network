#pragma once
#include <assert.h>
#include <vector>
#include <random>
#include <iostream>

template <typename Type>
class Matrix
{
	friend Matrix<double> Hadamard(const Matrix<double>& a1, const  Matrix<double>& a2);
public:
	enum class Init_Type
	{
		ZERO_INIT = 0,
		RANDOM_INIT = 1,
	};
public:
	Matrix( int Columns = 1, int Rows = 1,Init_Type init= Init_Type::ZERO_INIT , std::mt19937_64* gen = nullptr, std::normal_distribution<double>* unif=nullptr)
		:
		Columns(Columns), Rows(Rows)
	{
		MatPtr.resize(Columns * Rows);

		switch (init)
		{
		case Init_Type::ZERO_INIT:
			memset(MatPtr.data(), 0, sizeof(Type) * Rows * Columns);
			break;
		case Init_Type::RANDOM_INIT:
		{
			for (int j = 0; j < Rows;j++)
			{
				for (int i = 0; i < Columns; i++)
				{
					MatPtr[j*Columns+i] = (*unif)(*gen);
				}
			}
		} break;
		}
	}

	Matrix<Type> ApplyFunction(Type (*Func)(Type))
	{
		Matrix<Type> out(Columns, Rows);
		for (int i = 0; i < Rows * Columns; i++)
		{
			out.MatPtr[i] = Func(this->MatPtr[i]);
		}
		return out;
	}
	Matrix<Type> Transpose() const
	{
		Matrix<Type> Transposed(Rows,Columns);

		for (int y = 0; y < Rows; y++)
		{
			for (int x = 0; x < Columns; x++)
			{
				Transposed.MatPtr[x * Rows + y] = MatPtr[y * Columns + x];
			}
		}

		return Transposed;
	}

	Type GetAt(int i,int j) const
	{
		return MatPtr[i * Columns + j];
	}
	int GetRows() const
	{
		return Rows;
	}
	void Clear()
	{
		memset(MatPtr.data(), 0, sizeof(double) * Columns * Rows);
	}


	// kopiowanie Matrixy po prawej stronie
	void operator =(const Matrix& rhs)
	{
		Columns = rhs.Columns;
		Rows = rhs.Rows;

		MatPtr.resize(Columns * Rows);
		memcpy(MatPtr.data(), rhs.MatPtr.data(), Rows * Columns * sizeof(Type));
	}

	// To make this function work all matrices MUST have te same number of columns and rows
	static Matrix<double> CopyFromVector( std::vector<Matrix<Type>>& List)
	{
		int nrRows = List[0].GetRows();
		int nrColumns = List[0].GetColumns();

		Matrix<double> Out(1, nrRows * nrColumns * List.size());
		for (int i = 0; i < List.size(); i++)
		{
			int index = List[0].Rows * List[0].Columns * i;
			memcpy(&Out.MatPtr[index], List[i].MatPtr.data() , sizeof(Type) * nrRows * nrColumns);
		}
		//memcpy(MatPtr.data(), Pointer, sizeof(Type) * List[0].Rows * List[0].Columns* List.size());
		return Out;
	}

	double& operator[](int index)
	{
		return MatPtr[index];
	}

	void SetValue(unsigned int y, unsigned int x,const Type& Value)
	{
		MatPtr[y * Columns + x] = Value;
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

	void operator+=(const double& rhs)
	{

		for (int i = 0; i < Rows * Columns; i++)
		{
			MatPtr[i] += rhs;
		}

	}

	int GetColumnMaxIndex(int i) const
	{

		Type Max = std::numeric_limits<Type>::min();
		int index = 0;
		for (int j = 0; j < Rows; j++)
		{
			if (Max < MatPtr[j * Columns + i])
			{
				Max = MatPtr[j * Columns + i];
				index = j;
			}
		}
		return index;
	}
	// odejmowanie Matrixy i zwracanie wyniku w postaci Matrixy
	Matrix operator-(const Matrix& rhs) const
	{
		assert(Columns == rhs.Columns && Rows == rhs.Rows);
		
			Matrix out(Columns, Rows);
			for (int i = 0; i < Rows * Columns; i++)
			{
				out[i] = MatPtr[i] - rhs.MatPtr[i];
			}
			return out;
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
	int GetColumns() const
	{
		return Columns;
	}
	Matrix operator++(int)
	{
		for (int i = 0; i < Rows * Columns; i++)
		{
			MatPtr[i]++;
		}
		return *this;
	}
	Matrix operator*(const Matrix& rhs) const
	{
		assert(Columns == rhs.Rows);
	
			Matrix out(rhs.Columns, Rows);
			for (int line1 = 0; line1 < Rows; line1++)
			{
				for (int line2 = 0; line2 < rhs.Columns; line2++)
				{
					for (int i = 0; i < rhs.Rows; i++)
					{
						auto l1 = MatPtr[line1 * Columns + i];
						auto l2 = rhs.MatPtr[i * rhs.Columns + line2];

						out[line1 * rhs.Columns + line2] += l1 * l2;
					}
				}
			}
			return out;
		
	}

	std::vector<Matrix<double>> ReshapeFlatMatrix(int nrColumns, int nrRows)
	{
		int MatrixCount = (Columns * Rows) / (nrColumns * nrRows);
		assert((double)(Columns * Rows) / (double)(nrColumns * nrRows) == MatrixCount);

		std::vector<Matrix<double>> out;
		out.resize(MatrixCount);
		for (int i = 0; i < MatrixCount; i++)
		{
			out[i].MatPtr.resize(nrColumns * nrRows);
			out[i].Columns = nrColumns;
			out[i].Rows = nrRows;
			memcpy(out[i].MatPtr.data(), &MatPtr.data()[i * nrColumns * nrRows], nrColumns * nrRows * sizeof(Type));
		}

		return out;
	}
	Matrix<Type> operator*(const double& number) const
	{
		Matrix<double> out(Columns, Rows);

		for (int i = 0; i < Rows * Columns; i++)
		{
			out.MatPtr[i] = MatPtr[i]* number;
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
	void ShowValues()
	{
		std::cout << "--------------------------------------------------------- \n";
		for (int i = 0; i < Rows; i++)
		{
			std::cout << "|";
			for (int g = 0; g < Columns; g++)
			{
				int index = i * Columns + g;
				double wartosc = MatPtr[index];
				std::cout << "   " << MatPtr[index];
			}
			std::cout << "|\n";
		}
	}
	static Matrix<Type> Convolution(const Matrix<Type>& Image,const Matrix<Type>& kernel,int Step_Size=1)
	{
		//Checks if specified kernel and step size can be used
		assert( (double)(Image.Columns - kernel.Columns) / Step_Size + 1 == floor((double)(Image.Columns - kernel.Columns) / Step_Size + 1) );
		assert((double)(Image.Rows - kernel.Rows) / Step_Size + 1 == floor((double)(Image.Rows - kernel.Rows) / Step_Size + 1));


		Matrix<Type> Out_Mat((Image.Columns- kernel.Columns)/Step_Size+1, (Image.Rows - kernel.Rows) / Step_Size + 1);

		for (int y = 0; y < Out_Mat.GetRows(); y++)
		{
			for (int x = 0; x < Out_Mat.GetColumns(); x++)
			{
//-------------------------------------------------------------------
				double result = 0;
				for (int y2 = 0; y2 < kernel.Rows; y2++)
				{
					for (int x2 = 0; x2 < kernel.Columns; x2++)
					{
						result += Image.GetAt(y2 + y, x2 + x) * kernel.GetAt(y2, x2);
					}
				}
//-------------------------------------------------------------------
				Out_Mat.SetValue(y, x, result);
			}
		}

		return Out_Mat;
	}
private:
	std::vector<Type> MatPtr;
	int Columns;
	int Rows;
};

typedef Matrix<double>::Init_Type  MatrixInit;
