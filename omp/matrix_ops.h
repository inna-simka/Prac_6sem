#include <iostream>
#include <vector>
#include <tuple>
#include <iomanip>
#include <cstring>
#include <cctype>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <mkl.h>

typedef double* Matrix;
typedef double*** blockMatrix;

Matrix initMatrix(int m, int n);
Matrix generateMatrix(int n, int a, int b, int seed=42);
void printMatrix(Matrix A, int n);

Matrix transpose(Matrix A, int n);
Matrix callDGEMM(Matrix A, Matrix B, Matrix C, int m, int n, int k, double alpha, double beta);
Matrix callDTRSM(Matrix A, Matrix B, int n, char type, double alpha);
blockMatrix getBlockMatrix(Matrix A, int n, int b);
Matrix restoreMatrix(blockMatrix blockA, int m, int b);

std::tuple<Matrix, Matrix> extractLU(Matrix A, int n);
Matrix LUDecomposition(Matrix A, int n);
Matrix LUBlock(Matrix A, int n, int b);
double precision(Matrix A, Matrix LU, int n);

Matrix manualInput(int n);
Matrix randomInput(int n);