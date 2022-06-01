#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <cmath>
#include <tuple>
#include <mkl.h>
#include <map>
#include <vector>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <utility>
#include <string>

class Matrix
{
    private:
        double *mtx;
        double ***block_mtx;
    public:
        int block_size;
        int rows;
        int cols;
        
        Matrix(int rows_val, int cols_val, int block_size = 0, bool generate = false, int seed = 0);
        explicit Matrix(double *mtx_val, int rows_val, int cols_val, int block_size_val = 0);
        explicit Matrix(double ***block_mtx_val, int block_rows_val, int block_cols_val, int block_size_val);
        void print();
        void block_form();
        void factor();
        void transpose(bool fortran = false);
        void inverse(char type);
        double *ptr();
        double *** &block_ptr();
        double* &get_block(int i, int j);
        void gather(int mpi_k);
        void block_transpose();
        void solve(Matrix A, char type);
};
const double MS = 1e-2;
Matrix operator*(Matrix M1, Matrix M2);
Matrix operator-(Matrix M1, Matrix M2);
std::tuple<Matrix, Matrix> extractLU(Matrix A);
double precision(Matrix A, Matrix LU);
void msg(std::string str);
double time(double a, double b);