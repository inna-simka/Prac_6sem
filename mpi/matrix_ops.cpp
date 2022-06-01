#include "matrix_ops.h"

Matrix::Matrix(int rows_val, int cols_val, int block_size_val, bool generate, int seed) : 
    rows(rows_val), cols(cols_val), block_size(block_size_val) 
{
    mtx = new double[rows * cols];
    if (generate) {
        double a = 0.0, b = 20.0;
        srand(seed * 10 + seed);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                mtx[i*cols + j] = a + (rand() % static_cast<int>(b - a + 1));
            }
        }
    } else {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                mtx[i*cols + j] = 0.0;
            }
        }
    }
}

Matrix::Matrix(double *mtx_val, int rows_val, int cols_val, int block_size_val) :
    rows(rows_val), cols(cols_val), block_size(block_size_val), mtx(mtx_val) {}

Matrix::Matrix(double ***block_mtx_val, int block_rows_val, int block_cols_val, int block_size_val) :
    rows(block_rows_val * block_size_val), cols(block_cols_val * block_size_val), block_size(block_size_val), block_mtx(block_mtx_val)
{
    mtx = new double[rows * cols];
    for (int i = 0; i < block_rows_val; ++i) {
        for (int j = 0; j < block_cols_val; ++j) {
            for (int k = 0; k < block_size; ++k) {
                for (int l = 0; l < block_size; ++l) {
                    mtx[(i*cols + j)*block_size + l + k*cols] = block_mtx[i][j][k*block_size + l];
                }
            }
        }
    }
}

void Matrix::print() 
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (j != 0) {
                std::cout << " ";
            }
            std::cout << std::fixed << std::setiosflags(std::ios::left) << std::setw(6)
                      << std::setfill(' ') << std::setprecision(2) << mtx[i*cols+j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Matrix::block_form() 
{
    int block_mtx_rows = rows / block_size, block_mtx_cols = cols / block_size;

    block_mtx = new double**[block_mtx_rows];
    for (int i = 0; i < block_mtx_rows; ++i) {
        block_mtx[i] = new double*[block_mtx_cols];
        for (int j = 0; j < block_mtx_cols; ++j) {
            block_mtx[i][j] = new double[block_size * block_size];
        }
    }

    for (int i = 0; i < block_mtx_rows; ++i) {
        for (int j = 0; j < block_mtx_cols; ++j) {
            for (int k = 0; k < block_size; ++k) {
                for (int l = 0; l < block_size; ++l) {
                    block_mtx[i][j][k*block_size + l] = mtx[(i*cols + j)*block_size + k*cols + l];
                }
            }
        }
    }
}

void Matrix::factor()
{
    if (rows != cols) {
        std::cout << "Cannot factorize non-square matrix." << std::endl;
        exit(10);
    }
    double *LU = new double[rows * cols];
    for (int i = 0; i < rows; ++i) {
        for (int j = i; j < cols; ++j) {
            double sum = 0;
            for (int k = 0; k < i; ++k) {
                sum += LU[i*cols + k] * LU[k*cols + j];
            }
            LU[i*cols+j] = (mtx[i*cols + j] - sum);
        }

        for (int j = i + 1; j < cols; ++j) {
            double sum = 0;
            for (int k = 0; k < i; ++k) {
                sum += LU[j*cols + k] * LU[k*cols + i];
            }
            LU[j*cols + i] = (mtx[j*cols + i] - sum) / LU[i*cols + i];
        }
    }
    mtx = LU;
}

double* &Matrix::get_block(int i, int j)
{
    if (block_mtx) {
        return block_mtx[i][j];
    } else {
        std::cout << "Failed to get matrix block " << i << " " << j << std::endl;
        exit(0);
    }
}

double *Matrix::ptr() {
    return mtx;
}

double *** &Matrix::block_ptr() {
    return block_mtx;
}

void Matrix::transpose(bool fortran)
{
    double *mtx_T = new double[cols * rows];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mtx_T[j*rows + i] = mtx[i*cols + j];
        }
    }
        std::swap(rows, cols);
    mtx = mtx_T;
}

double time(double a, double b){
    return (b - a) * MS;
}

void Matrix::block_transpose()
{
    Matrix dummy(cols, rows, block_size);
    dummy.block_form();

    for (int i = 0; i < rows / block_size; ++i) {
        for (int j = 0; j < cols / block_size; ++j) {
//           std::cout << "block " << i << " " << j << std::endl;
//            Matrix(block_mtx[i][j], block_size, block_size).print();
            dummy.get_block(j, i) = block_mtx[i][j];
        }
    }
    block_mtx = dummy.block_ptr();
    mtx = Matrix(block_mtx, cols / block_size, rows / block_size, block_size).ptr();
    std::swap(rows, cols);
}

std::tuple<Matrix, Matrix> extractLU(Matrix A)
{
    int rows = A.rows, cols = A.cols, block_size = A.block_size;
    Matrix L(rows, cols, block_size);
    Matrix U(rows, cols, block_size);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (i > j) {
                L.ptr()[i*cols + j] = A.ptr()[i*cols + j];
            } else if (i < j) {
                U.ptr()[i*cols + j] = A.ptr()[i*cols + j];
            } else if (i == j) {
                L.ptr()[i*cols + j] = 1;
                U.ptr()[i*cols + j] = A.ptr()[i*cols + j];
            }
        }
    }
    return std::make_tuple(L, U);
}

void Matrix::solve(Matrix A, char type)
{
    char cL = 'L';
    char cN = 'N';
    char cR = 'R';
    char cU = 'U';
    char cT = 'T';
    double alpha = 1.0;

    int m = rows, n = cols;
    transpose();
    A.transpose();
    if (type == 'L') {
        dtrsm(&cL, &cL, &cN, &cN, &m, &n, &alpha, A.ptr(), &m, mtx, &m);
    } else if (type == 'R') {
        dtrsm(&cR, &cU, &cN, &cN, &m, &n, &alpha, A.ptr(), &n, mtx, &m);
    }
    transpose(); // Fortran --> C++*/

/*    
    transpose();
    if (type == 'L') {
        dtrsm(&cR, &cL, &cT, &cN, &m, &n, &alpha, A.ptr(), &n, mtx, &m);
    } else if (type == 'R') {
        dtrsm(&cL, &cU, &cT, &cN, &m, &n, &alpha, A.ptr(), &m, mtx, &m);
    }
    transpose();*/
}

Matrix operator*(Matrix M1, Matrix M2)
{
    if (M1.cols != M2.rows) {
        std::cout << "Cannot multiply matrices of these dimensions." << std::endl;
        exit(1);
    }
    
    int m = M1.rows, k = M1.cols, n = M2.cols, block_size = M1.block_size;
    char cN = 'N';
    char cT = 'T';
    double alpha = 1.0;
    double beta = 0.0;

    Matrix result(n, m, block_size);
    dgemm(&cT, &cT, &m, &n, &k, &alpha, M1.ptr(), &k, M2.ptr(), &n, &beta, result.ptr(), &m);
    result.transpose();
    return result;
}

Matrix operator-(Matrix M1, Matrix M2)
{
    if (M1.cols != M2.cols or M1.rows != M2.rows) {
        std::cout << "Cannot subtract matrices of these dimensions." << std::endl;
        exit(1);
    }
    Matrix M3(M1.rows, M1.cols, M1.block_size);
    for (int i = 0; i < M1.rows; ++i) {
        for (int j = 0; j < M1.cols; ++j) {
            M3.ptr()[i*M1.cols + j] = M1.ptr()[i*M1.cols + j] - M2.ptr()[i*M1.cols + j];
        }
    }
    return M3;
}

void Matrix::gather(int mpi_k)
{
    int N = rows / block_size, n = N / mpi_k;
    for (int id = 0; id < mpi_k * mpi_k; ++id) {
        double *proc = new double[n * block_size * n * block_size];
        proc = mtx + id * (n * block_size * n * block_size);
        Matrix proc_mtx = Matrix(proc, n * block_size, n * block_size, block_size);
        proc_mtx.block_form();
            
        int id_i = id / mpi_k, id_j = id % mpi_k; 
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                block_mtx[id_i + i*mpi_k][id_j + j*mpi_k] = proc_mtx.get_block(i, j);
            }
        }
    }
    mtx = Matrix(block_mtx, N, N, block_size).ptr();
}

double precision(Matrix A, Matrix LU)
{
    Matrix L = std::get<0>(extractLU(LU));
    Matrix U = std::get<1>(extractLU(LU));
    Matrix A_PREC = L * U - A;
 //   A_PREC.print();
    double norm = 0;
    for (int i = 0; i < A_PREC.rows; ++i) {
        for (int j = 0; j < A_PREC.cols; ++j) {
            norm += A_PREC.ptr()[i*A_PREC.cols+j] * A_PREC.ptr()[i*A_PREC.cols+j];
        }
    }
    return std::sqrt(norm);
}

void msg(std::string str)
{
    std::cout << str << std::endl;
}