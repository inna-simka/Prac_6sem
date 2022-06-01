#include "matrix_ops.h"

Matrix initMatrix(int m, int n)
{
    double *A = new double[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i*m+j] = 0.0;
        }
    }
    return A;
}

Matrix generateMatrix(int n, int a, int b, int seed)
{
    srand(seed);
    double *mat = new double[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i*n+j] = a + (rand() % static_cast<int>(b - a + 1));
        }
    }
    return mat;
}

void printMatrix(Matrix A, int n)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j != 0) {
                std::cout << " ";
            }
            std::cout << std::fixed << std::setiosflags(std::ios::left) << std::setw(6)
                      << std::setfill(' ') << std::setprecision(2) << A[i*n+j];
        }
        std::cout << std::endl;
    }
}

Matrix transpose(Matrix A, int n)
{
    Matrix A_T = initMatrix(n, n);
    std::memcpy(A_T, A, n*n*sizeof(A[0]));
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            std::swap(A_T[n*i + j], A_T[n*j + i]);
        }
    }
    return A_T;
}

Matrix callDGEMM(Matrix A, Matrix B, Matrix C, int m, int n, int k, double alpha, double beta)
{
    char cN = 'N';
    Matrix A_T = transpose(A, n);
    Matrix B_T = transpose(B, n);
    Matrix C_T = transpose(C, n);
    dgemm(&cN, &cN, &m, &n, &k, &alpha, A_T, &m, B_T, &k, &beta, C_T, &m);
    return transpose(C_T, n);
}

Matrix callDTRSM(Matrix A, Matrix B, int n, char type='L', double alpha = 1.0)
{
    char cL = 'L';
    char cN = 'N';
    char cU = 'U';
    char cR = 'R';
    Matrix A_T = transpose(A, n);
    Matrix B_T = transpose(B, n);
    if (type == 'L') {
        dtrsm(&cL, &cL, &cN, &cN, &n, &n, &alpha, A_T, &n, B_T, &n);
    } else if (type == 'R') {
        dtrsm(&cR, &cU, &cN, &cN, &n, &n, &alpha, A_T, &n, B_T, &n);
    }
    return transpose(B_T, n);
}

blockMatrix getBlockMatrix(Matrix A, int n, int b)
{
    int m = n / b;
    blockMatrix blockA = new Matrix*[m];
    for (int i = 0; i < m; ++i) {
        blockA[i] = new Matrix[m];
        for (int j = 0; j < m; ++j) {
            blockA[i][j] = new double[b*b];
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < b; ++k) {
                for (int l = 0; l < b; ++l) {
                    blockA[i][j][k*b+l] = A[(i*n+j)*b + l + k*n];
                }
            }
        }
    }
    return blockA;
}

Matrix restoreMatrix(blockMatrix blockA, int m, int b)
{
    int n = m * b;
    Matrix A = initMatrix(n, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < b; ++k) {
                for (int l = 0; l < b; ++l) {
                    A[(i*n+j)*b + l + k*n] = blockA[i][j][k*b+l];
                }
            }
        }
    }
    return A;
}

std::tuple<Matrix, Matrix> extractLU(Matrix A, int n)
{
    Matrix L = initMatrix(n, n);
    Matrix U = initMatrix(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i > j) {
                L[i*n+j] = A[i*n+j];
            } else if (i < j) {
                U[i*n+j] = A[i*n+j];
            } else if (i == j) {
                L[i*n+j] = 1;
                U[i*n+j] = A[i*n+j];
            }
        }
    }
    return std::make_tuple(L, U);
}

Matrix LUDecomposition(Matrix A, int n)
{
    Matrix LU = initMatrix(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            double sum = 0;
            for (int k = 0; k < i; ++k) {
                sum += LU[i*n+k] * LU[k * n + j];
            }
            LU[i*n+j] = (A[i*n+j] - sum);
        }

        for (int j = i + 1; j < n; ++j) {
            double sum = 0;
            for (int k = 0; k < i; ++k) {
                sum += LU[j * n + k] * LU[k * n + i];
            }
            LU[j * n + i] = (A[j * n + i] - sum) / LU[i*n+i];
        }
    }
    return LU;
}

Matrix LUBlock(Matrix A, int n, int b)
{
    int m = n / b;
    blockMatrix blockA = getBlockMatrix(A, n, b);

    auto start = std::chrono::steady_clock::now();
    for (int k = 0; k < m; ++k) {
        blockA[k][k] = LUDecomposition(blockA[k][k], b); //factorize A[k][k];
        Matrix L = std::get<0>(extractLU(blockA[k][k], b));
        Matrix U = std::get<1>(extractLU(blockA[k][k], b));
        #pragma omp parallel
        {
            if (k < m - 1) {
                #pragma omp for
                for (int i = k + 1; i < m; ++i) {
                    blockA[i][k] = callDTRSM(U, blockA[i][k], b, 'R');
                }
                #pragma omp for
                for (int j = k + 1; j < m; ++j) {
                    blockA[k][j] = callDTRSM(L, blockA[k][j], b, 'L');
                }
                #pragma omp for collapse(2)
                for (int i = k + 1; i < m; ++i) {
                    for (int j = k + 1; j < m; ++j) {
                        blockA[i][j] = callDGEMM(blockA[i][k], blockA[k][j], blockA[i][j], b, b, b, -1.0, 1.0);
                    }
                }
            } 
        } 
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "Time spent: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                  << " ms" << std::endl;
    return restoreMatrix(blockA, m, b);
}

double precision(Matrix A, Matrix LU, int n)
{
    Matrix L = std::get<0>(extractLU(LU, n));
    Matrix U = std::get<1>(extractLU(LU, n));
    Matrix A_PREC = initMatrix(n, n);
    std::memcpy(A_PREC, A, n*n*sizeof(A[0]));

    A_PREC = callDGEMM(L, U, A_PREC, n, n, n, -1.0, 1.0);

    double norm = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            norm += A_PREC[i*n+j] * A_PREC[i*n+j];
        }
    }
    return std::sqrt(norm);
}

Matrix manualInput(int n)
{
    Matrix A = initMatrix(n, n);
    std::cout << "Enter square " << n << "-dimensional matrix:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> A[i*n+j];
        }
    }
    return A;
}

Matrix randomInput(int n)
{
    int a, b;
    std::cout << "Define range of values (a, b) for elemnts of randomly generated square " << n <<"-dimensional matrix" << std::endl;
    std::cout << "a = ";
    std::cin >> a;
    std::cout << "b = ";
    std::cin >> b;
    return generateMatrix(n, a, b);
}