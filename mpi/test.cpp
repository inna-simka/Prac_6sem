#include "matrix_ops.h"
#include <cstdio>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    int matrix_size, block_k;
    sscanf(argv[1], "%d", &matrix_size);
    sscanf(argv[2], "%d", &block_k);
    
    int id, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int mpi_k = (int)sqrt(mpi_size); 
    int id_i = id / mpi_k, id_j = id % mpi_k; 
    int aux_id = 0;

    MPI_Comm COL_COMM, ROW_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, id_j, id_i, &ROW_COMM);
    MPI_Comm_split(MPI_COMM_WORLD, id_i, id_j, &COL_COMM);

    int n = matrix_size / mpi_k / block_k;

    Matrix M1(2, 4, 1, true, id);
    Matrix M2(4, 2, 1, true, id);
    Matrix M3(2, 2, 1, true, id + 2);
    auto L = std::get<0>(extractLU(M3));
    auto U = std::get<1>(extractLU(M3));  
//----------------------------------------------------------------------//
    double start, dgemmt;
    if (id == aux_id) {
        M1.print();
        L.print();
        msg("L solve");
        M1.solve(L, 'L');
        M1.print();

  /*      M2.print();
        U.print();
        msg("U solve");
        M2.solve(U, 'R');
        M2.print();*/
    }

    MPI_Finalize();
}