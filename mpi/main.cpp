#include "matrix_ops.h"
using namespace std;

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
    
//----------------------------------------------------------------------//

    Matrix mtx(n * block_k, n * block_k, block_k, 1, id);
    mtx.block_form();
    Matrix diag(mpi_k * block_k, mpi_k * block_k, block_k);
    diag.block_form();
    
    double sub_start, cycle_start;
    double start = MPI_Wtime();
    for (int i = 0; i < n; ++i) {
        // gather diagonal block

        MPI_Allgather(mtx.get_block(i, i), block_k * block_k, MPI_DOUBLE, diag.ptr(), block_k * block_k, MPI_DOUBLE, MPI_COMM_WORLD);
        //rearrange & compute diag

        for (int j = 0; j < mpi_k; ++j) {
            for (int k = 0; k < mpi_k; ++k) {
                diag.get_block(j, k) = &(diag.ptr()[(j * mpi_k + k) * block_k * block_k]);
            }
        }
        diag = Matrix(diag.block_ptr(), mpi_k, mpi_k, block_k);
	diag.LLtDecomposition();
        diag.block_form();
        mtx.get_block(i, i) = diag.get_block(id_i, id_j); //rewire diag block in process matrix

        if (i < n - 1) {
            Matrix diag_L = get<0>(extractLLt(diag));
            Matrix diag_Lt = get<1>(extractLLt(diag));
            //gather process matrix i-th row/columns
            int vec_len = (n - i - 1);
            Matrix row_vec(block_k, vec_len * block_k, block_k);
            Matrix col_vec(block_k, vec_len * block_k, block_k);
            row_vec.block_form();
            col_vec.block_form();

            for (int j = 0; j < vec_len; ++j) {
                row_vec.block_ptr()[0][j] = mtx.block_ptr()[i][j + i + 1];
                col_vec.block_ptr()[0][j] = mtx.block_ptr()[j + i + 1][i];
            }
            row_vec = Matrix(row_vec.block_ptr(), 1, vec_len, block_k);
            col_vec = Matrix(col_vec.block_ptr(), 1, vec_len, block_k);

            Matrix row_mat(mpi_k * block_k, vec_len * block_k, block_k);
            Matrix col_mat(mpi_k * block_k, vec_len * block_k, block_k);
            
            //gather i-th row/col matrices for respective process groups
            MPI_Request row_req, col_req;
            MPI_Status row_st, col_st;
            MPI_Iallgather(row_vec.ptr(), block_k * block_k * vec_len, MPI_DOUBLE, row_mat.ptr(), block_k * block_k * vec_len, MPI_DOUBLE, ROW_COMM, &row_req);
            MPI_Iallgather(col_vec.ptr(), block_k * block_k * vec_len, MPI_DOUBLE, col_mat.ptr(), block_k * block_k * vec_len, MPI_DOUBLE, COL_COMM, &col_req);
            MPI_Wait(&row_req, &row_st);
            MPI_Wait(&col_req, &col_st);

            row_mat.block_form();
            col_mat.block_form();
            col_mat.block_transpose();

            //compute i-th row/col matrices

            row_mat.solve(diag_L, 'L');
            col_mat.solve(diag_Lt, 'R');
            row_mat.block_form();
            col_mat.block_form();

            //modify i-th row/col matrices in process matrix
            for (int j = 0; j < vec_len; ++j) {
                mtx.get_block(i, j + i + 1) = row_mat.get_block(id_i, j);
                mtx.get_block(j + i + 1, i) = col_mat.get_block(j, id_j);
            }

            //subMatrix
            Matrix subMatrix = col_mat * row_mat;

            subMatrix.block_form();

            for (int j = i + 1; j < n; ++j) {
                for (int k = i + 1; k < n; ++k) {
                    for (int l = 0; l < block_k * block_k; ++l) {
                        mtx.get_block(j, k)[l] -= subMatrix.get_block(j - i - 1, k - i - 1)[l];
                    }
                }
            }
        }
        mtx = Matrix(mtx.block_ptr(), n, n, block_k);
    }
    auto end = MPI_Wtime();

    int N = n * mpi_k;
    Matrix full_mtx(N * block_k, N * block_k, block_k);
    full_mtx.block_form();
    MPI_Gather(mtx.ptr(), block_k * n * block_k * n, MPI_DOUBLE, full_mtx.ptr(), block_k * n * block_k * n, MPI_DOUBLE, aux_id, MPI_COMM_WORLD);

    if (id == aux_id) {
        full_mtx.gather(mpi_k);
    }

    Matrix old_full_mtx(N * block_k, N * block_k, block_k);
    old_full_mtx.block_form();
    mtx = Matrix(n * block_k, n * block_k, block_k, 1, id);
    MPI_Gather(mtx.ptr(), block_k * n * block_k * n, MPI_DOUBLE, old_full_mtx.ptr(), block_k * n * block_k * n, MPI_DOUBLE, aux_id, MPI_COMM_WORLD);
    if (id == aux_id) {
        old_full_mtx.gather(mpi_k);
    }

    if (id == aux_id) {
        cout << "Time spent: " << time(start, end) << " ms" << endl;
        cout << "Precision: " << precision(old_full_mtx, full_mtx) << endl;
    }

    MPI_Finalize();
}
