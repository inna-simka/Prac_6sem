CXX = OMPI_CXX=icpc mpicxx
FLAGS = -Wall -O2 -qopenmp -mkl=parallel
#source /opt/intel/mkl/bin/mklvars.sh intel64

run: runner
	salloc -p new -N 4 -c 3 mpirun ./runner 2048 128

runner: main.cpp matrix_ops.cpp
	${CXX} ${FLAGS} main.cpp matrix_ops.cpp -o runner
