CXX = icpc
CXXFLAGS = -Wall -O2 -qopenmp -mkl=sequential
OBJ = main.o matrix_ops.o

#for mkl run this in shell:
#source /opt/intel/mkl/bin/mklvars.sh intel64

all: ${OBJ}
	${CXX} ${CXXFLAGS} -o main ${OBJ}

main.o: main.cpp matrix_ops.h
	${CXX} ${CXXFLAGS} -c main.cpp

matrix_ops.o: matrix_ops.cpp matrix_ops.h
	${CXX} ${CXXFLAGS} -c matrix_ops.cpp