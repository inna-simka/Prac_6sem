#include "matrix_ops.h"

int main()
{
    int n;
    std::cout << "Enter square matrix (n x n) dimension:" << std::endl << "n = ";
    std::cin >> n;

    int b;
    while (1) {
        std::cout << "Enter block size b (n = " << n << " must be multiple of b):" << std::endl << "b = ";
        std::cin >> b;
        if (n % b != 0) {
            std::cout << "n = " << n << " is not multiple of b = " << b << std::endl;
        } else {
            break;
        }
    }

    char manual_mode;
    Matrix A;
    while (1) {
        std::cout << "Manual input of matrix? (Y/n)" << std::endl;
        std::cin >> manual_mode;
        manual_mode = std::toupper(manual_mode);
        if (manual_mode == 'Y') {
            A = manualInput(n);
            break;
        } else if (manual_mode == 'N') {
            A = randomInput(n);
            break;
        } else {
            std::cout << "Answer not recognized, try again." << std::endl;
        }
    }
    Matrix LU = LUBlock(A, n, b);
    std::cout << "Precision ||LU - A|| = " << precision (A, LU, n) << std::endl;
    if (manual_mode == 'Y') {
        printMatrix(LU, n);
    }
    return 0;
}