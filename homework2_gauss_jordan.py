import numpy as np

def matrix_inverse_gj(matrix):
    # if we cannot perform gauss jordan - matrix is not squared
    if len(matrix) != len(matrix[0]):
        return "Error: Matrix should be squared to build an inverse one."

    # create a copy of a matrix with type float() of the elements
    n = len(matrix)
    augmented = [row.copy() for row in matrix]
    for i in range(n):
        augmented[i] = [float(x) for x in augmented[i]]

    # add an identity matrix to the right side
    for i in range(n):
        augmented[i] += [1.0 if j == i else 0.0 for j in range(n)]

    # gauss jordan itself
    for pivot in range(n):
        max_row = pivot
        for i in range(pivot + 1, n):
            if abs(augmented[i][pivot]) > abs(augmented[max_row][pivot]):
                max_row = i

        if max_row != pivot:
            augmented[pivot], augmented[max_row] = augmented[max_row], augmented[pivot]

        # if we cannot perform gauss jordan - matrix is singular
        if abs(augmented[pivot][pivot]) == 0:
            return "Error: Matrix is singular. Building an inverse is impossible."

        pivot_val = augmented[pivot][pivot]
        for j in range(len(augmented[pivot])):
            augmented[pivot][j] /= pivot_val

        for i in range(n):
            if i != pivot and augmented[i][pivot] != 0:
                factor = augmented[i][pivot]
                for j in range(len(augmented[i])):
                    augmented[i][j] -= factor * augmented[pivot][j]

    inverse = []
    for i in range(n):
        inverse.append(augmented[i][n:])
    return np.array(inverse)

if __name__ == "__main__":
    # an example of matrix that can be inverted
    A = [[4, 7],
         [2, 6]]
    A_inv = matrix_inverse_gj(A)
    print("Matrix A:")
    print("Initial matrix:")
    print(np.array(A))
    print()
    print("Inverse matrix:")
    print(A_inv)

    # an example of singular matrix (i.e. gives an error and should exit the func)
    B = [[1, 2],
         [2, 4]]
    print()
    print("Matrix B:")
    B_inv = matrix_inverse_gj(B)
    print(B_inv)

    # an example of non-squared matrix (i.e. gives an error and should exit the func)
    C = [[1, 2, 3],
         [4, 5, 6]]
    print()
    print("Matrix C:")
    C_inv = matrix_inverse_gj(C)
    print(C_inv)
