import numpy as np

def gauss(matrix):
    # brinf the matrix to the triangular form
    n = len(matrix)
    matrix = np.flipud(matrix)
    for j in range(len(matrix) - 1):
        for i in range(len(matrix) - 1 - j):
            k = matrix[i][j] / matrix[len(matrix) - j - 1][j]
            row = matrix[len(matrix) - j - 1] * k
            matrix[i] = matrix[i] - row
    matrix = np.flipud(matrix)
    
    # divide augmented triangular matrix into two matrices
    # A - coefficients of the system
    # B - the values of the equations
    A, B = [], []
    for i in range(n):
        temp = []
        for j in range(n):
            temp.append(matrix[i][j])
        A.append(temp)
        B.append(matrix[i][-1])
    # finding the values of the variables
    x = np.zeros(len(B))
    for i in range(len(B)-1, -1, -1):
        x[i] = (B[i] - np.dot(A[i][i+1:], x[i+1:]))/A[i][i]
    
    return x


# we create any augmented matrix, this is just an example
augmented_matrix = np.array([[1, 1, 1, 6],
                            [1, 2, 2, 11],
                            [2, 3, -4, 3]], dtype=float)

solution = gauss(augmented_matrix)
print("Solution (Values of the variables):")
print(solution)
