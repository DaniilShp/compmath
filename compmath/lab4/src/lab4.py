import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=10)

def gauss(A, b, pivoting=False):
    n = len(A)
    extended_matrix = np.column_stack((A, b))
    for i in range(n):
        if pivoting:
            max_element_row = i
            for j in range(i+1, n):
               if abs(extended_matrix[j][i]) > abs(extended_matrix[max_element_row][i]):
                    max_element_row = j
            extended_matrix[[i, max_element_row]] = extended_matrix[[max_element_row, i]]
        for j in range(i + 1, n):
            factor = extended_matrix[j, i] / extended_matrix[i, i]
            extended_matrix[j, i:] -= factor * extended_matrix[i, i:]
    x = np.zeros(n+1, dtype=dtype)
    for i in range(n - 1, -1, -1):
        x[i] = (extended_matrix[i][-1] - np.dot(extended_matrix[i, i + 1:], x[i + 1:])) / extended_matrix[i, i]
    return x[:-1]

def thomas(A, b):
    local_A = A.copy()
    local_b = b.copy()
    n = len(A)
    gamma = np.zeros(n)
    betta = np.zeros(n)
    x = np.zeros(n)
    for i in range(n-1):
        gamma[i+1] = (-local_A[i, i+1]) / (local_A[i, i-1]*gamma[i] + local_A[i, i])
        betta[i+1] = (local_b[i] - local_A[i, i-1]*betta[i]) / (local_A[i, i-1]*gamma[i] + local_A[i, i])
    x[-1] = (local_b[-1] - local_A[-1, -2] * betta[-1]) / (local_A[-1, -1] + local_A[-1, -2] * gamma[-1])
    for i in range(n-1, 0, -1):
        x[i-1] = gamma[i]*x[i] + betta[i]
    return x

def cholesky(A, b):
    n = len(A)
    L = np.zeros((n, n), dtype=dtype)
    L[0, 0] = np.sqrt(A[0, 0])
    for i in range(n):
        for j in range(0, i):
            L[i, j] = 1 / L[j, j] * (A[i, j] - sum(L[i, k] * L[j, k] for k in range(j)))
        L[i, i] = np.sqrt(A[i, i] - sum(L[i, j] * L[i, j] for j in range(i)))
    L_T = np.transpose(L)
    y = np.zeros(n)
    y[0] = b[0] / L[0, 0]
    for i in range(n-1):
        y[i+1] = (b[i+1] - sum(L[i+1, j] * y[j] for j in range(i+1))) / L[i+1, i+1]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(L_T[i, i + 1:], x[i + 1:])) / L_T[i, i]
    return x


def generate_square_matrix(n):
    while True:
        A = np.random.uniform(low=-1, high=1, size=(n, n))
        diagonal = np.diag(A)
        if np.any(diagonal == 0):
            continue
        """for i in range(n):
            A[i, i] = 10"""
        if np.linalg.det(A) != 0:
            return A

def generate_tridioganal_matrix(n):
    while True:
        A = np.zeros((n, n), dtype=dtype)
        for i in range(n - 1):
            A[i, i + 1] = np.random.uniform(low=-1, high=1)
            A[i, i] = np.random.uniform(low=-1, high=1)
            A[i + 1, i] = np.random.uniform(low=-1, high=1)
        A[-1, -1] = np.random.uniform(low=-1, high=1)
        if np.linalg.det(A) != 0:
            return A

def generate_tridioganal_matrix_with_strict_diagonal_dominance(n):
    A = generate_tridioganal_matrix(n)
    for i in range(n):
        row_sum = sum(np.abs(A[i, j]) for j in range(n)) - np.abs(A[i, i])
        A[i, i] = np.random.uniform(low=0, high=1) + row_sum
    return A

def generate_pos_defined_matrix(n):
    while True:
        A = generate_square_matrix(n)
        A = (A + A.T) / 2
        for i in range(n):
            row_sum = sum(np.abs(A[i, j]) for j in range(n)) - np.abs(A[i, i])
            A[i, i] = np.random.uniform(low=0, high=1) + row_sum
        if np.all(np.linalg.eigvals(A) > 0):
            return A

def find_relative_errors(generate_method, solve_method, amount):
    b = np.array([1, 1, 1, 1, 1, 1], dtype=dtype)
    errors_L2 = np.array([], dtype=dtype)
    errors_L_inf = np.array([], dtype=dtype)
    conds = np.array([], dtype=dtype)
    spectral_radiuses = np.array([], dtype=dtype)
    eig_ratios = np.array([], dtype=dtype)
    for i in range(amount):
        A = generate_method(n=6)
        conds = np.append(conds, np.linalg.cond(A))
        spectral_radiuses = np.append(spectral_radiuses, np.max(np.abs(np.linalg.eigvals(A))))
        eig_ratios = np.append(eig_ratios, np.max(np.abs(np.linalg.eigvals(A))) / np.min(np.abs(np.linalg.eigvals(A))))
        x_default = solve_method(A, b)
        x_universal = gauss(A, b, True)
        errors_L2 = np.append(errors_L2, np.linalg.norm(x_default-x_universal) / np.linalg.norm(x_universal))
        errors_L_inf = np.append(errors_L_inf, np.linalg.norm(x_default - x_universal, ord=np.inf) / np.linalg.norm(x_universal, ord=np.inf))
    return errors_L2, errors_L_inf, conds, spectral_radiuses, eig_ratios

def plot(errors_L2, errors_L_inf, conds, spectral_radiuses, eig_ratios, title):
    values_dict = {
                    'относительные погрешности (среднеквадратичная норма)': errors_L2,
                    'относительные погрешности (супремум-норма)': errors_L_inf,
                    'числа обусловленности': conds,
                    'спектральные радиусы': spectral_radiuses,
                    'отношение макс. и мин. по модулю собственных чисел': eig_ratios
                  }
    for keys in values_dict.keys():
        plt.hist(values_dict[keys], bins=251, label=f'{title}. {keys}')
        rborder = max(values_dict[keys])
        lborder = min(values_dict[keys])
        adjust = (rborder - lborder) / 20
        plt.xlim(lborder - adjust, rborder - adjust)
        plt.legend()
        plt.show()



if __name__ == "__main__":
    dtype = np.float32
    amount = 1000
    errors_L2, errors_L_inf, conds, spectral_radiuses, eig_ratios = find_relative_errors(generate_square_matrix, gauss, amount)
    plot(errors_L2, errors_L_inf, conds, spectral_radiuses, eig_ratios, title='квадратные матрицы общего вида')
    errors_L2, errors_L_inf, conds, spectral_radiuses, eig_ratios = find_relative_errors(generate_tridioganal_matrix, thomas, amount)
    plot(errors_L2, errors_L_inf, conds, spectral_radiuses, eig_ratios, title='трехдиагональные матрицы')
    errors_L2, errors_L_inf, conds, spectral_radiuses, eig_ratios = find_relative_errors(generate_pos_defined_matrix, cholesky, amount)
    plot(errors_L2, errors_L_inf, conds, spectral_radiuses, eig_ratios, title='положительно определенные матрицы')
    errors_L2, errors_L_inf, conds, spectral_radiuses, eig_ratios = find_relative_errors(generate_tridioganal_matrix_with_strict_diagonal_dominance, thomas, amount)
    plot(errors_L2, errors_L_inf, conds, spectral_radiuses, eig_ratios, title='трехдиагональные матрицы со строгим диагональным преобладанием')
