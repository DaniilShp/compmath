import numpy as np
import matplotlib.pyplot as plt
import math

def create_sparse_matrix(filename, M): #построение разреженного множества интерполяционных узлов t, x, y
    sparse_matrix = [[], [], []]       # t целые числа с шагом 1
    t = 0
    j = 0
    with open(filename, 'r') as file:
        for line in file:
            columns = line.split()
            if len(columns) > 1 and j % M == 0:
                sparse_matrix[0].append(int(t))
                sparse_matrix[1].append(float(columns[0]))
                sparse_matrix[2].append(float(columns[1]))
                t+=1
            j+=1
    return sparse_matrix

def get_matrix_A(sparse_matrix):   #функция построения матрицы А использующейся для нахождения коэффициентов через матричное уравнение
    n = len(sparse_matrix[0])
    matrix_A = np.zeros((n, n)) #функция принимает разреженную матрицу координат (х,t) или (у,t)
    matrix_A[0][0] = 1
    matrix_A[n-1][n-1] = 1
    for i in range(1, n-1):
        j = i - 1 
        matrix_A[i][j] = int(sparse_matrix[0][j+1]-sparse_matrix[0][j])
        matrix_A[i][j+1] = 2 * (sparse_matrix[0][j+2]-sparse_matrix[0][j])
        matrix_A[i][j+2] = sparse_matrix[0][j+2]-sparse_matrix[0][j+1]
    return matrix_A

def get_matrix_b(sparse_matrix):
    n = len(sparse_matrix[0])
    matrix_b = np.zeros(n)
    for i in range(1, n-1):
        matrix_b[i] = 3 * (sparse_matrix[1][i+1] - sparse_matrix[1][i]) / (sparse_matrix[0][i+1] - sparse_matrix[0][i]) - 3 * (sparse_matrix[1][i] - sparse_matrix[1][i-1]) / (sparse_matrix[0][i] - sparse_matrix[0][i-1])
        matrix_b[0] = 0
        matrix_b[n-1] = 0
    return matrix_b

def get_coefs_spline(sparse_matrix, c): # a b c d коэффициенты кубического сплайна
    n = len(c)
    coefs = np.zeros((n, 4))
    for i in range(0, n):
        coefs[i][0] = sparse_matrix[1][i] #коэффициент а
    for i in range(0,n-1):
        h_i = (sparse_matrix[0][i+1]-sparse_matrix[0][i])
        coefs[i][1] = (coefs[i+1][0]-coefs[i][0]) / h_i - h_i / 3 * (c[i+1] + 2 * c[i]) #коэффициент b
        coefs[i][2] = c[i] #коэффициент с
        coefs[i][3] = (c[i+1] - c[i]) / (3 * h_i) #коэффициент d
    return coefs[:-1]

def gauss_method(A, b): #метод Гаусса
    n = len(A)
    for i in range(n):
        max_row = i
        for j in range(i+1, n):
            if abs(A[j][i]) > abs(A[max_row][i]):
                max_row = j
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        for j in range(i+1, n):
            ratio = A[j][i] / A[i][i]
            A[j] -= ratio * A[i]
            b[j] -= ratio * b[i]
    decision = np.zeros(n)
    for i in range(n-1, -1, -1):
        decision[i] = (b[i] - np.dot(A[i][i+1:], decision[i+1:])) / A[i][i]
    return decision

def show_cube_splines(xcoefs, ycoefs, M): 
    n = len(xcoefs)  # Получаем количество интервалов
    spline_coordinates = [[], []]
    for i in range(n):
        a = xcoefs[i][0] #извлекаем коэффициенты
        b = xcoefs[i][1]
        c = xcoefs[i][2]
        d = xcoefs[i][3]
        t = np.linspace(i, i+1, M)  #(для частоты шага h = 0.1 и M = 10 delta(t) = 1) берем 10 точек из каждого интервала
        x = a + b * (t - i) + c * (t - i) ** 2 + d * (t - i) ** 3  # Вычисляем значения кубического сплайна при некоторых t из интервала
        for j in range(0, M):
            spline_coordinates[0].append(x[j])
        a = ycoefs[i][0]
        b = ycoefs[i][1]
        c = ycoefs[i][2]
        d = ycoefs[i][3]
        y = a + b * (t - i) + c * (t - i) ** 2 + d * (t - i) ** 3
        for j in range(0, M):
            spline_coordinates[1].append(y[j])
            
    plt.plot(spline_coordinates[0], spline_coordinates[1], 'gray')
    plt.title('кубические сплайны')
    plt.axis('equal')
    return spline_coordinates

def plot_mandelbrot_points(filename):
    mandelbrot_points = [[], []]
    with open(filename, 'r') as file:
        for line in file:
            columns = line.split()
            if len(columns) > 0:
                mandelbrot_points[0].append(float(columns[0]))
                mandelbrot_points[1].append(float(columns[1])) 
        plt.scatter(mandelbrot_points[0], mandelbrot_points[1], 0.5, 'purple' )
        plt.title('визуализация множества Мандельброта')
        return mandelbrot_points

def find_distance(point_coordinates_spline, point_coordinates_mandelbrot):
    n = len(point_coordinates_spline[0])
    distances = np.zeros(n)
    for i in range(0, n):
        x1, y1 = point_coordinates_spline[0][i], point_coordinates_spline[1][i]
        x2, y2 = point_coordinates_mandelbrot[0][i], point_coordinates_mandelbrot[1][i]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances[i] = distance
    return distances

def lab1_base(filename_in, factor, filename_out):
    plt.subplot(2, 1, 1)
    plt.axis('equal')
    mandelbrot_points = plot_mandelbrot_points(filename_in)
    sparse_matrix = create_sparse_matrix(filename_in, factor)
    x = [sparse_matrix[0], sparse_matrix[1]]
    y = [sparse_matrix[0], sparse_matrix[2]]
    A_x = get_matrix_A(x)
    A_y = get_matrix_A(y)
    b_x = get_matrix_b(x)
    b_y = get_matrix_b(y)
    c_x = gauss_method(A_x, b_x)
    c_y = gauss_method(A_y, b_y)
    xcoefs = get_coefs_spline(x, c_x)
    ycoefs = get_coefs_spline(y, c_y)
    coefs = np.concatenate((xcoefs, ycoefs), 1)
    np.savetxt(filename_out, coefs, delimiter=' ')
    plt.subplot(2, 1, 2)
    spline_points = show_cube_splines(xcoefs, ycoefs, factor)
    distances = find_distance(spline_points, mandelbrot_points)
    print(f"среднее расстояние: {np.mean(distances)}")
    print(f"стандартное отклонение: {np.mean(np.std(distances))}")
    plt.show()

class DualNum:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual
    
    def __add__(self, other):
        if isinstance(other, DualNum):
            return DualNum(self.real + other.real, self.dual + other.dual)
        else:
            return DualNum(self.real + other, self.dual)

    def __sub__(self, other):
        if isinstance(other, DualNum):
            return DualNum(self.real - other.real, self.dual - other.dual)
        else:
            return DualNum(self.real - other, self.dual)

    def __mul__(self, other):
        if isinstance(other, DualNum):
            return DualNum(self.real * other.real, self.real * other.dual + self.dual * other.real)
        else:
            return DualNum(self.real * other, self.dual * other)

    def __pow__(self, power):
        if isinstance(power, int):
            return DualNum(self.real ** power, power * self.real ** (power - 1) * self.dual)
        else:
            raise ValueError("степень должна быть целой")
    
    def __str__(self):
        return f"{self.real} + ε{self.dual}"

def calculate_derivative(coefficients, x, x_i):
    x_dual = DualNum(x, 1)
    result = DualNum(coefficients[0], 0)
    for i in range(1, len(coefficients)):
        result += DualNum(coefficients[i], 0) * (x_dual - DualNum(x_i, 0)) ** i
    return result.dual

def get_vector_derivatives(filename, xcoefs, ycoefs, derivatives):
    t = 0
    i = 0
    with open(filename, 'r') as file:
        for line in file:
            columns = line.split()
            if len(columns) > 7:
                xcoefs.append([float(columns[0]), float(columns[1]), float(columns[2]), float(columns[3])])
                derivatives[0].append(calculate_derivative(xcoefs[i], t, t))
                ycoefs.append([float(columns[4]), float(columns[5]), float(columns[6]), float(columns[7])])
                derivatives[1].append(calculate_derivative(ycoefs[i], t, t))
                t+=1.0
                i+=1
    return derivatives

def plot_vector_and_normal(xcoefs, ycoefs, xderivatives, yderivatives):
    n = len(xderivatives)
    for i in range (1,n, 10):
        a = xcoefs[i][0] #извлекаем коэффициенты
        b = xcoefs[i][1]
        c = xcoefs[i][2]
        d = xcoefs[i][3]
        t0 = i
        t = np.linspace(t0-1, t0+1, 2)
        x = a + 1*xderivatives[i]*(t-t0)
        a = ycoefs[i][0]
        b = ycoefs[i][1]
        c = ycoefs[i][2]
        d = ycoefs[i][3]
        y = a + 1*yderivatives[i]*(t-t0)
        perpendicular_x, perpendicular_y = find_normal(x, y)

        plt.plot(perpendicular_x, perpendicular_y, 'green')
        plt.plot(x, y, 'red')

def find_normal(x, y):
    perpendicular_x = [0, 0]
    perpendicular_y = [0, 0]
    centre_x = x[0] + (x[1]-x[0])/2
    centre_y = y[0] + (y[1]-y[0])/2
    k = (y[1] - y[0]) / (x[1] - x[0]) #вычисление наклона касательной
    rnd_x = 0.01
    x2 = centre_x + rnd_x #случайный x
    y2 = centre_y - 1 / k * (x2-centre_x)
    maxsize = 0.0000000002 # максимальный размер перпендикуляра
    while math.sqrt((x2-centre_x)**2 + (y2-centre_y)**2) > maxsize:
        rnd_x /= 2
        x2 = centre_x + rnd_x
        y2 = centre_y - 1 / k * rnd_x
    perpendicular_x[0], perpendicular_x[1] = centre_x, x2
    perpendicular_y[0], perpendicular_y[1] = centre_y, y2
    return perpendicular_x, perpendicular_y

def lab1_advanced(filename_in, factor):
    derivatives = [[], []]
    xcoefs = []
    ycoefs = []
    derivatives = get_vector_derivatives(filename_in, xcoefs, ycoefs, derivatives)
    show_cube_splines(xcoefs, ycoefs, factor)
    plot_vector_and_normal(xcoefs, ycoefs, derivatives[0], derivatives[1])
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    M = 10
    print("базовая часть")
    lab1_base("contour.txt", M, "coefs.txt")
    print("продвинутая часть")
    lab1_advanced("coefs.txt", M)
