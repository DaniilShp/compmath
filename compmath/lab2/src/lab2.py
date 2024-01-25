import math
import numpy as np
import matplotlib.pyplot as plt

def composite_simpson(a, b, n, f):
    if n % 2 != 0:
        raise ValueError

    x = np.linspace(a, b, n+1)
    h = (b-a)/n

    integral_value = h/3.*(f(x[0]) + f(x[-1]))
    integral_value += h/3.*(2*np.sum([f(x_i) for x_i in x[2:-1:2]]))
    integral_value += h/3.*(4*np.sum([f(x_i) for x_i in x[1::2]]))
    return integral_value

def composite_trapezoid(a, b, n, f):
    h = (b-a) / n
    integral_value = h / 2. * (f(a) + f(b))

    for i in range(2, n+1):
        integral_value += h / 2. * 2 * f(a + (i-1) * h)
    return integral_value

def functional(t):
    C = 1.03439984
    g = 9.80
    yt = lambda t: C * (0.5 - 0.5 * np.cos(2*t))
    dy_dx = lambda t: np.sin(2*t) / (1 - np.cos(2*t))
    dx_dt = lambda t: C * (1 - np.cos(2*t))
    return np.sqrt((1 + dy_dx(t) ** 2) / yt(t) / (2 * g)) * dx_dt(t)

def count_functional(method):
    values = [[], []]
    C = 1.03439984
    T = 1.75418438

    for n in range(4, 10000, 10):
        values[0].append(n)
        values[1].append(method(1e-7, 1.75418438, n, functional))
    return values

def count_function_simpson(start, end, exp):
    values = [[],[]]

    for n in range(4, 10000, 10):
        values[0].append(n)
        values[1].append(composite_simpson(start, end, n, exp))
    return values

def count_function_trapezoid(start, end, exp):
    values = [[], []]

    for n in range(3, 10000, 10):
        values[0].append(n)
        values[1].append(composite_trapezoid(start, end, n, exp))
    return values

def plot_inaccuracy(values, precise, color, label):
    integration_steps = []
    integral_values = []
    C = 1.03439984
    T = 1.75418438

    for i in range(len(values[0])):
        integration_steps.append(T/values[0][i])
        integral_values.append(abs(precise - values[1][i]))

    plt.loglog(integration_steps, integral_values, 'o', color=color, label=label)

def show_methods_inaccuracy():
    function = lambda x: math.exp(x)
    start_integr = 1
    end_integr = 5

    plt.title('Оценка порядка точности составных методов, f=exp(x)')
    plt.xlabel('h')
    plt.ylabel('error')

    values = count_function_trapezoid(start_integr, end_integr, function)
    plot_inaccuracy(values, precise=math.exp(5) - math.exp(1), color='red', label='composite trapezoid')

    values = count_function_simpson(start_integr, end_integr, function)
    plot_inaccuracy(values, precise=math.exp(5) - math.exp(1), color='blue', label='composite simpson')

    h_for_scaling = np.logspace(-2, 0, 50)
    plt.loglog(h_for_scaling, 1 / 12 * h_for_scaling ** 2, 'k-', label='2 порядок точности')
    plt.loglog(h_for_scaling, 1 / 180 * h_for_scaling ** 4, 'k--', label='4 порядок точности')

    plt.xlabel('шаг интегрирования')
    plt.ylabel('абсолютная погрешность')
    plt.legend()
    plt.savefig('base1.pdf')
    plt.show()

def get_nodes(x_t, y_t, a, b, N):
    t = np.linspace(a, b, N + 1)
    x_nodes = []
    y_nodes = []
    for t_i in t:
        x_nodes.append(x_t(t_i))
        y_nodes.append(y_t(t_i))
    return x_nodes, y_nodes

def get_coefs_linear_spline(x_nodes, y_nodes, N):
    coefs = [[], []]
    for j in range(N):
        slope = (y_nodes[j + 1] - y_nodes[j]) / (x_nodes[j + 1] - x_nodes[j])
        intercept = y_nodes[j]
        coefs[0].append(slope)
        coefs[1].append(intercept)
    return coefs

def functional_value_with_linear_spline(x_nodes, coefs_matrix, N, n_integr):
    integral_value_simpson = 0
    integral_value_F = 0
    def create_linear_f(slope, intercept, x_i):
        linear_f = lambda x: slope * (x-x_i) + intercept
        linear_F = lambda x: slope / 2 * (x-x_i)**2 + intercept * x
        return linear_f, linear_F
    for i in range(N-1):
        f, F = create_linear_f(coefs_matrix[0][i], coefs_matrix[1][i], x_nodes[i])
        integral_value_simpson += composite_simpson(x_nodes[i], x_nodes[i+1], n_integr, f)
        integral_value_F += F(x_nodes[i+1])-F(x_nodes[i])
    return integral_value_simpson, integral_value_F

def plot_linear_spline(x_nodes, coefs_matrix, N):
    for i in range(N):
        x = np.linspace(x_nodes[i], x_nodes[i + 1], 2)
        y = coefs_matrix[0][i] * (x - x_nodes[i]) + coefs_matrix[1][i]
        plt.plot(x, y)
    plt.title('кусочно-линейная интерполяция кривой наискорейшего спуска')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().invert_yaxis()
    plt.savefig('advanced.pdf')
    plt.show()

def lab2_base():

    show_methods_inaccuracy()

    values = count_functional(composite_trapezoid)
    plot_inaccuracy(values, precise = math.sqrt(2*C/9.8)*T, color='red', label='composite trapezoid')

    values = count_functional(composite_simpson)
    plot_inaccuracy(values, precise = math.sqrt(2*C/9.8)*T, color='blue', label='composite simpson')

    h_for_scaling = np.logspace(-3, -1, 50)
    plt.loglog(h_for_scaling, h_for_scaling/1000, 'k-', label='1 порядок точности')

    plt.title('Оценка порядка точности составных методов, f=F[y]')
    plt.xlabel('шаг интегрирования')
    plt.ylabel('абсолютная погрешность')
    plt.legend()
    plt.savefig('base2.pdf')
    plt.show()

def lab2_advanced():
    N = 50
    n_integr = 2
    x_t = lambda t: (t - 0.5 * math.sin(2 * t))*C
    y_t = lambda t: (0.5 - 0.5 * math.cos(2 * t))*C
    x_nodes, y_nodes = get_nodes(x_t, y_t, 0.0, T, N)
    coefs = get_coefs_linear_spline(x_nodes, y_nodes, N)
    plot_linear_spline(x_nodes, coefs, N)
    print(functional_value_with_linear_spline(x_nodes, coefs, N, n_integr))

if __name__ == "__main__":
    C = 1.03439984
    T = 1.75418438
    lab2_base()
    lab2_advanced()