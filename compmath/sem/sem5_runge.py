import math
import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return (1 + math.exp(x/y))/ (math.exp(x/y)*(x/y-1))

def runge_kutta_4(f, x0, y0, h, n):
    x = x0
    y = y0
    result = [y0]
    nodes = [x0]
    for i in range(n):
        #
        print(i)
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        #print(k1, k2, k3, k4, sep=' ')
        #print(x, x + h/2, x+h/2, x+h)
        #print(y, y + k1/2, y + k2/2, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        x = x + h
        nodes.append(x)
        result.append(y)
    
    return result, nodes

def plot_precise_decision(interval_start, interval_end):
    x = np.linspace(interval_start, interval_end, 100)
    y = np.linspace(0.1, 10, 100)

    X, Y = np.meshgrid(x, y)
    Z = Y * np.exp(X / Y) + X - 1

    plt.contour(X, Y, Z, [0], colors='blue') 


interval_start = 0
interval_end = 0.3

plot_equation(interval_start, interval_end)
    
x0 = 0  
y0 = 1  
h = 0.05
n = 6

result, nodes = runge_kutta_4(f, x0, y0, h, n)
print(result)
print(nodes)
plt.scatter(nodes, result, color='red', s = 10, label='f(t_i, w(t_i))')
plt.axis('equal')
plt.title('Метод Рунге-Кутты 4 порядка')
plt.legend()
plt.show()
