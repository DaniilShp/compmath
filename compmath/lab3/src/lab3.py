import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import root
import timeit

def euler_method(init_0, t_n, f, h):
    num_steps = int(t_n / h)
    t = np.linspace(0, t_n, num_steps + 1)
    x = np.zeros(((num_steps + 1), 2))
    x[0] = init_0
    for i in range(num_steps):
        x[i+1] = x[i] + h * f(t[i], x[i])

    return x, t

def adams_moulton(init_0, t_n, f, h):
    num_steps = int(t_n / h)
    t = np.linspace(0, t_n, num_steps + 1)
    x = np.zeros(((num_steps + 1), 2))
    initial_conditions, _ = runge_kutta(init_0, 3*h, f, h)
    for i in range(3):
        x[i] = initial_conditions[i]
    for i in range(2, num_steps):
        #x[i+1] = x[i] + h/24. * (9. * f(t[i+1], x[i+1]) + 19. * f(t[i], x[i]) + 5*f(t[i-1], x[i-1]) + f(t[i-2], x[i-2]))
        def equation(x_i_plus_1):
            return x[i] + h/24. * (9. * f(t[i+1], x_i_plus_1) + 19. * f(t[i], x[i]) - 5. * f(t[i-1], x[i-1]) + f(t[i-2], x[i-2])) - x_i_plus_1
        solution = root(equation, init_0)
        x[i+1] = solution.x
    return x, t


def milne_simpson(init_0, t_n, f, h):
    num_steps = int(t_n / h)
    t = np.linspace(0, t_n, num_steps + 1)
    x = np.zeros(((num_steps + 1), 2))
    x_approx_by_milne = np.zeros(((num_steps + 1), 2))
    initial_conditions, _ = runge_kutta(init_0, 3*h, f, h)
    x[0] = initial_conditions[0]
    x[1] = initial_conditions[1]
    for i in range(4):
        x_approx_by_milne[i] = initial_conditions[i]

    for i in range(1, 3):
        x[i+1] = x[i-1] + h/3. * (f(t[i-1], x[i-1]) + 4*f(t[i], x[i]) + f(t[i+1], x_approx_by_milne[i+1]))

    for i in range(3, num_steps):
        x_approx_by_milne[i+1] = x[i-3] + 4*h/3. * (2*f(t[i], x[i])-f(t[i-1], x[i-1]) + 2*f(t[i-2], x[i-2]))
        x[i+1] = x[i-1] + h/3. * (f(t[i-1], x[i-1]) + 4*f(t[i], x[i]) + f(t[i+1], x_approx_by_milne[i+1]))

    return x, t

def runge_kutta(init_0, t_n, f, h):
    num_steps = int(t_n / h)
    t = np.linspace(0, t_n, num_steps + 1)
    x = np.zeros(((num_steps + 1), 2))
    x[0] = init_0

    for i in range(num_steps):
        k1 = h * f(t[i], x[i])
        k2 = h * f(t[i] + h/2, x[i] + k1/2)
        k3 = h * f(t[i] + h/2, x[i] + k2/2)
        k4 = h * f(t[i] + h, x[i] + k3)
        x[i+1] = x[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x, t

def f(t, x):
    return np.array([x[1], -0.1 * x[1] - np.sin(x[0]) + np.cos(t)])

def plot_runge_kutta(x_0_arr):
    fig = plt.figure()
    fig.canvas.manager.window.setGeometry(420, 30, 700, 250)
    plt.title('траектории по методу Рунге-Кутта')
    plt.xlabel('t')
    plt.ylabel('θ')
    plt.gca().xaxis.set_label_coords(1.05, 0.0)
    for x_0 in x_0_arr:
        x, t = runge_kutta(x_0, t_n, f, h)
        y = []
        for i in range(len(x)):
            y.append(x[i][0])
        plt.plot(t, y, linewidth=1)

def plot_method_instability(method, h_arr):
    plt.title(f'Метод {method.__name__}')
    plt.xlabel('t')
    plt.ylabel('θ')
    x_0 = [0, 1.85]
    for h in h_arr:
        x, t = method(x_0, t_n, f, round(h, 3))
        y = []
        for i in range(len(x)):
            y.append(x[i][0])
        plt.plot(t, y, linewidth=1, label=f"h={round(h, 3)}")
        plt.legend()

def plot_milne_simpson(x_0_arr):
    fig = plt.figure()
    fig.canvas.manager.window.setGeometry(420, 310, 700, 250)
    plt.title('траектории по методу Милна-Симпсона')
    plt.xlabel('t')
    plt.ylabel('θ')
    plt.gca().xaxis.set_label_coords(1.05, 0.0)
    for x_0 in x_0_arr:
        x, t = milne_simpson(x_0, t_n, f, h)
        y = []
        for i in range(len(x)):
            y.append(x[i][0])
        plt.plot(t, y, linewidth=1)

def plot_adams_moulton(x_0_arr):
    fig = plt.figure()
    fig.canvas.manager.window.setGeometry(420, 590, 700, 250)
    plt.title('траектории по методу Адамса-Моултона')
    plt.xlabel('t')
    plt.ylabel('θ', rotation=90)
    plt.gca().xaxis.set_label_coords(1.05, 0.0)
    for x_0 in x_0_arr:
        x, t = adams_moulton(x_0, t_n, f, h)
        y = []
        for i in range(len(x)):
            y.append(x[i][0])
        plt.plot(t, y, linewidth=1)

def plot_phase_tracks(x_0_arr, method):
    plt.title(f'фазовая траектория по методу {method.__name__}')
    plt.xlabel('θ')
    plt.ylabel('dθ/dt')
    for x_0 in x_0_arr:
        x, t = method(x_0, t_n, f, h)
        y = []
        dy = []
        for i in range(len(x)):
            y.append(x[i][0])
            dy.append(x[i][1])
        plt.plot(y, dy, linewidth=1, label=f"x_0: {round(x_0[0], 2)};{round(x_0[1], 2)}")
        plt.legend(bbox_to_anchor=(1, 1))

def plot_phase_track_via_all_methods():
    plt.title(f'фазовые траектории')
    plt.xlabel('θ')
    plt.ylabel('dθ/dt')
    x = [[],[],[]]
    x_0 = [0, 2.06]
    x[0], t = runge_kutta(x_0, t_n, f, h)
    x[1], _ = adams_moulton(x_0, t_n, f, h)
    x[2], _ = milne_simpson(x_0, t_n, f, h)
    y = [[],[],[]]
    dy = [[],[],[]]
    titles = ["Рунге-Кутта", "Адамса-Моултона", "Милна-Симпсона"]
    for j in range(3):
        for i in range(len(t)):
            y[j].append(x[j][i][0])
            dy[j].append(x[j][i][1])
        plt.plot(y[j], dy[j], linewidth=1, label=f"{titles[j]}")
        plt.legend()

def meusure_execution_time():
    x_0 = [0, 2.]
    x_0_arr = [[0, 1.86], [0, 1.99], [0, 2.04]]
    t_n = 500
    h = [0.1, 0.075, 0.05]
    exec_time = [[],[],[]]
    for i in range(len(h)):
        start_time = timeit.default_timer()
        runge_kutta(x_0, t_n, f, h[i])
        exec_time[i].append(round(timeit.default_timer() - start_time, 3))
        start_time = timeit.default_timer()
        adams_moulton(x_0, t_n, f, h[i])
        exec_time[i].append(round(timeit.default_timer() - start_time, 3))
        start_time = timeit.default_timer()
        milne_simpson(x_0, t_n, f, h[i])
        exec_time[i].append(round(timeit.default_timer() - start_time, 3))
    print(f"time, ms: [h][method]: {exec_time}")

def asymptotic_states():
    x_0_arr = [[0, 2.06]]
    t_n = 70
    plot_phase_tracks(x_0_arr, runge_kutta)
    plt.show()
    t_n = 1000
    plot_phase_tracks(x_0_arr, runge_kutta)
    plt.show()
    plt.title(f'фазовая траектория для x_0 = {x_0_arr[0]} при t [100сек, 1000cек]')
    plt.xlabel('θ')
    plt.ylabel('dθ/dt')
    for x_0 in x_0_arr:
        x, t = runge_kutta(x_0, t_n, f, h)
        y, dy = [], []
        for i in range(len(x)):
            if i > 1000:
                y.append(x[i][0])
                dy.append(x[i][1])
        plt.plot(y, dy, linewidth=1)
        plt.show()


def lab3_base():
    plot_runge_kutta(random_initial_start_values)
    plt.savefig('plot_runge_kutta.pdf', format='pdf')
    plot_milne_simpson(random_initial_start_values)
    plt.savefig('plot_milne_simpson.pdf', format='pdf')
    plot_adams_moulton(random_initial_start_values)
    plt.savefig('plot_adams_moulton.pdf', format='pdf')
    plt.show()

    plot_method_instability(runge_kutta, h_arr=np.linspace(0.2, 1.2, 9))
    plt.savefig('runge_kutta_instability.pdf', format='pdf')
    plt.show()
    plot_method_instability(adams_moulton, h_arr=np.linspace(0.2, 1.2, 9))
    plt.savefig('adams_moulton_instability.pdf', format='pdf')
    plt.show()
    plot_method_instability(milne_simpson, h_arr=np.linspace(0.1, 0.4, 12))
    plt.savefig('milne_simpson_instability.pdf', format='pdf')

    plt.show()

def lab3_advanced():
    plot_phase_tracks(random_initial_start_values, runge_kutta)
    plt.savefig('runge_kutta_phase_tracks.pdf', format='pdf')
    plt.show()
    plot_phase_tracks(random_initial_start_values, adams_moulton)
    plt.savefig('adams_moulton_phase_tracks.pdf', format='pdf')
    plt.show()
    plot_phase_tracks(random_initial_start_values, milne_simpson)
    plt.savefig('milne_simpson_phase_tracks.pdf', format='pdf')
    plt.show()

    plot_phase_track_via_all_methods()
    plt.show()

    meusure_execution_time()

    asymptotic_states()



if __name__ == '__main__':
    t_n = 100
    h = .1
    random_initial_start_values = np.zeros((15, 2))
    random_initial_start_values[0] = [0, 1.85]
    for i in range(1, 15):
        random_initial_start_values[i] = 0, random.uniform(1.85, 2.1)
    lab3_base()
    lab3_advanced()





