import matplotlib.pyplot as plt
import numpy as np

def plot_solution_and_target(solution, target_func_name, dataset):
    fig, ax = plt.subplots()
    interval = np.linspace(dataset[0, 0], dataset[-1, 0], 1000)
    solution_vals = list(map(lambda x: solution.compute_tree(x), interval))
    ax.scatter(dataset[:, 0], dataset[:, 1])
    ax.plot(interval, solution_vals)
    ax.legend(['best solution', 'target function'])
    ax.set(xlabel='input', ylabel='output', title=target_func_name)
    plt.show()

def simple_line_plot(xs, ys, xlabel='', ylabel='', title=''):
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.show()
