import matplotlib.pyplot as plt
import numpy as np

def plot_solution_and_target(solution, target_func_name, dataset):
    fig, ax = plt.subplots()
    interval = np.arange(dataset[0, 0], dataset[-1, 0], 0.1)
    solution_vals = list(map(lambda x: solution.compute_tree([x]), interval))
    ax.scatter(dataset[:, 0], dataset[:, 1])
    ax.plot(interval, solution_vals)
    ax.legend(['best solution', 'target function'])
    ax.set(xlabel='input', ylabel='output', title=target_func_name)
    plt.show()

def simple_line_plot(xs, ys, xlabel='', ylabel='', title='', label=''):
    plt.plot(xs, ys, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.show()

def target_with_predictor(dataset, solution, predictor_idxs):
    plt.plot(dataset[:, 0], dataset[:, 1])
    solution_vals = list(map(lambda x: solution.compute_tree([x]), dataset[:, 0]))
    plt.plot(dataset[:, 0], solution_vals, color='red')
    plt.scatter(dataset[predictor_idxs, 0], dataset[predictor_idxs, 1], color='red')
