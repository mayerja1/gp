import numpy as np
from functools import reduce

def binary_search(l, v, a, b):
    if a > b:
        return b, a
    mid = a + (b - a) // 2
    if l[mid] == v:
        return mid, mid
    elif l[mid] > v:
        return binary_search(l, v, a, mid - 1)
    return binary_search(l, v, mid + 1, b)

def linear_interpolation(x1, y1, x2, y2, x):
    return y1 + (x - x1) / (x2 - x1) * (y2 - y1)

def prepare_data(x_axis, y_axis):
    xs = sorted(set(reduce(lambda a, b: a + b, [list(x) for x in x_axis])))
    ys = np.zeros((len(x_axis), len(xs)))

    for i, x in enumerate(x_axis):
        y = y_axis[i]
        for j, xx in enumerate(xs):
            a, b = binary_search(x, xx, 0, len(x) - 1)
            if a == b:
                ys[i, j] = y[a]
            elif b < len(x) and a >= 0:
                ys[i, j] = linear_interpolation(x[a], y[a], x[b], y[b], xx)
            elif b >= len(x):
                ys[i, j] = y[-1]
            else:
                ys[i, j] = y[0]
    return xs, ys

if __name__ == '__main__':
    print(binary_search([1,3,6,7,10,11], -10, 0, 5))
