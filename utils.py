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
