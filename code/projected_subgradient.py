import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import linprog


def random_matrix(m, n):
    a = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            a[i][j] = random.random() * 1000
    return a


def projected_subgradient(A, b, n, max_iter):
    proj = np.identity(n) - np.matmul(np.matmul(A.T, np.linalg.inv(np.matmul(A, A.T))), A)

    objective = lambda x: np.linalg.norm(x, ord=1)

    f_best_arr = []
    x_arr = []

    x = np.matmul(np.matmul(A.T, np.linalg.inv(np.matmul(A, A.T))), b)
    f_best = objective(x)

    x_arr.append(x)
    f_best_arr.append(f_best)
    for k in range(1, max_iter):
        subgrad = np.sign(x)
        stepsize = 100 / (k * np.sqrt(n) * np.linalg.norm(subgrad, ord=2))
        x -= stepsize * np.matmul(proj, subgrad)
        f = objective(x)
        if f < f_best:
            f_best = f
        x_arr.append(x)
        f_best_arr.append(f_best)
    return (f_best_arr, x_arr)


def linear_programming(A, b, n):
    c = [1] * (2 * n)
    A_eq = np.concatenate((A, -A), axis=1)
    b_eq = b
    bounds = [(0, None)] * (2 * n)
    return linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

n = 1000
m = 500
A = random_matrix(m, n)
b = random_matrix(m, 1)
(f_best_arr, x_arr) = projected_subgradient(A, b, n, 1000)
iterations = len(f_best_arr)
res = linear_programming(A, b, n)
plt.figure()
plt.plot(list(range(1, iterations + 1)), f_best_arr)
plt.plot(list(range(1, iterations + 1)), [res.fun] * iterations)
plt.legend(["Objective", "Optimal Value"])
plt.xlabel("Iterations")
plt.title("Projected Subgradient Method")
plt.show()

