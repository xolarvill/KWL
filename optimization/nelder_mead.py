import numpy as np

def nelder_mead(func, x_start,
                step=0.1, tol=1e-6,
                no_improve_break=10, max_iter=1000,
                alpha=1., gamma=2., beta=0.5, delta=0.5):
    """
    Nelder-Mead algorithm to minimize a function.

    Parameters:
    func : callable
        The objective function to be minimized.目标函数，接收一个向量并返回一个标量
    x_start : ndarray
        Initial guess.初始点，应该是一个包含目标函数维度的向量
    step : float
        Look-around radius in initial step.
    tol : float
        Break after no_improvement_thr iterations with no improvement.收敛容忍度
    no_improve_break : int
        Number of iterations with no improvement to break after.
    max_iter : int
        Maximum number of iterations to perform.最大迭代次数
    alpha : float
        Reflection coefficient.反射系数
    gamma : float
        Expansion coefficient.扩展系数
    beta : float
        Contraction coefficient.收缩系数
    delta : float
        Shrink coefficient.缩小系数

    Returns:
    best : ndarray
        The point which minimizes the function.
    """

    # Initial simplex
    dim = len(x_start)
    prev_best = func(x_start)
    no_improve = 0
    res = [[x_start, prev_best]]
    for i in range(dim):
        x = np.copy(x_start)
        x[i] = x[i] + step
        score = func(x)
        res.append([x, score])
    res.sort(key=lambda x: x[1])
    best = res[0][1]

    iterations = 0
    while True:
        if max_iter and iterations >= max_iter:
            break
        iterations += 1

        # Order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        if best < prev_best - tol:
            no_improve = 0
            prev_best = best
        else:
            no_improve += 1

        if no_improve >= no_improve_break:
            break

        # Centroid
        x0 = np.zeros(dim)
        for tup in res[:-1]:
            x0 += tup[0]
        x0 /= (len(res) - 1)

        # Reflection
        xr = x0 + alpha * (x0 - res[-1][0])
        rscore = func(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # Expansion
        if rscore < res[0][1]:
            xe = x0 + gamma * (x0 - res[-1][0])
            escore = func(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # Contraction
        xc = x0 + beta * (x0 - res[-1][0])
        cscore = func(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # Reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + delta * (tup[0] - x1)
            score = func(redx)
            nres.append([redx, score])
        res = nres

    return res[0][0]

# Example usage of the nelder_mead function
    

def example_func(x):
    return x[0]**2 + x[1]**2 + 1

initial_guess = np.array([1.0, 1.0])
result = nelder_mead(example_func, initial_guess)
print("Optimized parameters:", result)
print("Function value at optimized parameters:", example_func(result))