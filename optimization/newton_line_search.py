import numpy as np
from scipy.linalg import lu

def newton(func, initial_guess, tolerance=1e-6, max_iter=100):
    """
    使用Newton法求解目标函数的最小化问题
    当前代码是用来最小化一个目标函数的，但最大化的过程可以通过最小化负的目标函数来实现
    
    输入:
    func (callable): 目标函数
    initial_guess (numpy.array): 初始猜测
    tolerance (float): 容忍度（收敛标准）
    max_iter (float): 最大迭代次数
    
    输出:
    x (np.array)：最优解
    step_size (float)：最优步长
    iter_count (float)：迭代次数
    se (np.array)：标准误向量
    """
    # 将初始猜测值转换为NumPy数组 
    x = np.array(initial_guess)  
    
    # 使用LU分解求解线性方程组 A * x = b，返回解 x
    def lu_solve(A, b):
        # LU分解
        P, L, U = lu(A)
        # 解 L * y = b
        y = np.linalg.solve(L, np.dot(P.T, b))  # 先通过前向替代求解 y
        # 解 U * x = y
        x = np.linalg.solve(U, y)  # 再通过后向替代求解 x
        return x
    
    # 计算目标函数的梯度
    def gradient(func, x, h=1e-6):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x1 = np.copy(x)
            x2 = np.copy(x)
            x1[i] += h
            x2[i] -= h
            grad[i] = (func(x1) - func(x2)) / (2 * h)
        return grad
    
    # 计算目标函数的海塞矩阵
    def hessian(func, x, h=1e-6):
        n = len(x)
        hessian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_ij = np.copy(x)
                x_ij[i] += h
                x_ij[j] += h
                f_ij = func(x_ij)
                
                x_i = np.copy(x)
                x_i[i] += h
                f_i = func(x_i)
                
                x_j = np.copy(x)
                x_j[j] += h
                f_j = func(x_j)
                
                f = func(x)
                
                hessian[i, j] = (f_ij - f_i - f_j + f) / (h ** 2)
        return hessian
      
    # 计算标准误，在log likelihood函数中，Fisher信息矩阵和负的海塞矩阵是相等的
    def standard_error(hessian_matrix):
        try:
            # Fisher是参数不确定性的下界
            fisher_info = -hessian_matrix
            # 协方差矩阵是Fisher信息矩阵的逆，反映参数估计的不确定性
            covariance_matrix = np.linalg.inv(fisher_info)
            # 协方差矩阵的对角元素的平方根就是标准误
            standard_errors = np.sqrt(np.diag(covariance_matrix))
            return standard_errors
        except np.linalg.LinAlgError:
            raise ValueError("Hessian matrix is not invertible. Check the likelihood function or data.")
    
    # 主迭代
    for iter_count in range(max_iter):
        grad = gradient(func, x)
        hess = hessian(func, x)

        # 使用LU分解方法求解 H(x) * delta_x = -grad
        delta_x = lu_solve(hess, -grad)
        
        # 更新 x
        x_new = x + delta_x
        
        # 计算步长
        step_size = np.linalg.norm(delta_x)
        
        # 检查是否收敛
        if step_size < tolerance:
            se = standard_error(hess)
            return x_new, step_size, iter_count, se
        
        # 更新 x
        x = x_new
        
    return x, step_size, iter_count + 1, se
    

# 示例：测试该函数
if __name__ == "__main__":
    def func(x):
        return np.sum(x**2) + 3 * np.sum(x) + 2
    
    initial_guess = np.array([5.0])  # 初始猜测值
    tolerance = 1e-5  # 容忍度
    max_iter = 100  # 最大迭代次数
    
    # 调用newton方法
    optimal_solution, optimal_step, num_iterations, standard_errors = newton(func, initial_guess, tolerance, max_iter)
    
    print(f"最优解x*={optimal_solution}")
    print(f"最优步长：{optimal_step}")
    print(f"迭代次数：{num_iterations}")
    print(f"标准误：{standard_errors}")

